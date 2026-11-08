"""Cross-role audit of the manipulation checks. EXPLORATORY.

Every number in Sections 5 and 6 of the AES Fall 2026 poster and the audit
section of the extended abstract comes from this script. None of it requires a
new call: it re-scores the shipped check records. All of it is exploratory,
because those records were collected and inspected under A2/A3 before any of
these statistics were computed. The registered verdict (A2.2 as amended by
A3.2) is unchanged by anything here.

1. ``matrix`` -- for each operation prompt *i*, the paired d of check *j* on
   role *i*'s operation records against role *i*'s neutral records. The
   diagonal reproduces ``manipulation_report.json``.

2. ``discriminant`` -- the margin criterion of draft amendment A4.6: a role is
   specific if its diagonal d exceeds the largest absolute off-diagonal d in its
   row by at least ``MARGIN``. The criterion was fixed in
   ``logistics/A4_DRAFT.frozen-2026-10-25T141552Z.md`` (SHA-256 in
   ``logistics/A4_DRAFT.FREEZE.sha256``) before the matrix was first computed.

3. ``signed_discriminant`` -- the same margin counting only off-diagonal
   effects in the same direction as the diagonal, so that an opposite-direction
   effect (for example, spending the output budget on rebuttals rather than
   hypotheses) is not scored as non-specificity. Added after the AES review;
   a sensitivity analysis, not a replacement for (2).

4. ``claim_count`` -- Pearson r of every check, and of output tokens, against
   the number of atomic claims in the record, over all records (every role and
   variant). Three checks are counts over the same claim list, so part of this
   is structural rather than an effect of any prompt.

5. ``persona_vs_neutral`` -- the bare role-name prompt against the neutral
   prompt on each role's own check, with Benjamini-Hochberg across the six roles.

6. ``guard_terms`` -- the observed values of the quantities the guards depend
   on: the share of claims with a self-reported ``source_or_check``, the share
   marked ``uncertain``, supported claims without a source, abstentions, and the
   distribution of local-claim's ``(1 + share sourced)`` factor.

7. ``guard_stress`` -- whether the two remaining guards can be beaten by a
   cheap strategy. Consistency: a record set that flags every item and never
   abstains. Calibration: the role's pooled ECE (the check applied to all 200
   records of an arm at once, as ``_check_calibration`` defines it) after
   shifting every stated confidence by a constant, clipped to [0, 1]. A shift
   that lowers ECE without looking at any item is calibration the guard was
   meant to rule out.

Usage::

    python3 analysis_cross_role.py --records runs/checks/check_records.jsonl \\
        --out runs/cross_role.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts import PROMPT_VARIANTS, ROLE_PROMPTS  # noqa: E402
from prompts.manipulation_check import _bootstrap_paired  # noqa: E402
from utils import benjamini_hochberg, write_json  # noqa: E402

SEED = 0
#: Draft amendment A4.6, frozen 2026-10-25T14:15:52Z before the matrix existed.
MARGIN = 0.20
FDR_Q = 0.05
ROLES: List[str] = list(ROLE_PROMPTS)


def _load(path: str) -> List[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _present_everywhere(rows: Sequence[Mapping[str, Any]], role: str) -> List[int]:
    # Same exclusion rule as prompts.manipulation_check.check_role: an item
    # counts only if the role's record is present under every variant.
    return [
        index for index, row in enumerate(rows)
        if all(row[role][variant].get("present", True) for variant in PROMPT_VARIANTS)
    ]


def _scores(rows, role: str, variant: str, check_role: str, keep: Sequence[int]) -> np.ndarray:
    check = ROLE_PROMPTS[check_role].check
    return np.array([check([rows[k][role][variant]]) for k in keep], dtype=np.float64)


def matrix(rows: Sequence[Mapping[str, Any]], seed: int = SEED) -> Dict[str, Dict[str, Any]]:
    """Paired effect of each operation prompt on every check, against neutral."""
    cells: Dict[str, Dict[str, Any]] = {}
    for prompt_role in ROLES:
        keep = _present_everywhere(rows, prompt_role)
        for check_role in ROLES:
            op = _scores(rows, prompt_role, "operation", check_role, keep)
            ne = _scores(rows, prompt_role, "neutral", check_role, keep)
            b = _bootstrap_paired(op, ne, seed=seed)
            cells[f"{prompt_role}|{check_role}"] = {
                "d": b["standardized_effect"], "diff": b["diff"], "lo": b["lo"],
                "hi": b["hi"], "p": b["p_value"], "n": len(keep),
            }
    keys = list(cells)
    rejected, q_values = benjamini_hochberg([cells[k]["p"] for k in keys], alpha=FDR_Q)
    for key, is_rejected, q_value in zip(keys, rejected, q_values):
        # Descriptive only: BH across all 36 cells of one family.
        cells[key]["q36"] = float(q_value)
        cells[key]["sig36"] = bool(is_rejected)
    return cells


def discriminant(cells: Mapping[str, Mapping[str, Any]], signed: bool) -> Dict[str, Dict[str, Any]]:
    """Diagonal d against the strongest competing off-diagonal d in each row."""
    out: Dict[str, Dict[str, Any]] = {}
    for role in ROLES:
        own = float(cells[f"{role}|{role}"]["d"])
        off = {j: float(cells[f"{role}|{j}"]["d"]) for j in ROLES if j != role}
        competing = {j: v for j, v in off.items() if math.isfinite(v)}
        if signed:
            # Only same-direction cross-effects compete; with none, the bar is 0.
            competing = {j: v for j, v in competing.items() if np.sign(v) == np.sign(own)}
            strongest = max(competing, key=lambda j: abs(competing[j])) if competing else None
            bar = abs(competing[strongest]) if strongest else 0.0
            margin = abs(own) - bar
            supported = own > 0 and margin >= MARGIN
        else:
            strongest = max(competing, key=lambda j: abs(competing[j]))
            bar = abs(competing[strongest])
            margin = own - bar
            supported = margin >= MARGIN
        out[role] = {"own_d": own, "strongest_other_check": strongest,
                     "strongest_other_abs_d": bar, "margin": margin, "supported": bool(supported)}
    return out


def claim_count(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Pearson r of each check and of output tokens with the record's claim count."""
    records = [row[role][variant] for row in rows for role in ROLES for variant in PROMPT_VARIANTS]
    n_claims = np.array([len(r.get("atomic_claims") or []) for r in records], dtype=np.float64)
    out = {role: float(np.corrcoef(n_claims, [ROLE_PROMPTS[role].check([r]) for r in records])[0, 1])
           for role in ROLES}
    out["output_tokens"] = float(np.corrcoef(
        n_claims, [float(r.get("output_tokens") or 0) for r in records])[0, 1])
    out["n_records"] = len(records)
    return out


def persona_vs_neutral(rows: Sequence[Mapping[str, Any]], seed: int = SEED) -> Dict[str, Dict[str, Any]]:
    results = []
    for role in ROLES:
        keep = _present_everywhere(rows, role)
        b = _bootstrap_paired(_scores(rows, role, "persona", role, keep),
                              _scores(rows, role, "neutral", role, keep), seed=seed)
        results.append((role, b))
    rejected, q_values = benjamini_hochberg([b["p_value"] for _, b in results], alpha=FDR_Q)
    return {role: {"d": b["standardized_effect"], "diff": b["diff"], "lo": b["lo"], "hi": b["hi"],
                   "p": b["p_value"], "q": float(q), "fdr_rejected": bool(r)}
            for (role, b), q, r in zip(results, q_values, rejected)}


def guard_terms(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    records = [row[role][variant] for row in rows for role in ROLES for variant in PROMPT_VARIANTS]
    claims = [c for r in records for c in (r.get("atomic_claims") or [])]
    supported = [c for c in claims if c.get("evidence_status") == "supported"]
    factor = Counter()
    for row in rows:
        for variant in PROMPT_VARIANTS:
            cl = row["local_claim_checker"][variant].get("atomic_claims") or []
            share = float(np.mean([bool(c.get("source_or_check")) for c in cl])) if cl else 0.0
            factor[f"{1.0 + share:.3f}"] += 1
    return {
        "n_records": len(records),
        "n_claims": len(claims),
        "share_claims_with_source_or_check": float(np.mean([bool(c.get("source_or_check")) for c in claims])),
        "share_claims_uncertain": float(np.mean([c.get("evidence_status") == "uncertain" for c in claims])),
        "n_supported_claims": len(supported),
        "n_supported_claims_without_source": int(sum(not c.get("source_or_check") for c in supported)),
        "n_abstentions": int(sum(str(r.get("abstain")) == "True" or r.get("abstain") is True for r in records)),
        "local_claim_factor_counts": dict(sorted(factor.items())),
    }


#: Constant confidence shifts tried by ``guard_stress``.
SHIFTS = [round(-0.20 + 0.05 * k, 2) for k in range(9)]


def guard_stress(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    consistency = ROLE_PROMPTS["consistency_checker"].check(
        [{"failure_flags": ["contradiction"], "abstain": False}] * len(rows))
    calibration: Dict[str, Any] = {}
    check = ROLE_PROMPTS["confidence_calibrator"].check
    for variant in PROMPT_VARIANTS:
        records = [row["confidence_calibrator"][variant] for row in rows]
        ece = {}
        for shift in SHIFTS:
            shifted = [dict(r, confidence=min(1.0, max(0.0, float(r.get("confidence", 0.5)) + shift)))
                       for r in records]
            ece[f"{shift:+.2f}"] = -check(shifted)
        best = min(ece, key=ece.get)
        calibration[variant] = {"ece_by_shift": ece, "ece_unshifted": ece["+0.00"],
                                "best_shift": best, "ece_at_best_shift": ece[best],
                                "defeated": ece[best] < ece["+0.00"]}
    return {"consistency_flag_all_never_abstain": consistency, "calibration_constant_shift": calibration}


def analyse(rows: Sequence[Mapping[str, Any]], seed: int = SEED) -> Dict[str, Any]:
    cells = matrix(rows, seed=seed)
    return {
        "status": "EXPLORATORY -- computed on records already observed under A2/A3",
        "margin": MARGIN,
        "matrix": cells,
        "discriminant": discriminant(cells, signed=False),
        "signed_discriminant": discriminant(cells, signed=True),
        "claim_count": claim_count(rows),
        "persona_vs_neutral": persona_vs_neutral(rows, seed=seed),
        "guard_terms": guard_terms(rows),
        "guard_stress": guard_stress(rows),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", required=True, help="check_records.jsonl")
    p.add_argument("--out", default=None, help="destination for the JSON report")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    report = analyse(_load(args.records), seed=args.seed)
    if args.out:
        write_json(report, args.out)

    short = {r: r.split("_")[0][:8] for r in ROLES}
    print("prompt \\ check".ljust(14) + "".join(short[j].rjust(10) for j in ROLES))
    for i in ROLES:
        print(short[i].ljust(14) + "".join(
            f"{report['matrix'][f'{i}|{j}']['d']:+9.3f}{'*' if report['matrix'][f'{i}|{j}']['sig36'] else ' '}"
            for j in ROLES))
    print(f"\n{'role':26s} {'own d':>8s} {'margin':>8s} {'signed':>8s} {'r(claims)':>10s} {'persona d':>10s}")
    for role in ROLES:
        print(f"{role:26s} {report['discriminant'][role]['own_d']:+8.3f} "
              f"{report['discriminant'][role]['margin']:+8.3f} "
              f"{report['signed_discriminant'][role]['margin']:+8.3f} "
              f"{report['claim_count'][role]:10.3f} {report['persona_vs_neutral'][role]['d']:+10.3f}")
    n_unsigned = sum(v["supported"] for v in report["discriminant"].values())
    n_signed = sum(v["supported"] for v in report["signed_discriminant"].values())
    print(f"\nDiscriminant margin >= {MARGIN}: {n_unsigned}/6 unsigned, {n_signed}/6 signed. EXPLORATORY.")
    print("guard terms:", json.dumps(report["guard_terms"]))
    stress = report["guard_stress"]
    print(f"consistency, flag all and never abstain: {stress['consistency_flag_all_never_abstain']:.3f}")
    for variant, c in stress["calibration_constant_shift"].items():
        print(f"calibration ECE, {variant}: {c['ece_unshifted']:.3f} -> {c['ece_at_best_shift']:.3f} "
              f"at shift {c['best_shift']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
