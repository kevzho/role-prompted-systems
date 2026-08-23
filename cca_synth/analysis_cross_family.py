"""Cross-family replication of the six manipulation checks.

``\\S5`` and ``\\S7`` recorded single-family scope as the study's largest
untested assumption: every result came from one pretraining lineage, so nothing
spoke to whether the pattern was a property of role prompting or of one model.
This script contrasts the same six checks, on the same 200 development items,
run against a second family.

Two questions are separated, because they have different answers:

*aggregate*
    Do most role prompts fail their own check on both families? This is the
    paper's central claim and it is what replication is for.

*per role*
    Does the *same* role fire on both? A role can fail on one lineage and pass
    on another, and that is not a contradiction of the aggregate claim -- it is
    the reason a per-model check is necessary rather than a lookup table.

A sign flip is reported only where at least one family shows a non-negligible
effect. Two nulls of opposite sign are not a divergence, and counting them as
one would manufacture disagreement out of noise.

Usage::

    python3 analysis_cross_family.py --primary runs/checks \
        --replication runs/checks-grok --out runs/cross_family.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Mapping

import numpy as np

from utils import format_table, write_json

#: Below this, an effect is treated as null and its sign carries no meaning.
NEGLIGIBLE_D: float = 0.20


def _results(report_path: str) -> Dict[str, Mapping[str, Any]]:
    """Per-role rows from a manipulation report, keyed by role."""
    with open(report_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {row["role"]: row for row in payload["results"]}


def _confidence(records_glob: str) -> np.ndarray:
    """Every stated confidence in a check run's records."""
    values: List[float] = []
    for path in sorted(glob.glob(records_glob)):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                for candidate in json.loads(line).get("candidates", []):
                    stated = candidate.get("confidence")
                    if stated is not None:
                        values.append(float(stated))
    return np.asarray(values)


def cross_family(primary_dir: str, replication_dir: str) -> Dict[str, Any]:
    """Contrast one family's manipulation checks against another's."""
    primary = _results(os.path.join(primary_dir, "manipulation_report.json"))
    replication = _results(os.path.join(replication_dir, "manipulation_report.json"))

    roles: List[Dict[str, Any]] = []
    for role, row in primary.items():
        other = replication[role]
        d_primary = float(row["vs_neutral"]["standardized_effect"])
        d_replication = float(other["vs_neutral"]["standardized_effect"])
        both_negligible = (
            abs(d_primary) < NEGLIGIBLE_D and abs(d_replication) < NEGLIGIBLE_D
        )
        if both_negligible:
            agreement = "null_on_both"
        elif (d_primary > 0) != (d_replication > 0):
            agreement = "sign_flip"
        elif row["passed"] == other["passed"]:
            agreement = "replicates"
        else:
            agreement = "same_sign_different_verdict"
        roles.append({
            "role": role,
            "d_primary": d_primary,
            "d_replication": d_replication,
            "diff_primary": float(row["vs_neutral"]["diff"]),
            "diff_replication": float(other["vs_neutral"]["diff"]),
            "lo_primary": float(row["vs_neutral"]["lo"]),
            "hi_primary": float(row["vs_neutral"]["hi"]),
            "lo_replication": float(other["vs_neutral"]["lo"]),
            "hi_replication": float(other["vs_neutral"]["hi"]),
            "passed_primary": bool(row["passed"]),
            "passed_replication": bool(other["passed"]),
            "q_primary": float(row["q_value"]),
            "q_replication": float(other["q_value"]),
            "agreement": agreement,
        })

    confidence = {}
    for label, directory in (("primary", primary_dir), ("replication", replication_dir)):
        values = _confidence(os.path.join(directory, "records.*.jsonl"))
        confidence[label] = {
            "n": int(values.size),
            "mean": float(values.mean()) if values.size else float("nan"),
            "sd": float(values.std()) if values.size else float("nan"),
            "share_at_or_above_0.95": float((values >= 0.95).mean()) if values.size else float("nan"),
            "n_distinct_values": int(len(set(np.round(values, 2)))) if values.size else 0,
        }

    return {
        "roles": roles,
        "n_passed_primary": sum(1 for r in roles if r["passed_primary"]),
        "n_passed_replication": sum(1 for r in roles if r["passed_replication"]),
        "n_roles": len(roles),
        "n_replicating": sum(1 for r in roles
                             if r["agreement"] in ("replicates", "null_on_both")),
        "n_sign_flips": sum(1 for r in roles if r["agreement"] == "sign_flip"),
        "verbalised_confidence": confidence,
        "negligible_d_threshold": NEGLIGIBLE_D,
        "note": (
            "Same six checks, same 200 development items, two pretraining "
            "lineages. A sign flip is counted only where at least one family "
            "exceeds |d| = 0.20; two nulls of opposite sign are noise, not "
            "disagreement. The aggregate claim (most roles fail their own "
            "check) and the per-role verdicts are reported separately because "
            "they replicate differently."
        ),
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", default="runs/checks")
    parser.add_argument("--replication", default="runs/checks-grok")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = cross_family(args.primary, args.replication)
    print(format_table(report["roles"]))
    print(f"\npassed: primary {report['n_passed_primary']}/{report['n_roles']}, "
          f"replication {report['n_passed_replication']}/{report['n_roles']}")
    print(f"agreeing roles: {report['n_replicating']}/{report['n_roles']}, "
          f"sign flips: {report['n_sign_flips']}")
    print("\nverbalised confidence:")
    print(format_table([{"family": k, **v} for k, v in report["verbalised_confidence"].items()]))
    if args.out:
        write_json(report, args.out)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
