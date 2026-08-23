"""Run the manipulation checks and decide whether each role passes.

``PREREGISTRATION.md`` §1 treats a passing manipulation check as a
*precondition* for interpreting any agent-level result: a role that does not
move its predeclared quantity relative to a length-matched neutral control is
not implementing its operation, whatever it scores on a benchmark.

The comparison is paired at the item level and bootstrapped, matching the
statistics used everywhere else in this repository, so a check result is
directly comparable to a benchmark contrast.

Usage::

    python -m prompts.manipulation_check --records dev_records.jsonl

The input is a JSONL file where each line carries one item's records under all
three prompt variants::

    {"item_id": "...", "operation": {...}, "neutral": {...}, "persona": {...}}

Each variant value is a §7 candidate record. For the confidence calibrator the
records must additionally carry ``"correct": true|false``, since expected
calibration error cannot be computed without outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import PROMPT_VARIANTS, ROLE_PROMPTS, prompt_manifest  # noqa: E402
from utils import benjamini_hochberg, format_table, write_json  # noqa: E402

__all__ = ["check_role", "run_checks", "main"]

#: Minimum *standardized* paired effect for a role to pass, per
#: ``PREREGISTRATION.md`` §11 amendment A2.2 (H6).
#:
#: Standardized rather than absolute, because the six checks do not share a
#: unit. ``_check_counterevidence`` and ``_check_claim_decomposition`` return
#: unbounded per-record counts; ``_check_contradiction_detection`` returns a rate
#: difference in [-1, 1]. A single absolute threshold across them is not
#: interpretable -- 0.10 is a rounding error on a claim count and a large move on
#: a bounded rate -- and the earlier absolute 0.10 threshold used here was
#: replaced for exactly that reason. A2.1 records the unit heterogeneity.
MIN_STANDARDIZED_EFFECT: float = 0.40

#: Benjamini-Hochberg target false discovery rate across the six role checks,
#: per ``PREREGISTRATION.md`` §11 amendment A3.2.
#:
#: The six roles are tested simultaneously against the same neutral control and
#: form one family. Six independent tests at 0.05 give a family-wise error rate
#: near 26%, which is not an acceptable basis for the study's *primary* outcome.
#: BH is used rather than Bonferroni for consistency with the agent-by-benchmark
#: family in ``engine.py``, which already controls FDR at this level.
FDR_Q: float = 0.05

#: Retained for reporting only. A role is no longer judged on this quantity;
#: it is still emitted because the raw difference in check units is what a
#: reader needs to see alongside the standardized one.
MIN_EFFECT: float = 0.10


def _bootstrap_paired(
    operation_scores: np.ndarray,
    control_scores: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap of the per-item difference.

    Args:
        operation_scores: Per-item check score under the operation prompt.
        control_scores: Per-item score under the control prompt.
        n_bootstrap: Resamples.
        seed: RNG seed.

    Returns:
        Mapping with ``diff``, ``lo``, ``hi``, ``prob_positive``,
        ``standardized_effect`` (paired Cohen's *d*), and
        ``interval_excludes_zero``.
    """
    difference = operation_scores - control_scores
    n = difference.size
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "prob_positive": float("nan"),
                "standardized_effect": float("nan"),
                "p_value": 1.0,
                "interval_excludes_zero": False}
    rng = np.random.default_rng(seed)
    draws = difference[rng.integers(0, n, size=(n_bootstrap, n))].mean(axis=1)

    # Paired Cohen's d: the mean difference in units of its own dispersion.
    # A zero-variance difference is not an infinitely strong effect -- it is
    # either a constant shift on every item (report it as such) or no difference
    # at all -- so guard rather than divide.
    spread = float(difference.std(ddof=1)) if n > 1 else 0.0
    mean_difference = float(difference.mean())
    if spread > 0.0:
        standardized = mean_difference / spread
    elif mean_difference == 0.0:
        standardized = 0.0
    else:
        standardized = float("inf") if mean_difference > 0 else float("-inf")

    lo = float(np.quantile(draws, 0.025))
    hi = float(np.quantile(draws, 0.975))
    prob_positive = float((draws > 0).mean())

    # Two-sided bootstrap p-value, needed because A3.2 registers Benjamini-
    # Hochberg control across the six role checks and BH consumes p-values, not
    # intervals.
    #
    # Computed from the NON-STRICT tail masses, not from ``prob_positive``. A
    # difference that is identically zero on every item -- which happens when a
    # factor is saturated in both arms -- yields prob_positive == 0.0, and
    # ``2 * min(0, 1)`` would report p = 0: maximal significance for a variable
    # that never moved. Using P(draw <= 0) and P(draw >= 0) sends that case to
    # 2 * min(1, 1), which clips to 1.0, as it must.
    #
    # Floored at 1/n_bootstrap: a resampling procedure cannot resolve a p-value
    # finer than its own resolution, and reporting 0.0 would claim it can.
    p_value = 2.0 * min(float((draws <= 0.0).mean()), float((draws >= 0.0).mean()))
    p_value = float(min(1.0, max(p_value, 1.0 / n_bootstrap)))

    return {
        "diff": mean_difference,
        "lo": lo,
        "hi": hi,
        "prob_positive": prob_positive,
        "standardized_effect": standardized,
        "p_value": p_value,
        "interval_excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def check_role(
    role: str,
    records_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: int = 0,
) -> Dict[str, Any]:
    """Score one role's manipulation check against both controls.

    Args:
        role: Functional role identifier.
        records_by_variant: Mapping from variant name to that variant's records,
            item-aligned across variants.
        seed: Bootstrap seed.

    Returns:
        A result dict with the paired contrast against the neutral control, the
        contrast against the persona control, and a pass/fail verdict.

    Raises:
        KeyError: If the role is unknown.
        ValueError: If the variants are not item-aligned.
    """
    spec = ROLE_PROMPTS[role]
    lengths = {v: len(records_by_variant.get(v, [])) for v in PROMPT_VARIANTS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"variants are not item-aligned for {role}: {lengths}")

    # Keep only items whose record is present under *every* variant. An absent
    # record scores near zero on every check, so including it would count a
    # refusal or a truncation as "the operation did not happen" -- and it would
    # do so asymmetrically, since refusal rates differ by variant. Section 4
    # forbids imputing missing agents; dropping the item pairwise and reporting
    # the count is the version of that rule which applies here.
    n_total = lengths[PROMPT_VARIANTS[0]]
    keep = [
        index
        for index in range(n_total)
        if all(
            records_by_variant[variant][index].get("present", True)
            for variant in PROMPT_VARIANTS
        )
    ]

    if not keep:
        return {
            "role": role,
            "check": spec.check_name,
            "primary_benchmark": spec.primary_benchmark,
            "n_items": 0,
            "n_items_collected": int(n_total),
            "n_excluded_absent": int(n_total),
            "passed": False,
            "error": (
                "no item has this role's record present under all three "
                "variants; the check cannot be scored"
            ),
        }

    # The check is defined over a record *set*, so score each item individually
    # by passing a one-element set; this is what makes the contrast paired.
    per_item = {
        variant: np.array(
            [spec.check([records_by_variant[variant][index]]) for index in keep],
            dtype=np.float64,
        )
        for variant in PROMPT_VARIANTS
    }

    versus_neutral = _bootstrap_paired(per_item["operation"], per_item["neutral"], seed=seed)
    versus_persona = _bootstrap_paired(
        per_item["operation"], per_item["persona"], seed=seed + 1
    )

    # A2.2's rule is conjunctive and is applied here rather than described in
    # prose: the interval must exclude zero AND the standardized effect must
    # reach the threshold. Either alone is insufficient -- a tight interval
    # around a trivial shift is precision about nothing, and a large ratio with
    # an interval spanning zero is noise.
    #
    # NOTE (A3.2): this is the *uncorrected* verdict. The registered rule adds
    # Benjamini-Hochberg control across the six roles, which cannot be decided
    # from one role in isolation -- ``run_checks`` applies it over the family and
    # overwrites ``passed``. A caller invoking ``check_role`` directly gets the
    # uncorrected verdict and must not report it as the registered outcome.
    passed = (
        versus_neutral["interval_excludes_zero"]
        and versus_neutral["standardized_effect"] >= MIN_STANDARDIZED_EFFECT
    )
    return {
        "passed_uncorrected": bool(passed),
        "p_value": float(versus_neutral["p_value"]),
        "role": role,
        "check": spec.check_name,
        "description": spec.check_description,
        "primary_benchmark": spec.primary_benchmark,
        "n_items": int(per_item["operation"].size),
        "n_items_collected": int(n_total),
        "n_excluded_absent": int(n_total - len(keep)),
        "mean_operation": float(per_item["operation"].mean()),
        "mean_neutral": float(per_item["neutral"].mean()),
        "mean_persona": float(per_item["persona"].mean()),
        "vs_neutral": versus_neutral,
        "vs_persona": versus_persona,
        "passed": bool(passed),
        # The operation-versus-persona contrast is the RQ3 quantity at the prompt
        # level: how much of the effect survives when the label is held constant
        # and only the operational content is removed.
        "operation_over_label_share": (
            float(versus_persona["diff"] / versus_neutral["diff"])
            if abs(versus_neutral["diff"]) > 1e-9
            else float("nan")
        ),
    }


def run_checks(path: str, output: Optional[str] = None, seed: int = 0) -> Dict[str, Any]:
    """Run every role's manipulation check over a dev-set records file.

    Args:
        path: JSONL file, one line per item, with per-variant records per role.
        output: Optional destination for the JSON report.
        seed: Bootstrap seed.

    Returns:
        The full report.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"dev records not found: {path}")

    by_role: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {
        role: {variant: [] for variant in PROMPT_VARIANTS} for role in ROLE_PROMPTS
    }
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            for role in ROLE_PROMPTS:
                entry = payload.get(role)
                if not entry:
                    continue
                for variant in PROMPT_VARIANTS:
                    if variant in entry:
                        by_role[role][variant].append(entry[variant])

    results = []
    for role in ROLE_PROMPTS:
        if not by_role[role]["operation"]:
            results.append(
                {"role": role, "passed": False, "error": "no records for this role"}
            )
            continue
        results.append(check_role(role, by_role[role], seed=seed))

    # A3.2: the six role checks form one family. Apply Benjamini-Hochberg across
    # the roles that actually produced a contrast, then decide each role's
    # registered verdict conjunctively: FDR-rejected AND standardized effect at
    # or above threshold. Roles that could not be scored carry no p-value and are
    # excluded from the family rather than entered with p = 1, which would dilute
    # the correction for every other role.
    scorable = [r for r in results if "p_value" in r and not r.get("error")]
    if scorable:
        rejected, q_values = benjamini_hochberg(
            [float(r["p_value"]) for r in scorable], alpha=FDR_Q
        )
        for result, is_rejected, q_value in zip(scorable, rejected, q_values):
            result["q_value"] = float(q_value)
            result["fdr_rejected"] = bool(is_rejected)
            result["passed"] = bool(
                is_rejected
                and result["vs_neutral"]["standardized_effect"] >= MIN_STANDARDIZED_EFFECT
            )
    for result in results:
        result.setdefault("q_value", float("nan"))
        result.setdefault("fdr_rejected", False)

    n_family = len(scorable)
    report = {
        "prompt_manifest": prompt_manifest(),
        "thresholds": {
            "min_standardized_effect": MIN_STANDARDIZED_EFFECT,
            "fdr_q": FDR_Q,
            "family_size": n_family,
            "rule": (
                "PREREGISTRATION.md A2.2 (H6) as amended by A3.2: the six role "
                "checks form one family; Benjamini-Hochberg FDR control at "
                f"q={FDR_Q} is applied across them, and a role passes only if it "
                "is FDR-rejected AND its standardized paired effect is at least "
                f"{MIN_STANDARDIZED_EFFECT}. Uncorrected per-role intervals and "
                "verdicts are reported alongside as 'interval_excludes_zero' and "
                "'passed_uncorrected' so the correction's effect is visible."
            ),
            "reported_only_min_effect": MIN_EFFECT,
        },
        "n_passed_uncorrected": sum(1 for r in results if r.get("passed_uncorrected")),
        "n_passed": sum(1 for r in results if r.get("passed")),
        "n_roles": len(ROLE_PROMPTS),
        "results": results,
        "interpretation": (
            "A role that fails its manipulation check is not implementing its "
            "predeclared operation, and its benchmark results must not be "
            "interpreted as evidence about that operation "
            "(PREREGISTRATION.md section 1)."
        ),
        "caveats": [
            "The shared output schema (prompts._OUTPUT_CONTRACT) requires every "
            "variant to emit atomic_claims, counterevidence, and failure_flags, "
            "so the neutral control is also asked for them and will produce "
            "some. These checks measure amount rather than presence, so the "
            "contrast survives, but it is compressed relative to a design where "
            "the control emitted nothing. Registered in PREREGISTRATION.md A2.1.",
            "Items where a role's record is absent under any variant are "
            "excluded from that role's contrast rather than scored as zero. "
            "Scoring an absent record would conflate a refused or truncated "
            "call with a performed-but-empty operation, and section 4 forbids "
            "imputing missing agents.",
        ],
    }
    if output:
        write_json(report, output)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector.

    Returns:
        ``0`` if every role passes, ``1`` otherwise, so this can gate a run.
    """
    parser = argparse.ArgumentParser(description="Run role manipulation checks.")
    parser.add_argument("--records", required=True, help="Dev-set records JSONL.")
    parser.add_argument("--out", default=None, help="Destination for the JSON report.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    args = parser.parse_args(argv)

    report = run_checks(args.records, output=args.out, seed=args.seed)
    rows = [
        {
            "role": r["role"],
            "check": r.get("check", "-"),
            "n": r.get("n_items", 0),
            "excl": r.get("n_excluded_absent", 0),
            "operation": r.get("mean_operation", float("nan")),
            "neutral": r.get("mean_neutral", float("nan")),
            "delta": r.get("vs_neutral", {}).get("diff", float("nan")),
            "d": r.get("vs_neutral", {}).get("standardized_effect", float("nan")),
            "p": r.get("p_value", float("nan")),
            "q": r.get("q_value", float("nan")),
            "ci!=0": r.get("vs_neutral", {}).get("interval_excludes_zero", False),
            "uncorr": r.get("passed_uncorrected", False),
            "passed": r.get("passed", False),
        }
        for r in report["results"]
    ]
    print(format_table(rows))
    print(
        f"\nRule: Benjamini-Hochberg q <= {FDR_Q} across the "
        f"{report['thresholds']['family_size']}-role family AND d >= "
        f"{MIN_STANDARDIZED_EFFECT} (PREREGISTRATION.md A2.2 as amended by A3.2). "
        f"'uncorr' is the pre-correction verdict, shown for comparison: "
        f"{report['n_passed_uncorrected']} role(s) would pass without the "
        "correction."
    )
    print(f"{report['n_passed']}/{report['n_roles']} roles passed.")
    if report["n_passed"] < report["n_roles"]:
        print(
            "Roles that failed are not implementing their operation. Do not "
            "interpret their benchmark results as evidence about that operation."
        )
    return 0 if report["n_passed"] == report["n_roles"] else 1


if __name__ == "__main__":
    sys.exit(main())
