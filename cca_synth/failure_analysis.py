"""Failure analysis, negative results, and regressions (P15; limitations L12, L13).

The analysis the concept paper lacks entirely. Three requirements, and the third
is the one that is usually written as prose and therefore never enforced:

* **L12** — every agent scored on every benchmark, not only its predeclared
  target, with Benjamini-Hochberg q-values. The machinery already exists in
  ``engine.agent_by_benchmark_family``; what this module adds is the *reading*
  of it, including the off-target cells that a confirmation-biased write-up
  would skip.
* **L13** — for each role, whether its target benefit was purchased with an
  unacceptable regression elsewhere. ``PREREGISTRATION.md`` section 3 states a
  predicted harm, a measurement site, and an exact "unacceptable if" threshold
  for all six roles. Those thresholds are encoded here, so a role's verdict is
  computed from the numbers rather than argued.
* **The stopping rule** — if the council does not beat its budget-matched
  controls, that is the finding, and it is written as the finding.

Nothing here decides whether a result is *good*. It decides whether the
pre-registered criteria were met, which is a different and much narrower
question, and the only one a confirmatory study is entitled to answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "PREDICTED_HARMS",
    "HarmPrediction",
    "role_regression_report",
    "off_target_report",
    "primary_result_verdict",
    "failure_analysis_report",
]


@dataclass(frozen=True)
class HarmPrediction:
    """One row of ``PREREGISTRATION.md`` section 3.

    Attributes:
        role: Functional role identifier.
        predicted_harm: What the role is expected to make worse.
        measured_by: The metric key carrying the measurement.
        threshold: The value at which the regression becomes unacceptable.
        direction: ``"increase"`` when exceeding the threshold is bad,
            ``"decrease"`` when falling below it is bad.
        unacceptable_if: The registration's wording, quoted for the report.
    """

    role: str
    predicted_harm: str
    measured_by: str
    threshold: float
    direction: str
    unacceptable_if: str

    def breached(self, value: float) -> Optional[bool]:
        """Whether a measured value breaches the registered threshold.

        Args:
            value: The measured quantity.

        Returns:
            ``True`` if unacceptable, ``False`` if acceptable, ``None`` if the
            value is missing. ``None`` is *not* a pass -- an unmeasured harm is
            reported as unmeasured, because L13 is about what was checked.
        """
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return value > self.threshold if self.direction == "increase" else value < self.threshold


#: ``PREREGISTRATION.md`` section 3, encoded. Changing a threshold here changes
#: the registered analysis and requires a dated amendment in section 11.
PREDICTED_HARMS: Tuple[HarmPrediction, ...] = (
    HarmPrediction(
        role="counter_evidence_checker",
        predicted_harm="Over-flagging; raised abstention on well-supported claims",
        measured_by="coverage_drop_vs_control",
        threshold=0.05,
        direction="increase",
        unacceptable_if="Coverage drops >0.05 at equal accuracy",
    ),
    HarmPrediction(
        role="consistency_checker",
        predicted_harm="Token and latency inflation from re-checking",
        measured_by="p95_latency_ratio_vs_single_agent",
        threshold=2.0,
        direction="increase",
        unacceptable_if="p95 latency >2x single agent",
    ),
    HarmPrediction(
        role="local_claim_checker",
        predicted_harm="Degraded long-range integration",
        measured_by="long_range_accuracy_drop",
        threshold=0.03,
        direction="increase",
        unacceptable_if="Long-range accuracy drops >0.03",
    ),
    HarmPrediction(
        role="explore_then_prune",
        predicted_harm="Verbosity; synthesizer capture",
        measured_by="agent_dominance",
        threshold=0.60,
        direction="increase",
        unacceptable_if="Dominance >0.60 or >35% of council tokens",
    ),
    HarmPrediction(
        role="rare_hypothesis_checker",
        predicted_harm="Fabricated alternatives displacing correct canonical answers",
        measured_by="fabricated_alternative_rate",
        threshold=0.05,
        direction="increase",
        unacceptable_if="Fabrication rate >0.05",
    ),
    HarmPrediction(
        role="confidence_calibrator",
        predicted_harm="Under-confidence; inappropriate stubbornness when the user is right",
        measured_by="capitulation_when_right_delta",
        threshold=0.05,
        direction="increase",
        unacceptable_if="Capitulation-when-right worsens >0.05",
    ),
)

#: Secondary threshold for explore_then_prune, whose registered rule is a
#: disjunction. Kept separate so the report can say which limb fired.
_TOKEN_SHARE_THRESHOLD = 0.35


def role_regression_report(
    measurements: Mapping[str, Mapping[str, float]],
    shapley_by_role: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, Any]:
    """Score each role against its pre-registered harm prediction (L13).

    A role **passes** only if it shows a positive contribution on its target
    *and* does not breach its registered harm threshold. A benefit purchased
    with an unacceptable regression is not a benefit.

    Args:
        measurements: Role to metric mapping. Keys should include the
            ``measured_by`` field of the corresponding prediction; for
            ``explore_then_prune`` also ``council_token_share``.
        shapley_by_role: Optional role to ``{"shapley", "ci_lo", "ci_hi"}``, used
            to decide whether a target benefit exists at all.

    Returns:
        Per-role verdicts and a roll-up of roles that failed, plus roles whose
        harm was never measured.
    """
    shapley_by_role = shapley_by_role or {}
    rows: List[Dict[str, Any]] = []

    for prediction in PREDICTED_HARMS:
        observed = dict(measurements.get(prediction.role, {}))
        value = observed.get(prediction.measured_by)
        breached = prediction.breached(value)

        # explore_then_prune's registered rule is a disjunction; check both.
        second_limb = None
        if prediction.role == "explore_then_prune":
            share = observed.get("council_token_share")
            if share is not None and not (isinstance(share, float) and math.isnan(share)):
                second_limb = share > _TOKEN_SHARE_THRESHOLD
                if second_limb:
                    breached = True

        shapley = shapley_by_role.get(prediction.role, {})
        ci_lo = shapley.get("ci_lo")
        benefit = None if ci_lo is None else ci_lo > 0.0

        if breached is None:
            verdict = "UNMEASURED"
            reason = (
                f"'{prediction.measured_by}' was not measured. L13 requires the "
                "predicted harm to be checked; an unmeasured harm is not a pass."
            )
        elif breached:
            verdict = "FAIL"
            reason = (
                f"Registered rule breached: {prediction.unacceptable_if}. "
                f"Measured {prediction.measured_by} = {value:.4f}."
            )
            if second_limb:
                reason += (
                    f" Token-share limb also breached "
                    f"({observed.get('council_token_share'):.4f} > {_TOKEN_SHARE_THRESHOLD})."
                )
        elif benefit is False:
            verdict = "NO BENEFIT"
            reason = (
                "Harm threshold respected, but the role's Shapley interval does "
                "not exclude zero on its target, so there is no benefit to weigh."
            )
        else:
            verdict = "PASS"
            reason = (
                f"{prediction.measured_by} = {value:.4f}, within the registered "
                f"threshold of {prediction.threshold}."
            )

        rows.append(
            {
                "role": prediction.role,
                "predicted_harm": prediction.predicted_harm,
                "measured_by": prediction.measured_by,
                "measured_value": value,
                "threshold": prediction.threshold,
                "unacceptable_if": prediction.unacceptable_if,
                "harm_breached": breached,
                "target_benefit_present": benefit,
                "verdict": verdict,
                "reason": reason,
            }
        )

    return {
        "roles": rows,
        "n_failed": sum(1 for r in rows if r["verdict"] == "FAIL"),
        "n_unmeasured": sum(1 for r in rows if r["verdict"] == "UNMEASURED"),
        "n_passed": sum(1 for r in rows if r["verdict"] == "PASS"),
        "failing_roles": [r["role"] for r in rows if r["verdict"] == "FAIL"],
        "unmeasured_roles": [r["role"] for r in rows if r["verdict"] == "UNMEASURED"],
        "note": (
            "Thresholds are quoted from PREREGISTRATION.md section 3 and were "
            "fixed before data collection. A role passes only if its target "
            "benefit is not purchased with an unacceptable regression (L13)."
        ),
    }


def off_target_report(family: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the agent-by-benchmark family for off-target effects (L12).

    The cells a confirmation-biased write-up skips are the ones where a role
    hurts a benchmark that is not its target. This surfaces them explicitly and
    counts them against the same FDR correction as everything else.

    Args:
        family: Output of ``engine.agent_by_benchmark_family``.

    Returns:
        On-target and off-target breakdowns, and every significant harm found.
    """
    cells = list(family.get("cells", []))
    on_target = [c for c in cells if c.get("predeclared_primary")]
    off_target = [c for c in cells if not c.get("predeclared_primary")]

    def significant_harms(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Cells where the marginal value is negative and survives FDR."""
        return [
            {
                "role": c["role"],
                "benchmark": c["benchmark"],
                "marginal_value": c["marginal_value"],
                "ci": [c.get("ci_lo"), c.get("ci_hi")],
                "q_value": c.get("q_value"),
            }
            for c in rows
            if c.get("significant_after_fdr") and c.get("marginal_value", 0.0) < 0.0
        ]

    off_harms = significant_harms(off_target)
    on_harms = significant_harms(on_target)

    return {
        "n_cells": len(cells),
        "n_on_target": len(on_target),
        "n_off_target": len(off_target),
        "n_significant_off_target_harms": len(off_harms),
        "off_target_harms": off_harms,
        "on_target_harms": on_harms,
        "roles_helping_only_off_target": sorted(
            {
                c["role"]
                for c in off_target
                if c.get("significant_after_fdr") and c.get("marginal_value", 0.0) > 0.0
            }
            - {
                c["role"]
                for c in on_target
                if c.get("significant_after_fdr") and c.get("marginal_value", 0.0) > 0.0
            }
        ),
        "note": (
            "Every agent is scored on every benchmark (L12). A role that helps "
            "only off-target has not validated its predeclared mechanism, "
            "whatever its headline number."
        ),
    }


def primary_result_verdict(
    contrasts: Sequence[Mapping[str, Any]],
    *,
    sesoi: float = 0.03,
    cost_matched_comparator: str = "cost_matched_best_control",
) -> Dict[str, Any]:
    """Apply the pre-registration's stopping rule to the primary contrast.

    ``PREREGISTRATION.md`` section 8: *"If H1 fails under cost-matching, the
    paper reports a negative primary result. It does not switch to the
    unconstrained comparison as the headline, and it does not substitute a
    secondary outcome."*

    Encoding it here is the point. The rule exists to remove the temptation that
    arrives with the numbers, and a rule that has to be remembered at writing
    time is not a rule.

    Args:
        contrasts: Paired contrasts, each with ``comparator``, ``diff``, ``lo``,
            ``hi``.
        sesoi: H1's smallest effect size of interest.
        cost_matched_comparator: Name of the cost-matched control arm.

    Returns:
        The verdict, and the headline the paper is permitted to claim.
    """
    by_name = {c.get("comparator"): c for c in contrasts}
    cost_matched = by_name.get(cost_matched_comparator)

    if cost_matched is None:
        return {
            "h1_testable": False,
            "h1_supported": False,
            "permitted_headline": (
                "No cost-matched contrast is present, so H1 cannot be evaluated. "
                "Reporting an unconstrained gain as the headline is forbidden by "
                "the registered stopping rule."
            ),
            "reason": f"comparator '{cost_matched_comparator}' not found",
        }

    diff = cost_matched.get("diff", float("nan"))
    lo = cost_matched.get("lo", float("nan"))
    includes_zero = lo <= 0.0
    below_sesoi = diff < sesoi
    supported = not includes_zero and not below_sesoi

    if supported:
        headline = (
            f"The council exceeds the cost-matched control by {diff:+.4f} "
            f"[{lo:+.4f}, {cost_matched.get('hi', float('nan')):+.4f}], clearing "
            f"the registered SESOI of {sesoi}."
        )
    else:
        parts = []
        if includes_zero:
            parts.append("the cost-matched interval includes zero")
        if below_sesoi:
            parts.append(f"the point estimate {diff:+.4f} is below the SESOI of {sesoi}")
        headline = (
            "**Negative primary result.** H1 is falsified because "
            + " and ".join(parts)
            + ". Per the registered stopping rule this is the headline: the "
            "unconstrained comparison may not be substituted, and no secondary "
            "outcome may be promoted in its place."
        )

    return {
        "h1_testable": True,
        "h1_supported": supported,
        "cost_matched_diff": diff,
        "cost_matched_ci": [lo, cost_matched.get("hi")],
        "sesoi": sesoi,
        "interval_includes_zero": includes_zero,
        "below_sesoi": below_sesoi,
        "permitted_headline": headline,
    }


def failure_analysis_report(
    family: Mapping[str, Any],
    measurements: Mapping[str, Mapping[str, float]],
    contrasts: Sequence[Mapping[str, Any]],
    shapley_by_role: Optional[Mapping[str, Mapping[str, float]]] = None,
    *,
    sesoi: float = 0.03,
) -> Dict[str, Any]:
    """Assemble the complete P15 analysis.

    Args:
        family: Output of ``engine.agent_by_benchmark_family``.
        measurements: Per-role harm measurements.
        contrasts: Paired primary contrasts.
        shapley_by_role: Per-role Shapley estimates.
        sesoi: H1's smallest effect size of interest.

    Returns:
        The regression table, off-target analysis, primary verdict, and a
        plain-language summary of what the study is entitled to claim.
    """
    regressions = role_regression_report(measurements, shapley_by_role)
    off_target = off_target_report(family)
    primary = primary_result_verdict(contrasts, sesoi=sesoi)

    claims: List[str] = [primary["permitted_headline"]]
    if regressions["failing_roles"]:
        claims.append(
            "Roles breaching their registered harm threshold: "
            + ", ".join(f"`{r}`" for r in regressions["failing_roles"])
            + ". These did not pass, whatever their target-benchmark number."
        )
    if regressions["unmeasured_roles"]:
        claims.append(
            "Roles whose predicted harm was never measured: "
            + ", ".join(f"`{r}`" for r in regressions["unmeasured_roles"])
            + ". No pass may be claimed for these."
        )
    if off_target["off_target_harms"]:
        claims.append(
            f"{len(off_target['off_target_harms'])} significant off-target harms "
            "survive FDR correction and are reported in the main table (L12)."
        )
    if not regressions["failing_roles"] and not off_target["off_target_harms"]:
        claims.append("No role breached its registered harm threshold.")

    return {
        "primary": primary,
        "regressions": regressions,
        "off_target": off_target,
        "permitted_claims": claims,
        "note": (
            "Every verdict here is computed from pre-registered thresholds. "
            "Where a threshold was not met, the failure is the finding and is "
            "reported with equal prominence to any success."
        ),
    }
