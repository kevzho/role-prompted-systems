"""Model-judge audit against human judgement (limitation L16).

The rule this module enforces:

> Do not use an LLM judge as a primary outcome without a human audit sample
> quantifying its error rate.

That is not advice, it is a gate. :func:`gate_primary_outcome` returns a refusal
when the audit is missing, too small, or shows the judge disagreeing with humans
by more than the benchmark's smallest effect size of interest -- because a judge
whose error exceeds the effect being measured cannot detect that effect.

The audit sample is *stratified*, not random. A uniform sample over-represents
the easy majority class and would report a flattering agreement rate driven
entirely by items no one disagrees about. Strata are formed over the judge's own
decision and, where available, the condition, so that the audit covers the cells
where a judge error would actually change a conclusion.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "stratified_audit_sample",
    "judge_human_agreement",
    "gate_primary_outcome",
    "MIN_AUDIT_ITEMS",
]

#: Minimum audited items before an agreement rate is reported at all. Below
#: this, the confidence interval on the agreement rate is wider than any
#: difference the study is powered to detect, so the audit answers nothing.
MIN_AUDIT_ITEMS = 100


def stratified_audit_sample(
    decisions: Sequence[Any],
    *,
    n: int = MIN_AUDIT_ITEMS,
    strata: Optional[Sequence[Any]] = None,
    seed: int = 20260728,
) -> List[int]:
    """Draw a stratified sample of model-judge decisions for human audit.

    Args:
        decisions: The model judge's decisions, one per item.
        n: Target sample size.
        strata: Optional stratum label per item; defaults to the decision
            itself, which is the minimum sensible stratification.
        seed: Sampling seed, recorded so the sample is reproducible.

    Returns:
        Sorted indices into ``decisions``.

    Raises:
        ValueError: If ``strata`` is misaligned, or ``n`` exceeds the population.
    """
    if strata is None:
        strata = [str(d) for d in decisions]
    if len(strata) != len(decisions):
        raise ValueError(f"length mismatch: {len(strata)} strata for {len(decisions)} decisions")
    if n > len(decisions):
        raise ValueError(
            f"cannot audit {n} of {len(decisions)} decisions; the audit sample "
            "cannot exceed the population"
        )

    buckets: Dict[Any, List[int]] = {}
    for index, label in enumerate(strata):
        buckets.setdefault(label, []).append(index)

    rng = random.Random(seed)
    for indices in buckets.values():
        rng.shuffle(indices)

    # Proportional allocation with a floor: every stratum contributes at least
    # one item, so a rare-but-decisive cell is never sampled away entirely.
    total = len(decisions)
    allocation: Dict[Any, int] = {}
    for label, indices in buckets.items():
        share = max(1, round(n * len(indices) / total))
        allocation[label] = min(share, len(indices))

    # Reconcile rounding against the target, largest strata first.
    order = sorted(buckets, key=lambda label: -len(buckets[label]))
    while sum(allocation.values()) > n:
        for label in order:
            if allocation[label] > 1:
                allocation[label] -= 1
                if sum(allocation.values()) == n:
                    break
        else:
            break
    while sum(allocation.values()) < n:
        for label in order:
            if allocation[label] < len(buckets[label]):
                allocation[label] += 1
                if sum(allocation.values()) == n:
                    break
        else:
            break

    sample: List[int] = []
    for label, count in allocation.items():
        sample.extend(buckets[label][:count])
    return sorted(sample)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because agreement rates sit near 1.0,
    where the normal interval runs above one and understates the uncertainty.

    Args:
        successes: Number of agreements.
        total: Number of audited items.
        z: Normal quantile; 1.96 for 95%.

    Returns:
        Tuple ``(lo, hi)``, or ``(nan, nan)`` when ``total`` is zero.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (
        z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    ) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def judge_human_agreement(
    model_decisions: Sequence[Any],
    human_decisions: Sequence[Any],
    *,
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Compare a model judge against human judgement on the audit sample.

    Args:
        model_decisions: Model-judge decisions on the audited items.
        human_decisions: Human decisions on the same items, in the same order.
        tolerance: For numeric ratings, the absolute difference treated as
            agreement. Ignored for non-numeric decisions.

    Returns:
        Mapping with the agreement rate, its 95% Wilson interval, the implied
        judge error rate, and directional bias.

    Raises:
        ValueError: If the sequences are not aligned.
    """
    if len(model_decisions) != len(human_decisions):
        raise ValueError(
            f"length mismatch: {len(model_decisions)} vs {len(human_decisions)}"
        )
    n = len(model_decisions)
    if n == 0:
        return {
            "n_audited": 0,
            "agreement_rate": float("nan"),
            "error_rate": float("nan"),
            "agreement_ci": [float("nan"), float("nan")],
            "sufficient": False,
        }

    agreements = 0
    model_higher = 0
    human_higher = 0
    for model, human in zip(model_decisions, human_decisions):
        if isinstance(model, (int, float)) and isinstance(human, (int, float)) and not isinstance(model, bool):
            if abs(float(model) - float(human)) <= tolerance:
                agreements += 1
            elif float(model) > float(human):
                model_higher += 1
            else:
                human_higher += 1
        else:
            if model == human:
                agreements += 1
            else:
                # Only meaningful for ordered decisions; recorded as a tie for
                # nominal ones so the bias figure stays interpretable.
                model_higher += int(bool(model) and not bool(human))
                human_higher += int(bool(human) and not bool(model))

    rate = agreements / n
    lo, hi = _wilson_interval(agreements, n)
    return {
        "n_audited": n,
        "agreement_rate": rate,
        "error_rate": 1.0 - rate,
        "agreement_ci": [lo, hi],
        "error_rate_upper_bound": 1.0 - lo,
        "model_judge_more_generous": model_higher / n,
        "human_judge_more_generous": human_higher / n,
        "directional_bias": (model_higher - human_higher) / n,
        "sufficient": n >= MIN_AUDIT_ITEMS,
        "note": (
            "error_rate_upper_bound uses the lower end of the agreement "
            "interval. Compare THAT against the benchmark's SESOI, not the "
            "point estimate: a judge whose error might exceed the effect size "
            "cannot establish the effect."
        ),
    }


def gate_primary_outcome(
    benchmark: str,
    sesoi: float,
    audit: Optional[Mapping[str, Any]],
    *,
    judge_is_human: bool = False,
) -> Dict[str, Any]:
    """Decide whether a judged primary outcome may be reported.

    Args:
        benchmark: Benchmark name, for the message.
        sesoi: The benchmark's smallest effect size of interest.
        audit: Output of :func:`judge_human_agreement`; ``None`` when no audit
            was run.
        judge_is_human: When the primary outcome comes from human raters
            directly, no model-judge audit is needed and this gate passes.

    Returns:
        Mapping with ``permitted`` and an explicit ``reason``.
    """
    if judge_is_human:
        return {
            "benchmark": benchmark,
            "permitted": True,
            "reason": (
                "Primary outcome is rated by humans directly; the model-judge "
                "audit gate does not apply. Inter-rater agreement is still "
                "required and reported separately."
            ),
        }

    if not audit:
        return {
            "benchmark": benchmark,
            "permitted": False,
            "reason": (
                "No human audit sample exists for this benchmark's model judge. "
                "Limitation L16 forbids using an LLM judge as a primary outcome "
                "without one. Report the outcome as secondary, or run the audit."
            ),
        }

    if not audit.get("sufficient"):
        return {
            "benchmark": benchmark,
            "permitted": False,
            "reason": (
                f"Audit sample of {audit.get('n_audited', 0)} items is below the "
                f"{MIN_AUDIT_ITEMS}-item minimum. The agreement interval is "
                "wider than any effect the study is powered to detect."
            ),
        }

    bound = audit.get("error_rate_upper_bound", float("nan"))
    if bound == bound and bound > sesoi:
        return {
            "benchmark": benchmark,
            "permitted": False,
            "reason": (
                f"Model-judge error could be as high as {bound:.3f}, above this "
                f"benchmark's smallest effect size of interest ({sesoi:.3f}). An "
                "effect of SESOI size cannot be distinguished from judge error, "
                "so no primary conclusion is drawn for this benchmark."
            ),
        }

    return {
        "benchmark": benchmark,
        "permitted": True,
        "reason": (
            f"Model-judge error bounded at {bound:.3f}, below the SESOI "
            f"({sesoi:.3f}). Primary outcome permitted with the audited "
            "agreement rate reported alongside it."
        ),
    }
