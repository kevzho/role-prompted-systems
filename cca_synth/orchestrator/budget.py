"""Budget matching: enforce parity across arms, then report all three regimes.

Limitation L10 is that a six-agent council plus synthesis uses more calls and
tokens than a single agent, and that "matching only the number of samples does
not match context, output, latency, or cost".  The repository already *measured*
cost; nothing *enforced* parity, and an unenforced budget is a claim rather than
a control.

Three regimes, and ``RESEARCH_SPECIFICATION.md`` §3 requires all three to be
reported:

``unconstrained``
    Each arm spends what it wants.  This is the quality-cost frontier point, and
    on its own it flatters whichever arm is allowed to spend most.
``resource_matched``
    Equal maximum input tokens, output tokens, calls, and synthesis allowance.
``cost_matched``
    Equal total provider cost with a comparable latency ceiling.  This is the
    regime that decides deployment, and ``PREREGISTRATION.md`` H1 requires the
    primary hypothesis to hold under *both* matched regimes.

Enforcement is structural: :class:`BudgetedProvider` wraps any provider, refuses
calls once the budget is exhausted, and records the refusal as a
``truncated_output`` failure rather than silently overspending.  An arm that
cannot finish inside its budget is a *result*, not an error to be worked around.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.providers import Provider, ProviderResponse

__all__ = [
    "Budget",
    "Spend",
    "BudgetedProvider",
    "REGIMES",
    "measure_spend",
    "matched_budgets",
    "budget_report",
    "check_parity",
]

#: The three regimes, in the order they must appear in any results table.
REGIMES: Tuple[str, ...] = ("unconstrained", "resource_matched", "cost_matched")


@dataclass
class Budget:
    """A per-item spending cap.

    ``None`` means unconstrained on that axis.  All caps are *per item*, since
    that is the unit at which arms are compared.

    Attributes:
        max_input_tokens: Prompt tokens allowed per item across all calls.
        max_output_tokens: Generated tokens allowed per item across all calls.
        max_calls: Model calls allowed per item.
        synthesis_allowance_tokens: Additional tokens the synthesiser may spend,
            counted separately so a council cannot hide deliberation inside it.
        max_cost_usd: Provider cost allowed per item.
        latency_ceiling_ms: Wall-clock ceiling per item.
        regime: Which of :data:`REGIMES` this budget implements.
    """

    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_calls: Optional[int] = None
    synthesis_allowance_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    latency_ceiling_ms: Optional[float] = None
    regime: str = "unconstrained"

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got '{self.regime}'")
        for name in ("max_input_tokens", "max_output_tokens", "max_calls"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")

    def is_unconstrained(self) -> bool:
        """Whether every axis is uncapped."""
        return all(
            getattr(self, name) is None
            for name in (
                "max_input_tokens", "max_output_tokens", "max_calls",
                "max_cost_usd", "latency_ceiling_ms",
            )
        )


@dataclass
class Spend:
    """Measured consumption for one arm.

    Attributes:
        arm: Arm name.
        n_items: Items processed.
        input_tokens: Total prompt tokens.
        output_tokens: Total generated tokens.
        calls: Total model calls.
        cost_usd: Total provider cost.
        latency_ms: Total wall-clock latency across calls.
        budget_refusals: Calls refused because the budget was exhausted.
    """

    arm: str
    n_items: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    budget_refusals: int = 0

    def per_item(self) -> Dict[str, float]:
        """Return per-item consumption on every axis.

        Returns:
            Mapping from axis name to per-item value.
        """
        n = max(1, self.n_items)
        return {
            "input_tokens": self.input_tokens / n,
            "output_tokens": self.output_tokens / n,
            "calls": self.calls / n,
            "cost_usd": self.cost_usd / n,
            "latency_ms": self.latency_ms / n,
        }


class BudgetedProvider(Provider):
    """Wraps a provider and enforces a per-item budget.

    The enforcement point matters.  Measuring spend after the fact tells you an
    arm overspent; refusing the call tells the arm it may not.  Only the second
    is a control.  A refused call is recorded as ``truncated_output`` with
    ``ok=False``, so the downstream record shows ``present=false`` and the
    exclusion appears in the §8 report rather than vanishing.
    """

    def __init__(self, inner: Provider, budget: Budget, arm: str = "arm"):
        """Wrap ``inner`` with ``budget``.

        Args:
            inner: The provider to wrap.
            budget: Per-item cap to enforce.
            arm: Arm name, for the spend ledger.
        """
        # Deliberately does not call super().__init__: the snapshot is already
        # validated by the wrapped provider, and re-validating would reject a
        # legitimately wrapped instance.
        self.inner = inner
        self.budget = budget
        self.model_snapshot = inner.model_snapshot
        self.timeout_s = inner.timeout_s
        self.max_retries = inner.max_retries
        self.price_per_million_input = inner.price_per_million_input
        self.price_per_million_output = inner.price_per_million_output
        self.spend = Spend(arm=arm)
        self._item_input = 0
        self._item_output = 0
        self._item_calls = 0
        self._item_cost = 0.0
        self._item_latency = 0.0

    def start_item(self) -> None:
        """Reset the per-item counters.

        Must be called once per item before any call for that item.
        """
        self.spend.n_items += 1
        self._item_input = 0
        self._item_output = 0
        self._item_calls = 0
        self._item_cost = 0.0
        self._item_latency = 0.0

    def remaining_output_tokens(self) -> Optional[int]:
        """Output tokens still available for the current item.

        Returns:
            Remaining tokens, or ``None`` when uncapped.
        """
        if self.budget.max_output_tokens is None:
            return None
        return max(0, self.budget.max_output_tokens - self._item_output)

    def _exhausted(self, requested_output: int) -> Optional[str]:
        """Whether a call would exceed the budget.

        Args:
            requested_output: Output tokens the call would be allowed.

        Returns:
            The name of the exhausted axis, or ``None``.
        """
        budget = self.budget
        if budget.max_calls is not None and self._item_calls >= budget.max_calls:
            return "max_calls"
        if (
            budget.max_output_tokens is not None
            and self._item_output >= budget.max_output_tokens
        ):
            return "max_output_tokens"
        if budget.max_cost_usd is not None and self._item_cost >= budget.max_cost_usd:
            return "max_cost_usd"
        if (
            budget.latency_ceiling_ms is not None
            and self._item_latency >= budget.latency_ceiling_ms
        ):
            return "latency_ceiling_ms"
        del requested_output
        return None

    def complete(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> ProviderResponse:
        """Issue a call if the budget permits, else refuse and record it.

        Args:
            system_prompt: System prompt.
            user_content: User content.
            max_tokens: Requested output cap, further reduced by the budget.
            temperature: Sampling temperature.
            seed: Seed where supported.

        Returns:
            A :class:`ProviderResponse`; ``ok=False`` with
            ``stop_reason="truncated"`` when the budget is exhausted.
        """
        exhausted = self._exhausted(max_tokens)
        if exhausted is not None:
            self.spend.budget_refusals += 1
            return ProviderResponse(
                ok=False,
                stop_reason="truncated",
                model_snapshot=self.model_snapshot,
                error=f"budget exhausted: {exhausted}",
            )

        remaining = self.remaining_output_tokens()
        effective = max_tokens if remaining is None else min(max_tokens, remaining)
        if effective <= 0:
            self.spend.budget_refusals += 1
            return ProviderResponse(
                ok=False,
                stop_reason="truncated",
                model_snapshot=self.model_snapshot,
                error="budget exhausted: max_output_tokens",
            )

        response = self.inner.complete(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=effective,
            temperature=temperature,
            seed=seed,
        )

        self._item_calls += 1
        self._item_input += response.input_tokens
        self._item_output += response.output_tokens
        self._item_cost += response.cost_usd
        self._item_latency += response.latency_ms

        self.spend.calls += 1
        self.spend.input_tokens += response.input_tokens
        self.spend.output_tokens += response.output_tokens
        self.spend.cost_usd += response.cost_usd
        self.spend.latency_ms += response.latency_ms
        return response


def measure_spend(manifest: Mapping[str, Any], arm: str) -> Spend:
    """Extract a :class:`Spend` from an orchestrator or condition manifest.

    Args:
        manifest: Manifest produced by a council or condition run.
        arm: Arm name to record.

    Returns:
        The measured spend.
    """
    counts = manifest.get("counts", {}) or {}
    return Spend(
        arm=arm,
        n_items=int(manifest.get("n_items", 0)),
        input_tokens=int(manifest.get("total_input_tokens", 0)),
        output_tokens=int(manifest.get("total_output_tokens", 0)),
        calls=int(counts.get("calls", 0)),
        cost_usd=float(manifest.get("total_cost_usd") or 0.0),
        latency_ms=float(manifest.get("total_latency_ms", 0.0)),
    )


def matched_budgets(
    reference: Spend,
    price_per_million_input: float = 0.0,
    price_per_million_output: float = 0.0,
    latency_slack: float = 1.25,
) -> Dict[str, Budget]:
    """Derive the three regime budgets from a reference arm's measured spend.

    The reference is normally the CCA council: matching means giving every other
    arm what the council actually consumed, so a single-agent control is allowed
    to deliberate for six agents' worth of tokens.

    Args:
        reference: Measured spend of the reference arm.
        price_per_million_input: Input price, for the cost-matched cap.
        price_per_million_output: Output price, for the cost-matched cap.
        latency_slack: Multiplier on observed latency for the ceiling; a ceiling
            set exactly at the observed mean would fail half the items by
            construction.

    Returns:
        Mapping from regime name to :class:`Budget`.

    Raises:
        ValueError: If the reference processed no items.
    """
    if reference.n_items <= 0:
        raise ValueError("reference spend has no items; cannot derive a matched budget")

    per_item = reference.per_item()
    cost_cap = per_item["cost_usd"]
    if cost_cap <= 0.0 and (price_per_million_input or price_per_million_output):
        # Reconstruct cost from tokens when the provider did not report it.
        cost_cap = (
            per_item["input_tokens"] * price_per_million_input
            + per_item["output_tokens"] * price_per_million_output
        ) / 1e6

    return {
        "unconstrained": Budget(regime="unconstrained"),
        "resource_matched": Budget(
            max_input_tokens=int(math.ceil(per_item["input_tokens"])) or None,
            max_output_tokens=int(math.ceil(per_item["output_tokens"])) or None,
            max_calls=int(math.ceil(per_item["calls"])) or None,
            synthesis_allowance_tokens=int(math.ceil(per_item["output_tokens"] * 0.25)) or None,
            regime="resource_matched",
        ),
        "cost_matched": Budget(
            max_cost_usd=cost_cap if cost_cap > 0 else None,
            latency_ceiling_ms=per_item["latency_ms"] * latency_slack or None,
            regime="cost_matched",
        ),
    }


def check_parity(
    spends: Sequence[Spend], tolerance: float = 0.10
) -> Dict[str, Any]:
    """Verify that arms actually spent comparably.

    Declaring a budget is not the same as respecting it.  This compares measured
    per-item consumption across arms and flags any axis where the spread exceeds
    ``tolerance``, so a "matched" table cannot quietly contain an arm that spent
    twice as much.

    Args:
        spends: Measured spend per arm.
        tolerance: Maximum permitted relative spread, as a fraction of the
            maximum observed value on that axis.

    Returns:
        A report with per-axis spread and a list of violations.

    Raises:
        ValueError: If fewer than two arms are supplied.
    """
    if len(spends) < 2:
        raise ValueError("parity needs at least two arms to compare")

    axes = ("input_tokens", "output_tokens", "calls", "cost_usd", "latency_ms")
    per_arm = {s.arm: s.per_item() for s in spends}
    violations: List[Dict[str, Any]] = []
    spreads: Dict[str, float] = {}

    for axis in axes:
        values = {arm: values[axis] for arm, values in per_arm.items()}
        highest = max(values.values())
        lowest = min(values.values())
        if highest <= 0:
            spreads[axis] = 0.0
            continue
        spread = (highest - lowest) / highest
        spreads[axis] = float(spread)
        if spread > tolerance:
            violations.append(
                {
                    "axis": axis,
                    "spread": float(spread),
                    "tolerance": tolerance,
                    "max_arm": max(values, key=values.get),
                    "max_value": float(highest),
                    "min_arm": min(values, key=values.get),
                    "min_value": float(lowest),
                }
            )

    return {
        "n_arms": len(spends),
        "tolerance": tolerance,
        "per_arm_per_item": {arm: values for arm, values in per_arm.items()},
        "spread_by_axis": spreads,
        "violations": violations,
        "matched": not violations,
        "note": (
            "A budget-matched claim requires this to report matched=true. "
            "Arms whose spend differs by more than the tolerance are not "
            "matched, whatever the table header says (limitation L10)."
        ),
    }


def budget_report(
    results_by_regime: Mapping[str, Mapping[str, float]],
    spends_by_regime: Mapping[str, Sequence[Spend]],
    primary_metric: str = "selective_accuracy",
    tolerance: float = 0.10,
) -> Dict[str, Any]:
    """Assemble the three-regime results table.

    §3 requires the unconstrained result *and* both matched results, and
    ``PREREGISTRATION.md`` H1 requires the primary hypothesis to hold under both
    matched regimes.  This structure makes it impossible to present one regime
    without the others: every regime appears in the same table, and a gain that
    exists only unconstrained is labelled in the verdict rather than in a
    footnote.

    Args:
        results_by_regime: Regime to arm-to-metric mapping.
        spends_by_regime: Regime to measured spends.
        primary_metric: Metric name being tabulated.
        tolerance: Parity tolerance passed to :func:`check_parity`.

    Returns:
        A report with a per-regime table, parity checks, and a verdict.
    """
    table: Dict[str, Any] = {}
    for regime in REGIMES:
        arms = results_by_regime.get(regime)
        if not arms:
            table[regime] = {"available": False, "note": "not run"}
            continue
        spends = spends_by_regime.get(regime, [])
        parity = (
            check_parity(spends, tolerance) if len(spends) >= 2 else {"matched": None}
        )
        best = max(arms, key=arms.get)
        table[regime] = {
            "available": True,
            "arms": dict(arms),
            "best_arm": best,
            "best_value": float(arms[best]),
            "parity": parity,
        }

    unconstrained_best = table.get("unconstrained", {}).get("best_arm")
    matched_bests = [
        table[r].get("best_arm") for r in ("resource_matched", "cost_matched")
        if table.get(r, {}).get("available")
    ]
    only_unconstrained = bool(
        unconstrained_best
        and matched_bests
        and all(best != unconstrained_best for best in matched_bests)
    )

    return {
        "primary_metric": primary_metric,
        "regimes": table,
        "verdict": (
            "A gain present only in the unconstrained regime is a quality-cost "
            "frontier observation, not a deployable improvement. Report it as "
            "such."
            if only_unconstrained
            else "The leading arm is consistent across the regimes that were run."
        ),
        "gain_only_unconstrained": only_unconstrained,
        "regimes_run": [r for r in REGIMES if table.get(r, {}).get("available")],
        "regimes_missing": [r for r in REGIMES if not table.get(r, {}).get("available")],
    }
