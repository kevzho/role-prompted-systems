"""Tests for budget matching.

The distinction under test is between *measuring* spend and *enforcing* it.
Measuring tells you an arm overspent; refusing the call tells the arm it may
not. Only the second is a control, and limitation L10 asks for a control.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROLES  # noqa: E402
from orchestrator import Council, Item, StubProvider  # noqa: E402
from orchestrator.budget import (  # noqa: E402
    REGIMES,
    Budget,
    BudgetedProvider,
    Spend,
    budget_report,
    check_parity,
    matched_budgets,
    measure_spend,
)
from orchestrator.providers import Provider, ProviderResponse  # noqa: E402


class _CountingProvider(Provider):
    """Provider that always succeeds and reports fixed usage."""

    def __init__(self, output_tokens: int = 100) -> None:
        super().__init__("counting-2026-07-28")
        self.output_tokens = output_tokens
        self.calls = 0
        self.max_tokens_seen: list = []

    def complete(self, system_prompt, user_content, max_tokens=1024,
                 temperature=0.0, seed=None):
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        return ProviderResponse(
            text=json.dumps({"answer_option": 0, "abstain": False, "confidence": 0.5,
                             "atomic_claims": [], "counterevidence": [],
                             "failure_flags": []}),
            input_tokens=50,
            output_tokens=min(self.output_tokens, max_tokens),
            latency_ms=100.0,
            cost_usd=0.001,
            model_snapshot=self.model_snapshot,
        )


# ---------------------------------------------------------------------------
# Budget validation
# ---------------------------------------------------------------------------


def test_budget_rejects_an_unknown_regime() -> None:
    """A regime outside the declared three must not be constructible."""
    with pytest.raises(ValueError, match="regime must be one of"):
        Budget(regime="best_effort")


def test_budget_rejects_non_positive_caps() -> None:
    """A zero cap would silently disable an arm rather than constrain it."""
    with pytest.raises(ValueError, match="must be positive"):
        Budget(max_calls=0, regime="resource_matched")


def test_unconstrained_budget_is_detected() -> None:
    """The unconstrained regime must report itself as uncapped."""
    assert Budget(regime="unconstrained").is_unconstrained()
    assert not Budget(max_calls=6, regime="resource_matched").is_unconstrained()


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


def test_call_budget_is_enforced_not_merely_measured() -> None:
    """Once max_calls is reached, further calls must be refused."""
    inner = _CountingProvider()
    provider = BudgetedProvider(inner, Budget(max_calls=3, regime="resource_matched"))
    provider.start_item()

    for _ in range(6):
        provider.complete("sys", "user")

    assert inner.calls == 3, "provider was called past its budget"
    assert provider.spend.budget_refusals == 3


def test_budget_refusal_is_recorded_as_truncation() -> None:
    """A refused call must surface as an exclusion, not vanish."""
    provider = BudgetedProvider(
        _CountingProvider(), Budget(max_calls=1, regime="resource_matched")
    )
    provider.start_item()
    provider.complete("sys", "user")
    refused = provider.complete("sys", "user")

    assert refused.ok is False
    assert refused.stop_reason == "truncated"
    assert "budget exhausted" in refused.error


def test_output_token_budget_shrinks_the_request() -> None:
    """Remaining budget must cap max_tokens, not just stop calls."""
    inner = _CountingProvider(output_tokens=100)
    provider = BudgetedProvider(
        inner, Budget(max_output_tokens=250, regime="resource_matched")
    )
    provider.start_item()

    for _ in range(4):
        provider.complete("sys", "user", max_tokens=100)

    # 100 + 100 + 50 consumed, then refused.
    assert provider.spend.output_tokens == 250
    assert inner.max_tokens_seen == [100, 100, 50]
    assert provider.spend.budget_refusals == 1


def test_cost_budget_is_enforced() -> None:
    """Cost-matched runs must stop at the dollar cap."""
    inner = _CountingProvider()
    provider = BudgetedProvider(inner, Budget(max_cost_usd=0.0025, regime="cost_matched"))
    provider.start_item()
    for _ in range(6):
        provider.complete("sys", "user")
    assert inner.calls == 3  # 3 x 0.001 reaches the cap
    assert provider.spend.budget_refusals == 3


def test_start_item_resets_the_per_item_budget() -> None:
    """Budgets are per item; a new item must get a fresh allowance."""
    inner = _CountingProvider()
    provider = BudgetedProvider(inner, Budget(max_calls=2, regime="resource_matched"))

    for _ in range(3):
        provider.start_item()
        for _ in range(4):
            provider.complete("sys", "user")

    assert inner.calls == 6  # 2 per item x 3 items
    assert provider.spend.n_items == 3


def test_unconstrained_budget_never_refuses() -> None:
    """The unconstrained regime must not interfere with any call."""
    inner = _CountingProvider()
    provider = BudgetedProvider(inner, Budget(regime="unconstrained"))
    provider.start_item()
    for _ in range(10):
        provider.complete("sys", "user")
    assert inner.calls == 10
    assert provider.spend.budget_refusals == 0


def test_budgeted_provider_works_inside_a_council() -> None:
    """Enforcement must compose with the orchestrator, not just in isolation."""
    inner = StubProvider(seed=1, failure_rate=0.0, malformed_rate=0.0, truncation_rate=0.0)
    provider = BudgetedProvider(inner, Budget(max_calls=3, regime="resource_matched"))
    council = Council({role: provider for role in ROLES})

    provider.start_item()
    record = council.run_item(
        Item(item_id="b-1", benchmark="math_aime", question="q",
             options=["a", "b", "c"], gold=0)
    )
    present = sum(1 for c in record["candidates"] if c["present"])
    assert present == 3, "council exceeded its call budget"
    assert provider.spend.budget_refusals == 3


# ---------------------------------------------------------------------------
# Deriving matched budgets
# ---------------------------------------------------------------------------


def test_matched_budgets_cover_all_three_regimes() -> None:
    """Section 3 requires all three; none may be silently omitted."""
    reference = Spend(arm="council", n_items=100, input_tokens=70000,
                      output_tokens=240000, calls=600, cost_usd=3.6, latency_ms=180000)
    budgets = matched_budgets(reference)
    assert set(budgets) == set(REGIMES)
    assert budgets["unconstrained"].is_unconstrained()
    assert budgets["resource_matched"].max_calls == 6
    assert budgets["resource_matched"].max_output_tokens == 2400
    assert budgets["cost_matched"].max_cost_usd == pytest.approx(0.036)


def test_cost_matched_budget_reconstructs_missing_cost() -> None:
    """A provider that reports no cost must not yield an uncapped budget."""
    reference = Spend(arm="council", n_items=10, input_tokens=1000,
                      output_tokens=5000, calls=60, cost_usd=0.0)
    budgets = matched_budgets(
        reference, price_per_million_input=3.0, price_per_million_output=15.0
    )
    expected = (100 * 3.0 + 500 * 15.0) / 1e6
    assert budgets["cost_matched"].max_cost_usd == pytest.approx(expected)


def test_matched_budgets_reject_an_empty_reference() -> None:
    """Deriving a budget from nothing must fail loudly."""
    with pytest.raises(ValueError, match="no items"):
        matched_budgets(Spend(arm="council", n_items=0))


def test_latency_ceiling_has_slack() -> None:
    """A ceiling set at the observed mean would fail half the items."""
    reference = Spend(arm="council", n_items=10, latency_ms=10000, calls=60)
    budgets = matched_budgets(reference, latency_slack=1.25)
    assert budgets["cost_matched"].latency_ceiling_ms == pytest.approx(1250.0)


# ---------------------------------------------------------------------------
# Parity checking
# ---------------------------------------------------------------------------


def test_parity_detects_an_unmatched_arm() -> None:
    """A 'matched' table containing a double-spending arm must be flagged."""
    spends = [
        Spend(arm="council", n_items=10, output_tokens=6000, calls=60, cost_usd=0.1),
        Spend(arm="single", n_items=10, output_tokens=1000, calls=10, cost_usd=0.02),
    ]
    report = check_parity(spends, tolerance=0.10)
    assert report["matched"] is False
    axes = {v["axis"] for v in report["violations"]}
    assert {"output_tokens", "calls"} <= axes


def test_parity_passes_when_arms_actually_match() -> None:
    """Arms within tolerance on every axis must report matched."""
    spends = [
        Spend(arm="a", n_items=10, input_tokens=1000, output_tokens=2000,
              calls=20, cost_usd=0.10, latency_ms=5000),
        Spend(arm="b", n_items=10, input_tokens=1050, output_tokens=2050,
              calls=21, cost_usd=0.104, latency_ms=5100),
    ]
    report = check_parity(spends, tolerance=0.10)
    assert report["matched"] is True
    assert not report["violations"]


def test_parity_needs_two_arms() -> None:
    """Parity is a comparison; one arm cannot be matched to itself."""
    with pytest.raises(ValueError, match="at least two arms"):
        check_parity([Spend(arm="only", n_items=5)])


# ---------------------------------------------------------------------------
# The three-regime report
# ---------------------------------------------------------------------------


def test_report_flags_a_gain_that_exists_only_unconstrained() -> None:
    """The regime that decides deployment must not be buried in a footnote."""
    report = budget_report(
        results_by_regime={
            "unconstrained": {"council": 0.85, "single": 0.78},
            "resource_matched": {"council": 0.77, "single": 0.79},
            "cost_matched": {"council": 0.76, "single": 0.80},
        },
        spends_by_regime={},
    )
    assert report["gain_only_unconstrained"] is True
    assert "not a deployable improvement" in report["verdict"]


def test_report_accepts_a_consistent_winner() -> None:
    """A gain surviving both matched regimes must not be flagged."""
    report = budget_report(
        results_by_regime={
            "unconstrained": {"council": 0.85, "single": 0.70},
            "resource_matched": {"council": 0.81, "single": 0.72},
            "cost_matched": {"council": 0.80, "single": 0.73},
        },
        spends_by_regime={},
    )
    assert report["gain_only_unconstrained"] is False
    assert report["regimes_missing"] == []


def test_report_names_missing_regimes() -> None:
    """An incomplete run must say which regimes were not executed."""
    report = budget_report(
        results_by_regime={"unconstrained": {"council": 0.85, "single": 0.70}},
        spends_by_regime={},
    )
    assert report["regimes_run"] == ["unconstrained"]
    assert set(report["regimes_missing"]) == {"resource_matched", "cost_matched"}


def test_measure_spend_reads_a_manifest() -> None:
    """Spend must be recoverable from an orchestrator manifest."""
    manifest = {
        "n_items": 50,
        "counts": {"calls": 300},
        "total_output_tokens": 120000,
        "total_cost_usd": 1.8,
    }
    spend = measure_spend(manifest, arm="cca_council")
    assert spend.n_items == 50
    assert spend.calls == 300
    assert spend.per_item()["output_tokens"] == pytest.approx(2400.0)
