"""Tests for the P15 failure analysis (limitations L12, L13).

The property that matters: **a benefit purchased with a registered regression
must not read as a pass**, and a negative primary result must not be quietly
replaced by a more flattering comparison. Both are enforced by computation here
rather than by the author's discipline at writing time.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from failure_analysis import (  # noqa: E402
    PREDICTED_HARMS,
    failure_analysis_report,
    off_target_report,
    primary_result_verdict,
    role_regression_report,
)


def _clean_measurements() -> dict:
    """Every role comfortably inside its registered threshold."""
    return {
        "counter_evidence_checker": {"coverage_drop_vs_control": 0.01},
        "consistency_checker": {"p95_latency_ratio_vs_single_agent": 1.4},
        "local_claim_checker": {"long_range_accuracy_drop": 0.005},
        "explore_then_prune": {"agent_dominance": 0.30, "council_token_share": 0.20},
        "rare_hypothesis_checker": {"fabricated_alternative_rate": 0.01},
        "confidence_calibrator": {"capitulation_when_right_delta": 0.01},
    }


def _positive_shapley() -> dict:
    """All six roles showing a benefit."""
    return {h.role: {"shapley": 0.05, "ci_lo": 0.02, "ci_hi": 0.08} for h in PREDICTED_HARMS}


# ---------------------------------------------------------------------------
# Registered thresholds
# ---------------------------------------------------------------------------


def test_all_six_roles_have_a_registered_harm():
    """L13 requires a predicted harm for every role, not just some."""
    assert len(PREDICTED_HARMS) == 6
    assert len({h.role for h in PREDICTED_HARMS}) == 6
    for harm in PREDICTED_HARMS:
        assert harm.threshold > 0
        assert harm.unacceptable_if


def test_clean_run_passes_every_role():
    """The happy path must be reachable."""
    report = role_regression_report(_clean_measurements(), _positive_shapley())
    assert report["n_passed"] == 6
    assert report["n_failed"] == 0
    assert report["failing_roles"] == []


def test_breaching_a_threshold_fails_the_role():
    """A benefit purchased with a registered regression is not a pass."""
    measurements = _clean_measurements()
    measurements["local_claim_checker"]["long_range_accuracy_drop"] = 0.09
    report = role_regression_report(measurements, _positive_shapley())
    assert "local_claim_checker" in report["failing_roles"]
    row = next(r for r in report["roles"] if r["role"] == "local_claim_checker")
    assert row["verdict"] == "FAIL"
    assert "Long-range accuracy drops >0.03" in row["reason"]


def test_unmeasured_harm_is_not_a_pass():
    """L13 is about what was checked; silence is not evidence of safety."""
    measurements = _clean_measurements()
    del measurements["rare_hypothesis_checker"]["fabricated_alternative_rate"]
    report = role_regression_report(measurements, _positive_shapley())
    row = next(r for r in report["roles"] if r["role"] == "rare_hypothesis_checker")
    assert row["verdict"] == "UNMEASURED"
    assert row["harm_breached"] is None
    assert "rare_hypothesis_checker" in report["unmeasured_roles"]


def test_nan_measurement_counts_as_unmeasured():
    """A nan is missing data, not a zero."""
    measurements = _clean_measurements()
    measurements["consistency_checker"]["p95_latency_ratio_vs_single_agent"] = float("nan")
    report = role_regression_report(measurements, _positive_shapley())
    row = next(r for r in report["roles"] if r["role"] == "consistency_checker")
    assert row["verdict"] == "UNMEASURED"


def test_explore_then_prune_disjunction_second_limb_fires():
    """Its registered rule is dominance >0.60 OR >35% of council tokens."""
    measurements = _clean_measurements()
    measurements["explore_then_prune"] = {
        "agent_dominance": 0.30,        # within threshold
        "council_token_share": 0.55,    # breaches the other limb
    }
    report = role_regression_report(measurements, _positive_shapley())
    row = next(r for r in report["roles"] if r["role"] == "explore_then_prune")
    assert row["verdict"] == "FAIL"
    assert "Token-share limb" in row["reason"]


def test_no_benefit_is_distinguished_from_a_regression():
    """A role with no target benefit is a different finding from a harmful one."""
    shapley = _positive_shapley()
    shapley["confidence_calibrator"] = {"shapley": 0.001, "ci_lo": -0.02, "ci_hi": 0.03}
    report = role_regression_report(_clean_measurements(), shapley)
    row = next(r for r in report["roles"] if r["role"] == "confidence_calibrator")
    assert row["verdict"] == "NO BENEFIT"
    assert row["harm_breached"] is False


# ---------------------------------------------------------------------------
# Off-target effects (L12)
# ---------------------------------------------------------------------------


def _family(cells) -> dict:
    """Wrap cells in the agent_by_benchmark_family shape."""
    return {"cells": list(cells), "alpha": 0.05, "n_tests": len(cells)}


def test_off_target_harms_are_surfaced():
    """The cells a biased write-up would skip."""
    family = _family([
        {"role": "counter_evidence_checker", "benchmark": "truthfulqa",
         "predeclared_primary": True, "marginal_value": 0.06,
         "significant_after_fdr": True, "q_value": 0.01},
        {"role": "counter_evidence_checker", "benchmark": "math_aime",
         "predeclared_primary": False, "marginal_value": -0.04,
         "significant_after_fdr": True, "q_value": 0.02},
    ])
    report = off_target_report(family)
    assert report["n_significant_off_target_harms"] == 1
    assert report["off_target_harms"][0]["benchmark"] == "math_aime"


def test_role_helping_only_off_target_is_named():
    """Helping somewhere other than the predeclared target validates nothing."""
    family = _family([
        {"role": "local_claim_checker", "benchmark": "advconsistency",
         "predeclared_primary": True, "marginal_value": 0.001,
         "significant_after_fdr": False, "q_value": 0.7},
        {"role": "local_claim_checker", "benchmark": "halueval",
         "predeclared_primary": False, "marginal_value": 0.05,
         "significant_after_fdr": True, "q_value": 0.01},
    ])
    report = off_target_report(family)
    assert report["roles_helping_only_off_target"] == ["local_claim_checker"]


def test_nonsignificant_negatives_are_not_reported_as_harms():
    """A negative point estimate that fails FDR is not a finding."""
    family = _family([
        {"role": "explore_then_prune", "benchmark": "halueval",
         "predeclared_primary": False, "marginal_value": -0.02,
         "significant_after_fdr": False, "q_value": 0.4},
    ])
    assert off_target_report(family)["n_significant_off_target_harms"] == 0


# ---------------------------------------------------------------------------
# The stopping rule
# ---------------------------------------------------------------------------


def test_h1_supported_when_cost_matched_clears_the_sesoi():
    """The positive path must be reachable."""
    verdict = primary_result_verdict(
        [{"comparator": "cost_matched_best_control", "diff": 0.055,
          "lo": 0.021, "hi": 0.089}]
    )
    assert verdict["h1_supported"] is True
    assert "exceeds the cost-matched control" in verdict["permitted_headline"]


def test_interval_including_zero_falsifies_h1():
    """Registered falsification condition."""
    verdict = primary_result_verdict(
        [{"comparator": "cost_matched_best_control", "diff": 0.04,
          "lo": -0.01, "hi": 0.09}]
    )
    assert verdict["h1_supported"] is False
    assert "Negative primary result" in verdict["permitted_headline"]
    assert "includes zero" in verdict["permitted_headline"]


def test_point_estimate_below_sesoi_falsifies_h1():
    """A real but decision-irrelevant gain is still a falsification."""
    verdict = primary_result_verdict(
        [{"comparator": "cost_matched_best_control", "diff": 0.012,
          "lo": 0.004, "hi": 0.020}]
    )
    assert verdict["h1_supported"] is False
    assert "below the SESOI" in verdict["permitted_headline"]


def test_unconstrained_result_cannot_be_substituted():
    """The stopping rule's whole purpose."""
    verdict = primary_result_verdict([
        {"comparator": "unconstrained_best_control", "diff": 0.20,
         "lo": 0.15, "hi": 0.25},
    ])
    assert verdict["h1_testable"] is False
    assert verdict["h1_supported"] is False
    assert "forbidden" in verdict["permitted_headline"]


# ---------------------------------------------------------------------------
# Assembled report
# ---------------------------------------------------------------------------


def test_report_states_a_negative_result_first():
    """If the council loses, that is the headline."""
    report = failure_analysis_report(
        _family([]),
        _clean_measurements(),
        [{"comparator": "cost_matched_best_control", "diff": 0.005,
          "lo": -0.02, "hi": 0.03}],
        _positive_shapley(),
    )
    assert report["primary"]["h1_supported"] is False
    assert "Negative primary result" in report["permitted_claims"][0]


def test_report_names_failing_and_unmeasured_roles():
    """Both failure modes appear in the permitted claims."""
    measurements = _clean_measurements()
    measurements["rare_hypothesis_checker"]["fabricated_alternative_rate"] = 0.30
    del measurements["consistency_checker"]["p95_latency_ratio_vs_single_agent"]

    report = failure_analysis_report(
        _family([]),
        measurements,
        [{"comparator": "cost_matched_best_control", "diff": 0.06,
          "lo": 0.04, "hi": 0.08}],
        _positive_shapley(),
    )
    joined = " ".join(report["permitted_claims"])
    assert "rare_hypothesis_checker" in joined
    assert "consistency_checker" in joined
    assert "No pass may be claimed" in joined


def test_clean_report_says_so_plainly():
    """A clean run should not read as though something were hidden."""
    report = failure_analysis_report(
        _family([]),
        _clean_measurements(),
        [{"comparator": "cost_matched_best_control", "diff": 0.06,
          "lo": 0.04, "hi": 0.08}],
        _positive_shapley(),
    )
    assert "No role breached its registered harm threshold." in report["permitted_claims"]
