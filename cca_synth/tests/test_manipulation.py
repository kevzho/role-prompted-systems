"""Tests for the manipulation-check measurement path.

This is the primary outcome registered in ``PREREGISTRATION.md`` §11 A2.1, so
the tests here pin the parts of it that could fail silently:

* the A2.2 pass rule is conjunctive, and neither half alone certifies a role;
* the threshold is standardized, because the six checks do not share a unit;
* an item whose record is absent under any variant is *excluded* rather than
  scored as zero, which would count a refusal as a non-performed operation;
* the ``correct`` outcome the calibration check needs is derived from the gold
  label, since without it that check returns 0.0 and its role can never pass;
* the three-variant join emits only the intersection and reports what it dropped.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROLES  # noqa: E402
from orchestrator.manipulation import (  # noqa: E402
    _with_outcome,
    collect_variants,
    join_variant_records,
)
from prompts import PROMPT_VARIANTS  # noqa: E402
from prompts.manipulation_check import (  # noqa: E402
    MIN_STANDARDIZED_EFFECT,
    check_role,
    run_checks,
)


# ---------------------------------------------------------------------------
# The A2.2 decision rule
# ---------------------------------------------------------------------------


def _counterevidence_records(counts, key: str = "counterevidence"):
    """Records whose counterevidence check equals each given count."""
    return [{key: ["e"] * int(c), "atomic_claims": [], "present": True} for c in counts]


def test_pass_rule_needs_both_halves():
    """A tight interval around a trivially small standardized effect fails."""
    rng = np.random.default_rng(0)
    n = 2000
    # A real but small shift, on items that vary a lot independently. The
    # interval excludes zero because n is large; the standardized effect stays
    # low because the per-item difference is noisy. Note the variance has to be
    # in the *difference*: a shared per-item baseline cancels when paired.
    records = {
        "operation": _counterevidence_records(rng.integers(0, 12, size=n) + 1),
        "neutral": _counterevidence_records(rng.integers(0, 12, size=n)),
        "persona": _counterevidence_records(rng.integers(0, 12, size=n)),
    }
    result = check_role("counter_evidence_checker", records)
    assert result["vs_neutral"]["interval_excludes_zero"] is True
    assert result["vs_neutral"]["standardized_effect"] < MIN_STANDARDIZED_EFFECT
    assert result["passed"] is False, "an interval alone must not certify a role"


def test_large_effect_with_a_wide_interval_fails():
    """A big standardized effect on too few items does not pass either."""
    records = {
        "operation": _counterevidence_records([9, 0, 9]),
        "neutral": _counterevidence_records([0, 9, 0]),
        "persona": _counterevidence_records([0, 0, 0]),
    }
    result = check_role("counter_evidence_checker", records)
    if not result["vs_neutral"]["interval_excludes_zero"]:
        assert result["passed"] is False


def test_threshold_is_standardized_not_absolute():
    """Scaling a check's units must not change the verdict.

    The six checks return counts, bounded rates, and negative ECE. A verdict
    that moved when the unit changed would make the threshold meaningless
    across roles, which is why A2.1 records the heterogeneity and A2.2
    standardizes.
    """
    rng = np.random.default_rng(7)
    n = 300
    small = rng.integers(0, 3, size=n)
    verdicts = []
    for scale in (1, 10):
        records = {
            "operation": _counterevidence_records((small + 2) * scale),
            "neutral": _counterevidence_records(small * scale),
            "persona": _counterevidence_records(small * scale),
        }
        result = check_role("counter_evidence_checker", records)
        verdicts.append(result["passed"])
        # The raw difference scales by 10x; the standardized effect must not.
    assert verdicts[0] == verdicts[1]


def test_zero_variance_difference_is_not_an_error():
    """A constant shift on every item is a valid, very strong effect."""
    records = {
        "operation": _counterevidence_records([3] * 40),
        "neutral": _counterevidence_records([0] * 40),
        "persona": _counterevidence_records([0] * 40),
    }
    result = check_role("counter_evidence_checker", records)
    assert np.isinf(result["vs_neutral"]["standardized_effect"])
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# Absent records are excluded, never imputed
# ---------------------------------------------------------------------------


def test_absent_records_are_excluded_rather_than_scored_zero():
    """A refusal must not be counted as a performed-but-empty operation."""
    n = 30
    operation = _counterevidence_records([4] * n)
    neutral = _counterevidence_records([0] * n)
    persona = _counterevidence_records([0] * n)
    # Ten items refused under the neutral variant only.
    for index in range(10):
        neutral[index] = {"counterevidence": [], "atomic_claims": [], "present": False}

    result = check_role("counter_evidence_checker", {
        "operation": operation, "neutral": neutral, "persona": persona,
    })
    assert result["n_items"] == n - 10
    assert result["n_excluded_absent"] == 10
    assert result["n_items_collected"] == n
    # Had the absent records been scored as zero they would have *inflated* the
    # contrast, since neutral is the subtrahend. The reported mean must come
    # from the 20 paired items only.
    assert result["mean_neutral"] == pytest.approx(0.0)
    assert result["vs_neutral"]["diff"] == pytest.approx(4.0)


def test_a_role_absent_everywhere_is_reported_not_crashed():
    """Zero paired items yields an error result, not a nan table."""
    absent = [{"present": False} for _ in range(5)]
    result = check_role("counter_evidence_checker", {
        "operation": absent, "neutral": absent, "persona": absent,
    })
    assert result["passed"] is False
    assert result["n_items"] == 0
    assert "cannot be scored" in result["error"]


# ---------------------------------------------------------------------------
# The derived outcome the calibration check needs
# ---------------------------------------------------------------------------


def test_outcome_is_derived_from_the_gold_label():
    """``correct`` reflects the gold answer, and does not mutate the input."""
    candidate = {"role": ROLES[0], "present": True, "answer_option": 2}
    enriched = _with_outcome(candidate, {"gold_option": 2})
    assert enriched["correct"] is True
    assert _with_outcome(candidate, {"gold_option": 3})["correct"] is False
    assert "correct" not in candidate, "input record was mutated"


def test_unlabelled_and_absent_items_carry_no_outcome():
    """A missing gold or a missing record asserts nothing about correctness."""
    candidate = {"role": ROLES[0], "present": True, "answer_option": 1}
    assert "correct" not in _with_outcome(candidate, {"gold_option": -1})
    assert "correct" not in _with_outcome(candidate, {})
    absent = {"role": ROLES[0], "present": False, "answer_option": -1}
    assert "correct" not in _with_outcome(absent, {"gold_option": 1})


def test_calibration_check_is_scoreable_once_outcomes_exist():
    """Without ``correct`` the calibration check is identically zero.

    That made the confidence calibrator unpassable regardless of model
    behaviour, which is the defect the derived outcome closes.
    """
    from prompts import ROLE_PROMPTS

    check = ROLE_PROMPTS["confidence_calibrator"].check
    without = check([{"confidence": 0.9, "present": True}])
    assert without == 0.0
    with_outcome = check([{"confidence": 0.9, "correct": False, "present": True}])
    assert with_outcome < 0.0


# ---------------------------------------------------------------------------
# The three-variant join
# ---------------------------------------------------------------------------


def _write_variant(directory: str, variant: str, item_ids, gold: int = 1) -> None:
    """Write one variant's records file."""
    path = os.path.join(directory, f"records.{variant}.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for item_id in item_ids:
            handle.write(json.dumps({
                "item_id": item_id,
                "benchmark": "halueval",
                "gold_option": gold,
                "candidates": [
                    {"role": role, "present": True, "answer_option": 1,
                     "confidence": 0.6, "counterevidence": [], "atomic_claims": []}
                    for role in ROLES
                ],
            }) + "\n")


def test_join_emits_only_the_intersection_and_reports_drops():
    """An item missing from one variant is not a paired observation."""
    with tempfile.TemporaryDirectory() as directory:
        _write_variant(directory, "operation", ["a", "b", "c"])
        _write_variant(directory, "neutral", ["a", "b"])
        _write_variant(directory, "persona", ["a", "b", "c", "d"])
        summary = join_variant_records(directory, os.path.join(directory, "joined.jsonl"))

    assert summary["n_intersection"] == 2
    assert summary["n_written"] == 2
    assert summary["n_dropped_not_in_all_variants"]["persona"] == 2
    assert summary["n_dropped_not_in_all_variants"]["neutral"] == 0


def test_join_rejects_duplicate_item_ids():
    """A duplicated id would make the join ambiguous."""
    with tempfile.TemporaryDirectory() as directory:
        for variant in PROMPT_VARIANTS:
            _write_variant(directory, variant, ["a", "a"])
        with pytest.raises(ValueError, match="duplicate item_id"):
            join_variant_records(directory, os.path.join(directory, "joined.jsonl"))


def test_join_requires_every_variant():
    """A missing variant file fails loudly rather than scoring two variants."""
    with tempfile.TemporaryDirectory() as directory:
        _write_variant(directory, "operation", ["a"])
        with pytest.raises(FileNotFoundError, match="persona|neutral"):
            join_variant_records(directory, os.path.join(directory, "joined.jsonl"))


# ---------------------------------------------------------------------------
# End to end on the stub provider
# ---------------------------------------------------------------------------


def test_end_to_end_collect_join_score_on_the_stub():
    """The whole path runs offline and stamps its provenance."""
    with tempfile.TemporaryDirectory() as directory:
        items_path = os.path.join(directory, "items.jsonl")
        with open(items_path, "w", encoding="utf-8") as handle:
            for index in range(12):
                handle.write(json.dumps({
                    "item_id": f"i-{index}", "benchmark": "halueval",
                    "question": f"Is claim {index} supported?",
                    "options": ["yes", "no"], "gold": index % 2,
                }) + "\n")

        out_dir = os.path.join(directory, "checks")
        collection = collect_variants(
            items_path, out_dir, "stub", "stub-provider-2026-07-28", limit=12,
        )
        # Provenance is derived, and the stub is synthetic; a report from this
        # data must never be readable as empirical.
        assert collection["provenance"] == "simulated"
        assert set(collection["variants"]) == set(PROMPT_VARIANTS)

        joined = os.path.join(out_dir, "check_records.jsonl")
        join_variant_records(out_dir, joined)
        report = run_checks(joined)

    assert report["n_roles"] == len(ROLES)
    assert len(report["results"]) == len(ROLES)
    assert report["thresholds"]["min_standardized_effect"] == MIN_STANDARDIZED_EFFECT
    # The schema-compression caveat A2.1 registers must reach the artifact.
    assert any("amount rather than presence" in c for c in report["caveats"])
