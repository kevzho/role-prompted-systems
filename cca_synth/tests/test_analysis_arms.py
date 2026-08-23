"""Tests for the two arm-level analyses that fill section 5 of the paper.

Covers:

* :mod:`analysis_single_vs_council` -- item pairing, the two tie conventions,
  and the exclusion of items on which the baseline supplies no answer;
* :mod:`analysis_confidence` -- the pool-adjacent-violators isotonic fit, the
  invariance of AUROC under it, and the confidence/correctness summary.

Both are exercised on hand-built records whose expected values are worked out
in the test rather than taken from a run, so a change in behaviour fails here
rather than silently changing a published number.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis_confidence import (  # noqa: E402
    _apply_isotonic,
    _pool_adjacent_violators,
    auroc,
    confidence_report,
    expected_calibration_error,
)
from analysis_cross_arm import ARMS  # noqa: E402
from analysis_single_vs_council import (  # noqa: E402
    BASELINE,
    _baseline_answer,
    single_vs_council,
)


def _council_record(item_id, gold, votes, confidences=None):
    """A council record whose six slots cast ``votes``."""
    confidences = confidences or [0.99] * len(votes)
    return {
        "item_id": item_id,
        "benchmark": "halueval",
        "n_options": 2,
        "gold_option": gold,
        "base_answer_option": -1,
        "candidates": [
            {"role": f"role_{i}", "present": True, "abstain": False,
             "answer_option": vote, "confidence": confidence}
            for i, (vote, confidence) in enumerate(zip(votes, confidences))
        ],
    }


def _write_arms(root, council_votes, baseline_rows):
    """Materialise an ``arms/`` directory the analyses can read."""
    arms_dir = os.path.join(root, "arms")
    os.makedirs(arms_dir, exist_ok=True)
    for arm in ARMS:
        path = os.path.join(arms_dir, f"records.{arm}.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for record in council_votes[arm]:
                handle.write(json.dumps(record) + "\n")
    path = os.path.join(arms_dir, f"records.{BASELINE}.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for row in baseline_rows:
            handle.write(json.dumps(row) + "\n")


def test_baseline_answer_treats_abstention_as_no_value_not_a_wrong_answer():
    """An abstaining baseline leaves v(empty) undefined rather than scoring 0."""
    assert _baseline_answer({"present": True, "abstain": False, "answer_option": 1}) == 1
    assert _baseline_answer({"present": True, "abstain": True, "answer_option": 1}) == -1
    assert _baseline_answer({"present": False, "abstain": False, "answer_option": 1}) == -1
    assert _baseline_answer({"present": True, "abstain": False, "answer_option": -1}) == -1


def test_items_without_a_usable_baseline_leave_the_contrast_entirely():
    """A baseline abstention removes the item from every arm, not just one."""
    with tempfile.TemporaryDirectory() as root:
        councils = {arm: [_council_record(f"i{n}", 0, [0] * 6) for n in range(4)]
                    for arm in ARMS}
        baseline = [
            {"item_id": "i0", "present": True, "abstain": False, "answer_option": 0},
            {"item_id": "i1", "present": True, "abstain": True, "answer_option": -1},
            {"item_id": "i2", "present": True, "abstain": False, "answer_option": 0},
            {"item_id": "i3", "present": True, "abstain": False, "answer_option": 0},
        ]
        _write_arms(root, councils, baseline)
        report = single_vs_council(root)

    # i1 is dropped; the remaining three are scored in every arm.
    assert report["n_items_scorable_common"] == 3
    assert report["n_baseline_without_usable_answer"] == 1
    for row in report["primary_ties_count_as_failure"]:
        assert row["n_items"] == 3


def test_the_two_tie_conventions_differ_only_on_tied_items():
    """A 3-3 split scores wrong under the primary rule and vanishes under the other."""
    with tempfile.TemporaryDirectory() as root:
        # Three items the council gets right, one it ties on. Gold is always 0.
        votes = [[0] * 6, [0] * 6, [0] * 6, [0, 0, 0, 1, 1, 1]]
        councils = {arm: [_council_record(f"i{n}", 0, v) for n, v in enumerate(votes)]
                    for arm in ARMS}
        baseline = [{"item_id": f"i{n}", "present": True, "abstain": False,
                     "answer_option": 0} for n in range(4)]
        _write_arms(root, councils, baseline)
        report = single_vs_council(root)

    primary = report["primary_ties_count_as_failure"][0]
    robust = report["robustness_ties_excluded"][0]

    # The baseline answers all four correctly.
    assert report["direct_answer_accuracy"] == pytest.approx(1.0)
    # Primary: the tie is a miss, so the council scores 3/4 against the baseline's 4/4.
    assert primary["n_items"] == 4
    assert primary["council_accuracy"] == pytest.approx(0.75)
    assert primary["delta"] == pytest.approx(-0.25)
    assert primary["n_council_no_answer"] == 1
    # Robustness: the tied item leaves, and the two are then identical.
    assert robust["n_items"] == 3
    assert robust["council_accuracy"] == pytest.approx(1.0)
    assert robust["delta"] == pytest.approx(0.0)


def test_pool_adjacent_violators_returns_a_non_decreasing_fit():
    """The PAV fit is monotone and averages the blocks it pools."""
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    knots, values = _pool_adjacent_violators(x, y)
    assert np.all(np.diff(values) >= 0.0)
    # The middle inversion (1.0 then 0.0) pools to 0.5.
    assert _apply_isotonic(knots, values, np.array([0.2]))[0] == pytest.approx(0.5)
    assert _apply_isotonic(knots, values, np.array([0.3]))[0] == pytest.approx(0.5)


def test_isotonic_cuts_calibration_error_without_touching_discrimination():
    """A monotone map is free to fix ECE and cannot change AUROC."""
    rng = np.random.default_rng(0)
    correct = rng.integers(0, 2, size=4000).astype(np.float64)
    # Confidence carries real signal but sits in a compressed, badly-placed band.
    stated = np.where(correct == 1.0, 0.99, 0.98) + rng.normal(0, 0.001, size=4000)
    before_ece = expected_calibration_error(stated, correct)
    before_auroc = auroc(correct, stated)

    knots, values = _pool_adjacent_violators(stated, correct)
    mapped = _apply_isotonic(knots, values, stated)

    assert expected_calibration_error(mapped, correct) < before_ece / 2
    assert auroc(correct, mapped) == pytest.approx(before_auroc, abs=0.01)


def test_auroc_is_one_half_when_confidence_is_constant():
    """A flat confidence carries no ordering information."""
    correct = np.array([1.0, 0.0, 1.0, 0.0])
    assert auroc(correct, np.full(4, 0.99)) == pytest.approx(0.5)


def test_confidence_report_recovers_the_planted_gap():
    """Stated confidence minus realised accuracy is reported per arm."""
    with tempfile.TemporaryDirectory() as root:
        # Half the slots answer correctly; every slot states 0.99.
        records = []
        for n in range(10):
            votes = [0, 0, 0, 1, 1, 1]
            records.append(_council_record(f"i{n}", 0, votes))
        councils = {arm: records for arm in ARMS}
        baseline = [{"item_id": f"i{n}", "present": True, "abstain": False,
                     "answer_option": 0} for n in range(10)]
        _write_arms(root, councils, baseline)
        report = confidence_report(root)

    stats = report["per_arm"][ARMS[0]]
    assert stats["n_agent_answers"] == 60
    assert stats["accuracy"] == pytest.approx(0.5)
    assert stats["mean_stated_confidence"] == pytest.approx(0.99)
    assert stats["gap"] == pytest.approx(0.49)
    # Confidence is constant, so it orders nothing.
    assert stats["auroc"] == pytest.approx(0.5)


def test_confidence_report_is_labelled_exploratory():
    """The artifact must carry its own non-pre-registered warning."""
    with tempfile.TemporaryDirectory() as root:
        councils = {arm: [_council_record("i0", 0, [0] * 6)] for arm in ARMS}
        baseline = [{"item_id": "i0", "present": True, "abstain": False,
                     "answer_option": 0}]
        _write_arms(root, councils, baseline)
        report = confidence_report(root)
    assert "EXPLORATORY" in report["note"]


def _report(tmpdir, name, rows):
    """Write a minimal manipulation report for cross-family testing."""
    import json
    import os

    os.makedirs(os.path.join(tmpdir, name), exist_ok=True)
    # vs_neutral carries the raw difference and its bootstrap bounds as well as
    # the standardised effect; the figure path rescales the interval into d units.
    payload = {"results": [
        {"role": role, "passed": passed, "q_value": q,
         "vs_neutral": {"standardized_effect": d, "diff": d,
                        "lo": d - 0.1, "hi": d + 0.1}}
        for role, d, passed, q in rows
    ]}
    with open(os.path.join(tmpdir, name, "manipulation_report.json"), "w") as fh:
        json.dump(payload, fh)
    return os.path.join(tmpdir, name)


def test_cross_family_separates_replication_from_divergence():
    """Sign flips, agreements, and null-on-both are classified distinctly."""
    import tempfile

    from analysis_cross_family import cross_family

    with tempfile.TemporaryDirectory() as root:
        a = _report(root, "primary", [
            ("passes_both",   1.03, True,  0.001),
            ("reverses_both", -0.95, False, 0.001),
            ("flips",         -0.18, False, 0.016),
            ("tiny",          -0.06, False, 0.55),
        ])
        b = _report(root, "replication", [
            ("passes_both",   0.52, True,  0.001),
            ("reverses_both", -1.66, False, 0.001),
            ("flips",         0.69, True,  0.001),
            ("tiny",          0.07, False, 0.33),
        ])
        report = cross_family(a, b)

    by_role = {r["role"]: r for r in report["roles"]}
    assert by_role["passes_both"]["agreement"] == "replicates"
    assert by_role["reverses_both"]["agreement"] == "replicates"
    assert by_role["flips"]["agreement"] == "sign_flip"
    # Two nulls of opposite sign are noise, not a divergence.
    assert by_role["tiny"]["agreement"] == "null_on_both"
    assert report["n_sign_flips"] == 1
    assert report["n_passed_primary"] == 1
    assert report["n_passed_replication"] == 2


def test_cross_family_does_not_manufacture_disagreement_from_noise():
    """A sign flip needs a real effect on at least one side."""
    import tempfile

    from analysis_cross_family import NEGLIGIBLE_D, cross_family

    with tempfile.TemporaryDirectory() as root:
        a = _report(root, "primary", [("r", NEGLIGIBLE_D - 0.01, False, 0.9)])
        b = _report(root, "replication", [("r", -(NEGLIGIBLE_D - 0.01), False, 0.9)])
        assert cross_family(a, b)["n_sign_flips"] == 0

        c = _report(root, "primary2", [("r", NEGLIGIBLE_D + 0.5, True, 0.01)])
        d = _report(root, "replication2", [("r", -(NEGLIGIBLE_D + 0.5), False, 0.01)])
        assert cross_family(c, d)["n_sign_flips"] == 1
