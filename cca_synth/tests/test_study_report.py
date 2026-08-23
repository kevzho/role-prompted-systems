"""Tests for the registered pass-check × moves-score classification."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROLES  # noqa: E402
from orchestrator.study_report import build_h7_classification  # noqa: E402


def _reports():
    checks = {
        "results": [
            {
                "role": role,
                "passed": index < 2,
                "vs_neutral": {"diff": index / 10, "lo": -0.1, "hi": 0.2},
            }
            for index, role in enumerate(ROLES)
        ]
    }
    ablation = {
        "leave_one_out": {
            "moves_threshold": 0.05,
            "roles": [
                {
                    "role": role,
                    "moves_benchmark": index in (1, 2),
                    "loo_contrast": index / 100,
                    "ci_lo": -0.01,
                    "ci_hi": 0.04,
                }
                for index, role in enumerate(ROLES)
            ],
        }
    }
    return checks, ablation


def test_h7_table_is_joined_by_role_not_row_position() -> None:
    checks, ablation = _reports()
    ablation["leave_one_out"]["roles"].reverse()
    report = build_h7_classification(checks, ablation)

    assert report["cells"] == {
        "passes_and_moves": 1,
        "passes_not_moves": 1,
        "fails_and_moves": 1,
        "fails_not_moves": 3,
    }
    assert report["n_off_diagonal"] == 2
    assert report["h7_supported"] is True
    assert report["strong_off_diagonal_evidence"] == 1
    assert report["weak_power_confounded_off_diagonal"] == 1


def test_h7_table_refuses_a_missing_role() -> None:
    checks, ablation = _reports()
    checks["results"].pop()
    with pytest.raises(ValueError, match="missing roles"):
        build_h7_classification(checks, ablation)
