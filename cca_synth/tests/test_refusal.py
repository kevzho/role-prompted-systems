"""Pins the refusal numbers the manuscript reports to the shipped council records."""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analysis_refusal  # noqa: E402


@pytest.fixture(scope="module")
def report():
    return analysis_refusal.analyse(os.path.join(HERE, "runs"))


def test_no_refusals_on_halueval(report):
    assert all(r["halueval"]["refused"] == 0 for r in report["rates"].values())


def test_truthfulqa_refusal_counts(report):
    counts = {arm: r["truthfulqa"]["refused"] for arm, r in report["rates"].items()}
    assert counts == {"cca_council": 185, "neutral_council": 165, "persona_only_council": 136}


def test_persona_refuses_less_than_operation(report):
    c = {x["contrast"]: x for x in report["truthfulqa_contrasts"]}
    persona_vs_op = c["persona_only_council - cca_council"]
    assert persona_vs_op["diff_points"] == pytest.approx(-1.633, abs=1e-3)
    assert persona_vs_op["fdr_rejected"]
    assert not c["cca_council - neutral_council"]["fdr_rejected"]
