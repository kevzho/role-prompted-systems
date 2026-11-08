"""Regression tests for the exploratory cross-role audit (analysis_cross_role.py).

These pin the numbers the AES poster and extended abstract report to the shipped
records, so a change to a check function or to the analysis cannot silently move
them. They read only shipped files and make no calls.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analysis_cross_role as acr  # noqa: E402

FAMILIES = {"checks": "cross_role.json", "checks-grok": "cross_role_grok.json"}


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def family(request):
    fam = request.param
    rows = acr._load(os.path.join(HERE, "runs", fam, "check_records.jsonl"))
    with open(os.path.join(HERE, "runs", fam, "manipulation_report.json"), encoding="utf-8") as f:
        registered = {r["role"]: r for r in json.load(f)["results"]}
    with open(os.path.join(HERE, "runs", FAMILIES[fam]), encoding="utf-8") as f:
        shipped = json.load(f)
    return fam, rows, registered, shipped, acr.analyse(rows)


def test_diagonal_reproduces_the_registered_effects(family):
    _, _, registered, _, report = family
    for role in acr.ROLES:
        assert report["matrix"][f"{role}|{role}"]["d"] == pytest.approx(
            registered[role]["vs_neutral"]["standardized_effect"], abs=1e-12)


def test_recomputation_matches_the_shipped_report(family):
    _, _, _, shipped, report = family
    for key, cell in shipped["matrix"].items():
        assert report["matrix"][key]["d"] == pytest.approx(cell["d"], abs=1e-12)
        assert report["matrix"][key]["q36"] == pytest.approx(cell["q36"], abs=1e-12)
    for section in ("discriminant", "signed_discriminant"):
        for role, entry in shipped[section].items():
            assert report[section][role]["supported"] == entry["supported"]
            assert report[section][role]["margin"] == pytest.approx(entry["margin"], abs=1e-12)
    assert report["guard_terms"] == shipped["guard_terms"]


def test_discriminant_counts_as_reported(family):
    fam, _, _, _, report = family
    unsigned = [r for r, v in report["discriminant"].items() if v["supported"]]
    signed = [r for r, v in report["signed_discriminant"].items() if v["supported"]]
    # 1 of 12 cells unsigned (terra counter-evidence only); counter-evidence on
    # both families once only same-direction cross-effects compete.
    assert unsigned == (["counter_evidence_checker"] if fam == "checks" else [])
    assert signed == ["counter_evidence_checker"]


def test_local_claim_guard_factor_is_constant(family):
    fam, _, _, _, report = family
    counts = report["guard_terms"]["local_claim_factor_counts"]
    # 600 + 591 = the "1,191 of 1,200" records in which the factor is exactly 2.
    assert counts.get("2.000") == (600 if fam == "checks" else 591)
    assert report["claim_count"]["local_claim_checker"] > 0.99


def test_bare_role_names_barely_move_their_own_checks(family):
    _, _, _, _, report = family
    assert max(abs(v["d"]) for v in report["persona_vs_neutral"].values()) <= 0.30


def test_guard_stress_reproduces_the_defeatable_cases(family):
    fam, _, _, _, report = family
    stress = report["guard_stress"]
    assert stress["consistency_flag_all_never_abstain"] == 1.0
    cal = stress["calibration_constant_shift"]
    if fam == "checks":
        # Manuscript appendix: 0.144 -> 0.112 (operation), 0.129 -> 0.065 (neutral).
        assert round(cal["operation"]["ece_unshifted"], 3) == 0.144
        assert round(cal["operation"]["ece_by_shift"]["-0.05"], 3) == 0.112
        assert round(cal["neutral"]["ece_unshifted"], 3) == 0.129
        assert round(cal["neutral"]["ece_by_shift"]["-0.10"], 3) == 0.065
        assert cal["operation"]["defeated"] and cal["neutral"]["defeated"]
    else:
        # On grok no downward shift helps the operation arm: the cheap strategy
        # that defeats the guard on terra does not transfer.
        assert all(cal["operation"]["ece_by_shift"][s] >= cal["operation"]["ece_unshifted"]
                   for s in ("-0.05", "-0.10", "-0.15", "-0.20"))
