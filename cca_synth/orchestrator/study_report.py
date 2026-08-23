"""Build the pre-registered pass-check × moves-score classification.

This is the executable bridge between ``manipulation_report.json`` and the
leave-one-out section of ``ablation.json``.  PREREGISTRATION.md A2.2 fixes the
classification before results exist, so the paper table must be generated from
artifacts rather than assembled by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROLES
from utils import write_json

__all__ = ["build_h7_classification", "main"]


def _index_unique(rows: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Mapping[str, Any]]:
    """Index rows by role and reject missing, unknown, or duplicate roles."""
    indexed: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if "role" not in row:
            raise ValueError(f"{label} contains a row without 'role'")
        role = str(row["role"])
        if role not in ROLES:
            raise ValueError(f"{label} contains unknown role '{role}'")
        if role in indexed:
            raise ValueError(f"{label} contains duplicate role '{role}'")
        indexed[role] = row
    missing = set(ROLES) - set(indexed)
    if missing:
        raise ValueError(f"{label} is missing roles: {sorted(missing)}")
    return indexed


def build_h7_classification(
    manipulation_report: Mapping[str, Any],
    ablation_report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Join check and benchmark verdicts under the registered H7 rule."""
    checks = _index_unique(manipulation_report.get("results", []), "manipulation report")
    loo = ablation_report.get("leave_one_out")
    if not isinstance(loo, Mapping):
        raise ValueError("ablation report has no leave_one_out result")
    effects = _index_unique(loo.get("roles", []), "leave-one-out report")

    cells = {
        "passes_and_moves": 0,
        "passes_not_moves": 0,
        "fails_and_moves": 0,
        "fails_not_moves": 0,
    }
    rows = []
    for role in ROLES:
        check = checks[role]
        effect = effects[role]
        passed = bool(check.get("passed", False))
        moves = bool(effect.get("moves_benchmark", False))
        if passed and moves:
            cell = "passes_and_moves"
        elif passed:
            cell = "passes_not_moves"
        elif moves:
            cell = "fails_and_moves"
        else:
            cell = "fails_not_moves"
        cells[cell] += 1
        rows.append(
            {
                "role": role,
                "passes_check": passed,
                "moves_benchmark": moves,
                "cell": cell,
                "check_effect": check.get("vs_neutral", {}).get("diff"),
                "check_ci": [
                    check.get("vs_neutral", {}).get("lo"),
                    check.get("vs_neutral", {}).get("hi"),
                ],
                "loo_contrast": effect.get("loo_contrast"),
                "loo_ci": [effect.get("ci_lo"), effect.get("ci_hi")],
            }
        )

    n_off_diagonal = cells["passes_not_moves"] + cells["fails_and_moves"]
    return {
        "hypothesis": "H7",
        "classification_rule": (
            "passes_check is read from the pre-registered manipulation check; "
            "moves_benchmark requires the leave-one-out interval to exclude zero "
            "and its point estimate to meet the registered threshold"
        ),
        "moves_threshold": loo.get("moves_threshold"),
        "n_roles": len(ROLES),
        "cells": cells,
        "n_off_diagonal": n_off_diagonal,
        "h7_supported": n_off_diagonal >= 1,
        "strong_off_diagonal_evidence": cells["fails_and_moves"],
        "weak_power_confounded_off_diagonal": cells["passes_not_moves"],
        "rows": rows,
        "interpretation_caveat": (
            "Per PREREGISTRATION.md A2.5, fails-and-moves is the strong "
            "off-diagonal result. Passes-without-moving is weak evidence because "
            "the scoped benchmark arm is underpowered below about 0.048."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manipulation", required=True)
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    with open(args.manipulation, "r", encoding="utf-8") as handle:
        manipulation = json.load(handle)
    with open(args.ablation, "r", encoding="utf-8") as handle:
        ablation = json.load(handle)
    report = build_h7_classification(manipulation, ablation)
    write_json(report, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
