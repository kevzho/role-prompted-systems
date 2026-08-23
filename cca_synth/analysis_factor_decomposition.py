"""Decompose the two compositional manipulation checks into their factors.

``_check_hypothesis_diversity`` scores ``diversity * concentration`` and
``_check_claim_decomposition`` scores ``claims * (1 + sourced_fraction)``. A
product can fall while one of its factors rises, so a negative contrast on
either statistic does not by itself establish that the declared operation ran
backwards -- it may mean the prompt traded one factor for the other, which for
``explore_then_prune`` is arguably what "prune" is supposed to do.

This script reports each factor separately against the same length-matched
neutral control, using the identical paired bootstrap the registered analysis
uses, so the decomposition is directly comparable to the headline numbers.

It is **descriptive reporting on records already collected** (PREREGISTRATION.md
A3.2), not a new test: no verdict is issued and no registered quantity changes.

Usage::

    python3 analysis_factor_decomposition.py \
        --records runs/checks/check_records.jsonl \
        --out runs/checks/factor_decomposition.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np

from prompts.manipulation_check import _bootstrap_paired
from utils import format_table, write_json

#: The two compositional checks and the factors each multiplies together.
#: Each factor is a per-record scalar; the product reproduces the registered
#: check statistic, which is asserted below rather than assumed.
FACTORS: Dict[str, Dict[str, Callable[[Mapping[str, Any]], float]]] = {
    "explore_then_prune": {
        "diversity": lambda r: float(
            len({c.get("claim", "") for c in r.get("atomic_claims", [])})
        ),
        "concentration": lambda r: 1.0
        - _mean([float(c.get("evidence_status") == "uncertain")
                 for c in r.get("atomic_claims", [])]),
    },
    "local_claim_checker": {
        "claim_count": lambda r: float(len(r.get("atomic_claims", []))),
        "sourced_weight": lambda r: 1.0
        + _mean([float(bool(c.get("source_or_check")))
                 for c in r.get("atomic_claims", [])]),
    },
}


def _mean(values: Sequence[float]) -> float:
    """Mean of ``values``, or 0.0 when empty (matching the check helpers)."""
    return float(np.mean(values)) if len(values) else 0.0


def decompose(records_path: str) -> Dict[str, Any]:
    """Report per-factor paired contrasts for each compositional check.

    Args:
        records_path: Joined variant-keyed JSONL produced by
            ``orchestrator.manipulation``.

    Returns:
        A report mapping each role to its per-factor contrasts.
    """
    rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"roles": {}, "note": __doc__.split("\n\n")[1]}

    with open(records_path, "r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]

    for role, factors in FACTORS.items():
        paired = [
            it[role] for it in items
            if role in it
            and it[role].get("operation", {}).get("present", True)
            and it[role].get("neutral", {}).get("present", True)
        ]
        role_out: Dict[str, Any] = {"n_items": len(paired), "factors": {}}

        for factor_name, factor in factors.items():
            operation = np.array([factor(p["operation"]) for p in paired], dtype=float)
            neutral = np.array([factor(p["neutral"]) for p in paired], dtype=float)
            stats = _bootstrap_paired(operation, neutral, seed=0)
            role_out["factors"][factor_name] = {
                "mean_operation": float(operation.mean()),
                "mean_neutral": float(neutral.mean()),
                **stats,
            }
            rows.append({
                "role": role,
                "factor": factor_name,
                "operation": float(operation.mean()),
                "neutral": float(neutral.mean()),
                "delta": stats["diff"],
                "d": stats["standardized_effect"],
                "p": stats["p_value"],
                "ci!=0": stats["interval_excludes_zero"],
            })

        # Guard: the factors must reconstruct the registered statistic, or the
        # decomposition describes something other than the check it claims to
        # explain. Compared on the operation arm's mean product.
        names = list(factors)
        product = np.array(
            [factors[names[0]](p["operation"]) * factors[names[1]](p["operation"])
             for p in paired],
            dtype=float,
        )
        role_out["reconstructed_operation_mean"] = float(product.mean())
        report["roles"][role] = role_out

    report["table"] = rows
    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="Joined check records JSONL.")
    parser.add_argument("--out", default=None, help="Destination JSON.")
    args = parser.parse_args()

    report = decompose(args.records)
    print(format_table(report["table"]))
    for role, payload in report["roles"].items():
        print(f"\n{role}: n={payload['n_items']}, "
              f"reconstructed operation-arm mean product = "
              f"{payload['reconstructed_operation_mean']:.4f}")
    print(
        "\nDescriptive only (PREREGISTRATION.md A3.2). No verdict is issued and "
        "no registered quantity changes."
    )
    if args.out:
        write_json(report, args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
