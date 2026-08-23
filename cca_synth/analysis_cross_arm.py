"""Paired cross-arm contrasts between the three prompt variants.

``PREREGISTRATION.md`` A2.3 makes the arm contrast itself the reported benchmark
quantity: absolute differences between the full-operation, persona-label-only,
and length-matched neutral councils. Each arm was analysed in its own pipeline
run, so the per-arm reports exist but the *between-arm* comparison does not.
This script supplies it.

Two contrasts are computed, and the distinction matters:

``synthesizer``
    Each arm's trained synthesiser, joined by ``item_id`` from
    ``audit_records.jsonl``. This is the deployed system's accuracy, but each
    arm trained its own synthesiser, so the contrast carries training variance
    in addition to the prompt effect.

``majority_vote``
    Plain majority vote over the six role answers, recomputed identically for
    every arm straight from the collected records. Nothing is trained, so the
    only thing differing between arms is the prompt. This is the cleaner
    estimate of the prompt effect and should be read as the primary one.

Both are paired at the item level over items present in every arm, and
bootstrapped with the same 2,000 resamples used everywhere else in this
repository so the numbers are directly comparable to the per-arm reports.

Usage::

    python3 analysis_cross_arm.py --runs-dir runs --out runs/cross_arm.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from prompts.manipulation_check import _bootstrap_paired
from utils import benjamini_hochberg, format_table, write_json

#: The three prompt variants, in the order the paper reports them.
ARMS: Tuple[str, ...] = ("cca_council", "persona_only_council", "neutral_council")


def _majority_vote(record: Mapping[str, Any]) -> int:
    """Majority answer over a record's present, non-abstaining candidates.

    Ties and empty councils resolve to ``-1`` (no answer), which scores as
    incorrect rather than being dropped: a council that cannot reach an answer
    has failed the item, and silently excluding those items would flatter
    whichever arm abstains most.

    Args:
        record: One item's collected record.

    Returns:
        The winning option index, or ``-1``.
    """
    votes: Dict[int, int] = {}
    for cand in record.get("candidates", []):
        if not cand.get("present", True) or cand.get("abstain"):
            continue
        option = cand.get("answer_option")
        if option is None or option < 0:
            continue
        votes[option] = votes.get(option, 0) + 1
    if not votes:
        return -1
    top = max(votes.values())
    winners = [opt for opt, n in votes.items() if n == top]
    return winners[0] if len(winners) == 1 else -1


def _load_records(runs_dir: str, arm: str) -> Dict[str, Mapping[str, Any]]:
    """Load one arm's collected records, keyed by item id."""
    path = os.path.join(runs_dir, "arms", f"records.{arm}.jsonl")
    out: Dict[str, Mapping[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                out[str(record["item_id"])] = record
    return out


def _load_synth(runs_dir: str, arm: str) -> Dict[str, int]:
    """Load one arm's synthesiser predictions, keyed by item id."""
    path = os.path.join(runs_dir, f"arms-{arm}", "audit_records.jsonl")
    out: Dict[str, int] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            answer = row.get("final_answer_option")
            out[str(row["item_id"])] = -1 if answer is None else int(answer)
    return out


def cross_arm(runs_dir: str) -> Dict[str, Any]:
    """Compute paired contrasts between every pair of arms.

    Args:
        runs_dir: Directory holding ``arms/`` and ``arms-<condition>/``.

    Returns:
        The report.
    """
    records = {arm: _load_records(runs_dir, arm) for arm in ARMS}
    synth = {arm: _load_synth(runs_dir, arm) for arm in ARMS}

    # Only items carrying a usable gold index can enter an accuracy contrast.
    # TruthfulQA's generation split has no answer index at all (gold_option is
    # -1 on every item), so it contributes zero scorable items; counting those
    # as incorrect would understate every arm equally but inflate the apparent
    # sample size, which is worse than reporting the true n.
    def _scorable(item_id: str) -> bool:
        record = records[ARMS[0]][item_id]
        gold = record.get("gold_option")
        return gold is not None and gold >= 0

    common_all = {i for i in set.intersection(*(set(r) for r in records.values()))
                  if _scorable(i)}
    common_synth = {i for i in set.intersection(*(set(s) for s in synth.values()))
                    if i in common_all}

    report: Dict[str, Any] = {
        "n_items_collected_per_arm": {a: len(records[a]) for a in ARMS},
        "n_items_common_all_arms": len(common_all),
        "n_items_common_synthesizer": len(common_synth),
        "estimators": {},
        "note": (
            "Paired item-level bootstrap, 2,000 resamples, matching the per-arm "
            "reports. Items absent from any arm (e.g. refused) are excluded from "
            "every arm so all contrasts use one common item set."
        ),
    }

    for estimator, ids in (("majority_vote", common_all), ("synthesizer", common_synth)):
        if not ids:
            continue
        order = sorted(ids)
        correct: Dict[str, np.ndarray] = {}
        for arm in ARMS:
            if estimator == "majority_vote":
                predicted = [_majority_vote(records[arm][i]) for i in order]
            else:
                predicted = [synth[arm][i] for i in order]
            gold = [records[arm][i].get("gold_option") for i in order]
            correct[arm] = np.array(
                [float(p == g and p is not None and p >= 0) for p, g in zip(predicted, gold)]
            )

        rows: List[Dict[str, Any]] = []
        pairs: List[Tuple[str, str, Dict[str, float]]] = []
        for i, left in enumerate(ARMS):
            for right in ARMS[i + 1:]:
                stats = _bootstrap_paired(correct[left], correct[right], seed=0)
                pairs.append((left, right, stats))

        rejected, q_values = benjamini_hochberg(
            [p[2]["p_value"] for p in pairs], alpha=0.05
        )
        for (left, right, stats), is_rejected, q_value in zip(pairs, rejected, q_values):
            rows.append({
                "contrast": f"{left} - {right}",
                "delta": stats["diff"],
                "ci_lo": stats["lo"],
                "ci_hi": stats["hi"],
                "d": stats["standardized_effect"],
                "p": stats["p_value"],
                "q": float(q_value),
                "sig_fdr": bool(is_rejected),
            })

        report["estimators"][estimator] = {
            "n_items": len(order),
            "accuracy": {a: float(correct[a].mean()) for a in ARMS},
            "contrasts": rows,
        }

    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = cross_arm(args.runs_dir)
    print(f"items per arm: {report['n_items_collected_per_arm']}")
    print(f"common to all arms: {report['n_items_common_all_arms']}")
    for estimator, payload in report["estimators"].items():
        print(f"\n=== {estimator} (n={payload['n_items']}) ===")
        print("  accuracy: " + "  ".join(
            f"{a}={v:.4f}" for a, v in payload["accuracy"].items()))
        print(format_table(payload["contrasts"]))
    print(
        "\nBenjamini-Hochberg applied across the three pairwise contrasts within "
        "each estimator. Reported descriptively alongside uncorrected p."
    )
    if args.out:
        write_json(report, args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
