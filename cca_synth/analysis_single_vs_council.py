"""Single-agent baseline against each council arm.

``PREREGISTRATION.md`` section 4 declares ``v(empty)`` to be the direct-answer
base model on identical per-item input, collected as condition 1. That arm
supplies the one contrast :mod:`analysis_cross_arm` cannot make: the three
council arms against a single call rather than against each other.

Two analyses are reported, and the difference between them is load-bearing.

``primary_ties_count_as_failure``
    Follows :func:`analysis_cross_arm._majority_vote`, whose rule -- fixed
    before collection -- resolves a tied or answerless council to ``-1`` and
    scores it wrong. A council that cannot reach an answer has failed the item,
    and dropping such items would flatter whichever arm abstains most.

``robustness_ties_excluded``
    Drops those items instead. Reported because the rule carries roughly half
    the effect for the two role-prompted arms, and a result that depends on a
    scoring convention should say so rather than rest on it silently.

The baseline collected for this study runs the *same* model as the councils, so
it is call-matched (one call against six) and roughly six times cheaper -- not
the cost-matched contrast the pre-registration specifies. See section 4 of the
paper; this script computes the comparison, it does not relabel it.

Usage::

    python3 analysis_single_vs_council.py --runs-dir runs \
        --out runs/single_vs_council.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from analysis_cross_arm import ARMS, _load_records, _majority_vote
from prompts.manipulation_check import _bootstrap_paired
from utils import benjamini_hochberg, format_table, write_json

#: Condition 1, the single-model direct-answer baseline supplying v(empty).
BASELINE: str = "direct_answer"

#: Resamples, matching every other paired contrast in this repository.
N_BOOTSTRAP: int = 2000


def _load_baseline(runs_dir: str) -> Dict[str, Mapping[str, Any]]:
    """Load the direct-answer arm, keyed by item id."""
    path = os.path.join(runs_dir, "arms", f"records.{BASELINE}.jsonl")
    out: Dict[str, Mapping[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                out[str(record["item_id"])] = record
    return out


def _baseline_answer(record: Mapping[str, Any]) -> int:
    """The baseline's answer, or ``-1`` where it supplies none.

    An abstention or a free-text response carrying no answer index is not a
    wrong answer; it is an item on which v(empty) is undefined. Those items are
    excluded from the contrast entirely rather than scored, which is the same
    treatment :func:`orchestrator.conditions.merge_base_answers` gives them.
    """
    if not record.get("present", True) or record.get("abstain"):
        return -1
    option = record.get("answer_option")
    return -1 if option is None or option < 0 else int(option)


def _contrasts(
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    baseline: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, int],
    items: Sequence[str],
    exclude_ties: bool,
) -> List[Dict[str, Any]]:
    """Contrast every arm against the baseline under one tie convention."""
    rows: List[Tuple[str, float, float, Dict[str, float], int, int]] = []
    p_values: List[float] = []
    for arm in ARMS:
        voted = [(_majority_vote(records[arm][i]), i) for i in items]
        n_no_answer = sum(1 for vote, _ in voted if vote < 0)
        kept = [(vote, i) for vote, i in voted if not (exclude_ties and vote < 0)]
        council = np.array([1.0 if vote == gold[i] else 0.0 for vote, i in kept])
        single = np.array(
            [1.0 if _baseline_answer(baseline[i]) == gold[i] else 0.0 for _, i in kept]
        )
        stats = _bootstrap_paired(council, single, n_bootstrap=N_BOOTSTRAP, seed=0)
        rows.append((arm, float(council.mean()), float(single.mean()),
                     stats, len(kept), n_no_answer))
        p_values.append(float(stats["p_value"]))

    rejected, q_values = benjamini_hochberg(p_values)
    return [
        {
            "contrast": f"{arm} - {BASELINE}",
            "council_accuracy": council_accuracy,
            "direct_answer_accuracy": single_accuracy,
            "delta": float(stats["diff"]),
            "ci_lo": float(stats["lo"]),
            "ci_hi": float(stats["hi"]),
            "d": float(stats["standardized_effect"]),
            "p": float(stats["p_value"]),
            "q": float(q_value),
            "sig_fdr": bool(is_rejected),
            "n_items": n_items,
            "n_council_no_answer": n_no_answer,
        }
        for (arm, council_accuracy, single_accuracy, stats, n_items, n_no_answer),
            q_value, is_rejected in zip(rows, q_values, rejected)
    ]


def single_vs_council(runs_dir: str) -> Dict[str, Any]:
    """Contrast the single-agent baseline with every council arm.

    Args:
        runs_dir: Directory holding ``arms/``.

    Returns:
        The report, carrying both tie conventions.
    """
    records = {arm: _load_records(runs_dir, arm) for arm in ARMS}
    baseline = _load_baseline(runs_dir)

    # Scorable items only. TruthfulQA's generation split carries no answer
    # index, so those items enter no accuracy contrast in any arm.
    gold: Dict[str, int] = {}
    for item_id, record in records[ARMS[0]].items():
        option = record.get("gold_option")
        if option is not None and option >= 0:
            gold[item_id] = int(option)

    usable_baseline = {i for i, r in baseline.items() if _baseline_answer(r) >= 0}
    items = sorted(
        set(gold)
        & usable_baseline
        & set.intersection(*[set(records[arm]) for arm in ARMS])
    )

    single = np.array(
        [1.0 if _baseline_answer(baseline[i]) == gold[i] else 0.0 for i in items]
    )
    return {
        "estimator": "majority_vote",
        "n_bootstrap": N_BOOTSTRAP,
        "n_items_scorable_common": len(items),
        "n_baseline_without_usable_answer": len(set(gold)) - len(set(gold) & usable_baseline),
        "direct_answer_accuracy": float(single.mean()) if single.size else float("nan"),
        "primary_ties_count_as_failure": _contrasts(
            records, baseline, gold, items, exclude_ties=False),
        "robustness_ties_excluded": _contrasts(
            records, baseline, gold, items, exclude_ties=True),
        "note": (
            "Single-agent direct-answer baseline (condition 1, v(empty)) against "
            "each council arm's majority vote, on the scorable items common to "
            "all arms and the baseline. Paired item-level bootstrap, 2,000 "
            "resamples, Benjamini-Hochberg across the three contrasts within "
            "each analysis. PRIMARY follows the pre-committed rule that a tied "
            "or answerless council has failed the item; ROBUSTNESS drops those "
            "items instead. The baseline runs the same model as the councils, "
            "so it is call-matched and cheaper, not cost-matched."
        ),
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = single_vs_council(args.runs_dir)
    print(f"scorable items common to all arms and the baseline: "
          f"{report['n_items_scorable_common']}")
    print(f"direct_answer accuracy: {report['direct_answer_accuracy']:.4f}")
    for key in ("primary_ties_count_as_failure", "robustness_ties_excluded"):
        print(f"\n=== {key} ===")
        print(format_table(report[key]))
    print(
        "\nThe two analyses differ only in whether a tied or answerless council "
        "scores as wrong. Where they disagree, no claim rests on the difference."
    )
    if args.out:
        write_json(report, args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
