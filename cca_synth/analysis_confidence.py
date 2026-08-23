"""Verbalised agent confidence against realised correctness.

EXPLORATORY. This analysis was not pre-registered. It is reported as an
observation about the collected records, not as a test of any registered
hypothesis, and the paper labels it as such.

Every agent returns a confidence alongside its answer. Comparing that number to
whether the answer was right separates two properties that a single calibration
figure conflates:

*discrimination*
    Whether confidence orders correct answers above incorrect ones, measured by
    AUROC. No monotone transform of the confidences can change it.

*calibration*
    Whether the stated number matches the realised rate, measured by expected
    calibration error. A monotone rescaling can change this freely.

The distinction is the point. A model whose confidences discriminate well but
calibrate badly is not failing to assess its own output; it is reporting a real
assessment on a compressed scale. This module reports both, and fits an
isotonic regression on half the answers to show how much of the calibration
error a monotone map removes from the other half. That fit is a diagnostic of
scale compression, not a proposed method.

Isotonic regression is implemented here by pool-adjacent-violators rather than
imported, so this module adds no dependency beyond ``requirements.txt``.

Usage::

    python3 analysis_confidence.py --runs-dir runs --out runs/confidence_analysis.json
"""

from __future__ import annotations

import argparse
import collections
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from analysis_cross_arm import ARMS, _load_records
from utils import format_table, write_json

#: Equal-width bins for expected calibration error.
N_ECE_BINS: int = 15

#: Confidence levels listed individually in the report, most frequent first.
N_LEVELS_REPORTED: int = 8


def _pool_adjacent_violators(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a non-decreasing step function to ``y`` over sorted ``x``.

    Args:
        x: Predictor values.
        y: Binary outcomes.

    Returns:
        ``(knots, values)``: the sorted unique ``x`` and the fitted value at
        each, non-decreasing in ``x``.
    """
    order = np.argsort(x, kind="stable")
    x_sorted, y_sorted = x[order], y[order]
    knots = np.unique(x_sorted)
    # Start from the mean outcome at each distinct predictor value.
    values = np.array([y_sorted[x_sorted == k].mean() for k in knots], dtype=np.float64)
    weights = np.array([float((x_sorted == k).sum()) for k in knots], dtype=np.float64)

    # Pool adjacent blocks until the sequence is non-decreasing.
    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue
        total = weights[i] + weights[i + 1]
        values[i] = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total
        weights[i] = total
        values = np.delete(values, i + 1)
        weights = np.delete(weights, i + 1)
        knots_pool = np.delete(knots, i + 1)
        knots = knots_pool
        i = max(i - 1, 0)
    return knots, values


def _apply_isotonic(knots: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a fitted step function at ``x``, clipping outside its range."""
    idx = np.searchsorted(knots, x, side="right") - 1
    return values[np.clip(idx, 0, len(values) - 1)]


def expected_calibration_error(
    probability: np.ndarray, outcome: np.ndarray, n_bins: int = N_ECE_BINS
) -> float:
    """Equal-width binned expected calibration error."""
    if probability.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (probability > low) & (probability <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(
                outcome[in_bin].mean() - probability[in_bin].mean())
    return float(error)


def auroc(outcome: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC; ``nan`` when one class is absent."""
    positives = float(outcome.sum())
    negatives = float(outcome.size - positives)
    if positives == 0.0 or negatives == 0.0:
        return float("nan")
    ranks = np.empty(outcome.size, dtype=np.float64)
    order = np.argsort(score, kind="stable")
    ranks[order] = np.arange(1, outcome.size + 1, dtype=np.float64)
    # Average ranks within ties so the statistic is tie-corrected.
    for value in np.unique(score):
        tied = score == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[outcome == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def _pull(
    records: Mapping[str, Mapping[str, Any]], gold: Mapping[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Confidence and correctness for every scorable, answering agent slot."""
    confidence: List[float] = []
    correct: List[float] = []
    for item_id, record in records.items():
        answer = gold.get(item_id)
        if answer is None:
            continue
        for candidate in record.get("candidates", []):
            if not candidate.get("present", True) or candidate.get("abstain"):
                continue
            option = candidate.get("answer_option")
            stated = candidate.get("confidence")
            if option is None or option < 0 or stated is None:
                continue
            confidence.append(float(stated))
            correct.append(1.0 if option == answer else 0.0)
    return np.asarray(confidence), np.asarray(correct)


def confidence_report(runs_dir: str, seed: int = 0) -> Dict[str, Any]:
    """Summarise verbalised confidence against correctness for every arm.

    Args:
        runs_dir: Directory holding ``arms/``.
        seed: RNG seed for the recalibration split.

    Returns:
        The report.
    """
    records = {arm: _load_records(runs_dir, arm) for arm in ARMS}
    gold: Dict[str, int] = {}
    for item_id, record in records[ARMS[0]].items():
        option = record.get("gold_option")
        if option is not None and option >= 0:
            gold[item_id] = int(option)

    per_arm: Dict[str, Dict[str, float]] = {}
    for arm in ARMS:
        confidence, correct = _pull(records[arm], gold)
        per_arm[arm] = {
            "n_agent_answers": int(confidence.size),
            "mean_stated_confidence": float(confidence.mean()),
            "accuracy": float(correct.mean()),
            "gap": float(confidence.mean() - correct.mean()),
            "sd_stated_confidence": float(confidence.std()),
            "auroc": auroc(correct, confidence),
            "ece": expected_calibration_error(confidence, correct),
            "share_at_or_above_0.95": float((confidence >= 0.95).mean()),
            "accuracy_when_at_or_above_0.95": float(correct[confidence >= 0.95].mean()),
        }

    # The reference arm carries the level table and the recalibration diagnostic.
    reference = ARMS[0]
    confidence, correct = _pull(records[reference], gold)
    levels = [
        {
            "stated_confidence": float(value),
            "n": int(count),
            "share": float(count / confidence.size),
            "accuracy": float(correct[confidence == value].mean()),
        }
        for value, count in collections.Counter(confidence).most_common(N_LEVELS_REPORTED)
    ]

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(confidence.size)
    half = confidence.size // 2
    fit_idx, test_idx = shuffled[:half], shuffled[half:]
    knots, values = _pool_adjacent_violators(confidence[fit_idx], correct[fit_idx])
    mapped = _apply_isotonic(knots, values, confidence[test_idx])
    recalibration = {
        "estimator": "isotonic_regression_pav",
        "fit": "random half, evaluated on the held-out half",
        "seed": seed,
        "ece_before": expected_calibration_error(confidence[test_idx], correct[test_idx]),
        "ece_after": expected_calibration_error(mapped, correct[test_idx]),
        "auroc_before": auroc(correct[test_idx], confidence[test_idx]),
        "auroc_after": auroc(correct[test_idx], mapped),
    }

    return {
        "per_arm": per_arm,
        f"confidence_levels_{reference}": levels,
        f"recalibration_{reference}": recalibration,
        "note": (
            "Verbalised agent confidence against realised correctness over the "
            "scorable items, one row per present, non-abstaining agent answer. "
            "EXPLORATORY: not pre-registered, reported as an observation. The "
            "isotonic fit is a diagnostic of scale compression, not a proposed "
            "method; a monotone map cannot change AUROC, so a large drop in ECE "
            "beside an unchanged AUROC is the signature of a rescaling problem "
            "rather than a knowledge problem."
        ),
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = confidence_report(args.runs_dir, seed=args.seed)
    print(format_table([{"arm": arm, **stats} for arm, stats in report["per_arm"].items()]))
    reference = ARMS[0]
    print(f"\n=== stated confidence levels, {reference} ===")
    print(format_table(report[f"confidence_levels_{reference}"]))
    recalibration = report[f"recalibration_{reference}"]
    print(f"\n=== monotone recalibration, {reference} ===")
    print(f"  ECE   {recalibration['ece_before']:.4f} -> {recalibration['ece_after']:.4f}")
    print(f"  AUROC {recalibration['auroc_before']:.4f} -> {recalibration['auroc_after']:.4f}"
          "   (a monotone map cannot change this)")
    print("\nEXPLORATORY: not pre-registered.")
    if args.out:
        write_json(report, args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
