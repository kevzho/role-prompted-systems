"""Design-balance verification and split freezing (specification section 6).

``ADVCONSISTENCY_DATACARD.md`` §2.3 promises that this module "verifies the
realised design and refuses to freeze a split whose cells are unbalanced beyond
a declared tolerance". This is that module, and the refusal is real: the freeze
functions raise rather than warn.

Three gates, each guarding a different way the benchmark could be quietly
invalid:

* :func:`balance_report` — the four factors §6 requires to be balanced actually
  are, within tolerance;
* :func:`control_report` — positive controls are present at the declared rate,
  so a flag-everything system cannot score well;
* :func:`freeze_split` — pairs are never split across the train/test boundary,
  every item carries an adjudicated expert label, and the frozen split is
  hashed for publication.

The pair-integrity check in the third is the one most easily overlooked. Two
members of a minimal pair differ in one clause; putting one in development and
its twin in the hidden test set leaks the answer almost completely.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .construct import (
    DISTANCES,
    LENGTH_BUCKETS,
    POSITIONS,
    TARGET_CONTROL_FRACTION,
    TOPICS,
    CandidatePassage,
)

__all__ = [
    "DEFAULT_TOLERANCE",
    "BalanceError",
    "balance_report",
    "control_report",
    "length_report",
    "pair_integrity_report",
    "freeze_split",
]

#: Maximum permitted relative deviation of any cell from its expected share.
#: 0.20 means a cell may hold 20% more or fewer items than an even split would
#: give. Tighter than that is unachievable with a hand-authored corpus; looser
#: lets a factor drift far enough to confound the position-bias measurement.
DEFAULT_TOLERANCE = 0.20


class BalanceError(RuntimeError):
    """Raised when a corpus fails a gate that blocks freezing."""


def _factor_balance(
    values: Sequence[str], levels: Sequence[str], tolerance: float
) -> Dict[str, Any]:
    """Check one factor's level counts against an even split.

    Args:
        values: Observed level per item.
        levels: The factor's declared levels.
        tolerance: Maximum relative deviation.

    Returns:
        Per-level counts, the worst deviation, and a pass flag.
    """
    counts = Counter(values)
    total = len(values)
    expected = total / len(levels) if levels else 0.0

    per_level: Dict[str, Any] = {}
    worst = 0.0
    for level in levels:
        observed = counts.get(level, 0)
        deviation = abs(observed - expected) / expected if expected else float("inf")
        worst = max(worst, deviation)
        per_level[level] = {
            "count": observed,
            "expected": round(expected, 2),
            "relative_deviation": round(deviation, 4),
        }

    unexpected = sorted(set(counts) - set(levels))
    return {
        "levels": per_level,
        "worst_relative_deviation": round(worst, 4),
        "tolerance": tolerance,
        "unexpected_levels": unexpected,
        "passed": worst <= tolerance and not unexpected,
        "n": total,
    }


def balance_report(
    passages: Sequence[CandidatePassage], tolerance: float = DEFAULT_TOLERANCE
) -> Dict[str, Any]:
    """Verify the four factors section 6 requires to be balanced.

    Position and distance are checked over the *contradictory* passages only:
    a consistent passage has no contradiction, so it has no position, and
    including it would dilute the very imbalance the check exists to catch.

    Args:
        passages: The corpus.
        tolerance: Maximum relative deviation per cell.

    Returns:
        Per-factor reports and an overall pass flag.
    """
    with_contradiction = [p for p in passages if p.contains_contradiction]

    factors = {
        "topic": _factor_balance([p.topic for p in passages], TOPICS, tolerance),
        "passage_length_bucket": _factor_balance(
            [p.passage_length_bucket for p in passages],
            tuple(LENGTH_BUCKETS),
            tolerance,
        ),
        "contradiction_position": _factor_balance(
            [p.contradiction_position or "" for p in with_contradiction],
            POSITIONS,
            tolerance,
        ),
        "contradiction_distance": _factor_balance(
            [p.contradiction_distance or "" for p in with_contradiction],
            DISTANCES,
            tolerance,
        ),
    }

    failing = sorted(name for name, report in factors.items() if not report["passed"])
    return {
        "n_passages": len(passages),
        "n_with_contradiction": len(with_contradiction),
        "factors": factors,
        "failing_factors": failing,
        "passed": not failing,
        "note": (
            "Position and distance are computed over contradictory passages "
            "only; a consistent passage has no contradiction to place."
        ),
    }


def control_report(
    passages: Sequence[CandidatePassage], minimum_fraction: float = 0.20
) -> Dict[str, Any]:
    """Verify that positive controls are present at a sufficient rate.

    Args:
        passages: The corpus.
        minimum_fraction: Lowest acceptable control fraction.

    Returns:
        Counts, the realised fraction, and the score a flag-everything system
        would achieve.
    """
    n = len(passages)
    controls = [p for p in passages if p.is_positive_control]
    fraction = len(controls) / n if n else 0.0

    # A system answering "contradiction" to everything is correct on exactly the
    # contradictory passages and wrong on everything else.
    flag_everything = sum(1 for p in passages if p.contains_contradiction) / n if n else 0.0

    return {
        "n_passages": n,
        "n_positive_controls": len(controls),
        "control_fraction": round(fraction, 4),
        "target_fraction": TARGET_CONTROL_FRACTION,
        "minimum_fraction": minimum_fraction,
        "flag_everything_accuracy": round(flag_everything, 4),
        "passed": fraction >= minimum_fraction,
        "note": (
            "Positive controls are passages whose apparent local oddity is "
            "resolved by wider context. Flagging one is a false alarm. Without "
            "enough of them, a flag-everything system scores well and the "
            "primary outcome measures nothing."
        ),
    }


def length_report(passages: Sequence[CandidatePassage]) -> Dict[str, Any]:
    """Check that declared length buckets match actual passage lengths.

    A passage filed under ``long`` that is actually 200 words silently
    invalidates the long-range integration subset, which is where the
    ``local_claim_checker``'s predicted *harm* is supposed to appear.

    Args:
        passages: The corpus.

    Returns:
        Mismatches and a pass flag.
    """
    mismatched = [
        {
            "item_id": p.item_id,
            "declared": p.passage_length_bucket,
            "word_count": p.word_count,
            "expected_range": list(LENGTH_BUCKETS[p.passage_length_bucket]),
        }
        for p in passages
        if not p.length_bucket_matches()
    ]
    return {
        "n_passages": len(passages),
        "n_mismatched": len(mismatched),
        "mismatched": mismatched[:50],
        "passed": not mismatched,
    }


def pair_integrity_report(
    splits: Mapping[str, Sequence[CandidatePassage]]
) -> Dict[str, Any]:
    """Verify that no minimal pair is split across two splits.

    Args:
        splits: Mapping from split name to its passages.

    Returns:
        Any pair whose members landed in different splits, and a pass flag.
    """
    location: Dict[str, List[Tuple[str, str]]] = {}
    for split_name, passages in splits.items():
        for passage in passages:
            location.setdefault(passage.pair_id, []).append(
                (split_name, passage.item_id)
            )

    broken = [
        {"pair_id": pair_id, "members": members}
        for pair_id, members in location.items()
        if len({split for split, _ in members}) > 1
    ]
    return {
        "n_pairs": len(location),
        "n_broken_pairs": len(broken),
        "broken": broken[:50],
        "passed": not broken,
        "note": (
            "Both members of a minimal pair must share a split. They differ in "
            "one clause, so placing one in a public split and its twin in the "
            "hidden test set leaks almost the whole answer."
        ),
    }


def freeze_split(
    splits: Mapping[str, Sequence[CandidatePassage]],
    adjudicated_labels: Mapping[str, bool],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    minimum_control_fraction: float = 0.20,
) -> Dict[str, Any]:
    """Freeze a split, refusing if any section 6 gate fails.

    This is the last point at which an invalid benchmark can be stopped, so it
    raises rather than warning. Every gate it enforces corresponds to a bullet
    in specification section 6 or a row in ``ADVCONSISTENCY_DATACARD.md`` §10.

    Args:
        splits: Mapping from split name to its passages.
        adjudicated_labels: Expert-adjudicated ground truth per ``item_id``.
            Construction-time intent is **not** accepted in its place.
        tolerance: Balance tolerance.
        minimum_control_fraction: Lowest acceptable positive-control rate.

    Returns:
        The frozen manifest, including a per-split hash for publication and the
        count of items where expert labels contradicted construction intent.

    Raises:
        BalanceError: If any gate fails. The message names every failure rather
            than stopping at the first, so one pass fixes everything.
    """
    everything: List[CandidatePassage] = [p for ps in splits.values() for p in ps]
    if not everything:
        raise BalanceError("cannot freeze an empty corpus")

    failures: List[str] = []

    balance = balance_report(everything, tolerance)
    if not balance["passed"]:
        failures.append(
            f"balance: factors {balance['failing_factors']} exceed tolerance "
            f"{tolerance}"
        )

    controls = control_report(everything, minimum_control_fraction)
    if not controls["passed"]:
        failures.append(
            f"positive controls: {controls['control_fraction']:.2%} is below the "
            f"{minimum_control_fraction:.0%} minimum; a flag-everything system "
            f"would score {controls['flag_everything_accuracy']:.2%}"
        )

    lengths = length_report(everything)
    if not lengths["passed"]:
        failures.append(
            f"length buckets: {lengths['n_mismatched']} passages do not match "
            "their declared bucket"
        )

    integrity = pair_integrity_report(splits)
    if not integrity["passed"]:
        failures.append(
            f"pair integrity: {integrity['n_broken_pairs']} minimal pairs are "
            "split across two splits, leaking the answer"
        )

    unlabelled = sorted(p.item_id for p in everything if p.item_id not in adjudicated_labels)
    if unlabelled:
        failures.append(
            f"annotation: {len(unlabelled)} items carry no adjudicated expert "
            f"label (first: {unlabelled[:3]}). Section 6 requires expert label "
            "creation; construction intent is not a label."
        )

    if failures:
        raise BalanceError(
            "refusing to freeze the AdvConsistency split:\n  - "
            + "\n  - ".join(failures)
        )

    # Experts disagreeing with construction intent is not an error -- it is the
    # ambiguity pilot doing its job -- but the rate must be published.
    disagreements = [
        p.item_id
        for p in everything
        if adjudicated_labels[p.item_id] != p.contains_contradiction
    ]

    frozen: Dict[str, Any] = {}
    for split_name, passages in splits.items():
        ids = sorted(p.item_id for p in passages)
        frozen[split_name] = {
            "n_items": len(ids),
            "n_pairs": len({p.pair_id for p in passages}),
            "sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
            "public": split_name != "test",
        }

    return {
        "splits": frozen,
        "n_items": len(everything),
        "balance": balance,
        "positive_controls": controls,
        "pair_integrity": integrity,
        "label_vs_intent_disagreements": len(disagreements),
        "label_vs_intent_disagreement_rate": len(disagreements) / len(everything),
        "disagreeing_item_ids": disagreements[:50],
        "note": (
            "The test split is hidden: publish its sha256 and balance table, "
            "not its items. label_vs_intent disagreements are cases where "
            "experts overruled the constructor -- report the rate, since a high "
            "one means the construction templates produce ambiguous items."
        ),
    }
