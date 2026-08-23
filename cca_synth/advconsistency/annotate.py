"""Expert annotation packets, agreement, and adjudication (section 6).

``ADVCONSISTENCY_DATACARD.md`` §3 specifies the annotation process; this module
implements the mechanics so the human part is a matter of reading passages
rather than managing spreadsheets.

Three properties are enforced rather than requested:

* **Pair partners are never shown to the same annotator.** Two passages
  differing in one clause make the task trivial when seen together, and the
  resulting agreement statistic would be meaningless.
* **Controls are not disclosed.** An annotator told which passages are positive
  controls will not treat them as ordinary items.
* **"Ambiguous" is a first-class response.** Forcing a binary judgement
  manufactures agreement; the ambiguity pilot exists to *remove* such items, so
  it has to be able to see them.

Agreement is Krippendorff's alpha, reused from ``benchmarks.metrics`` rather
than reimplemented, because two implementations of the same statistic drift.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .construct import CandidatePassage

__all__ = [
    "AMBIGUOUS",
    "AnnotationTask",
    "build_annotation_packets",
    "annotator_agreement",
    "adjudicate",
    "ambiguity_pilot_report",
]

#: The response an annotator gives when the item cannot be judged. Not a
#: failure state -- see the module docstring.
AMBIGUOUS = "ambiguous"


@dataclass
class AnnotationTask:
    """One passage as an annotator sees it.

    Carries no pair partner, no control flag, and no construction intent.

    Attributes:
        task_id: Opaque identifier; not derivable from the item id.
        item_id: Retained for the experimenter's join, never shown.
        passage: The passage text.
        instructions: The phenomenon definition and response options.
    """

    task_id: str
    item_id: str
    passage: str
    instructions: str

    def for_annotator(self) -> Dict[str, str]:
        """The view handed to an annotator.

        Returns:
            A mapping with the passage and instructions only. ``item_id`` is
            deliberately absent: ids encode pair membership, and an annotator
            who notices ``-c``/``-x`` suffixes has been unblinded.
        """
        return {
            "task_id": self.task_id,
            "passage": self.passage,
            "instructions": self.instructions,
        }


def _instructions() -> str:
    """Assemble annotator instructions from the canonical definition.

    Returns:
        The instruction text, including the phenomenon definition verbatim.
    """
    from benchmarks.advconsistency import PHENOMENON_DEFINITION

    return (
        "Read the passage in full, then decide whether it contains an internal "
        "contradiction.\n\n"
        f"DEFINITION\n{PHENOMENON_DEFINITION}\n\n"
        "RESPOND WITH ONE OF:\n"
        "  yes        - the passage contradicts itself; quote the clause\n"
        "  no         - the passage is internally consistent\n"
        "  ambiguous  - you genuinely cannot decide\n\n"
        "'ambiguous' is a real answer, not a failure. Items marked ambiguous "
        "are removed from the benchmark, which is what we want. Guessing "
        "produces agreement that does not exist.\n\n"
        "EXCLUDED - answer 'no' for these, they are not what we are measuring:\n"
        "  - contradictions needing outside knowledge to detect\n"
        "  - ambiguous pronoun reference\n"
        "  - two statements in tension where both could be true\n"
        "  - numeric inconsistencies needing more than one arithmetic step\n"
    )


def build_annotation_packets(
    passages: Sequence[CandidatePassage],
    annotators: Sequence[str],
    *,
    ratings_per_item: int = 3,
    seed: int = 20260728,
) -> Tuple[Dict[str, List[AnnotationTask]], Dict[str, str]]:
    """Allocate passages to annotators without ever pairing partners.

    Args:
        passages: The corpus.
        annotators: Annotator identifiers.
        ratings_per_item: Independent ratings each item must receive.
        seed: Allocation seed, recorded for reproducibility.

    Returns:
        Tuple ``(packets, task_to_item)``. ``packets`` maps annotator to their
        tasks; ``task_to_item`` is the experimenter's join key and must not be
        given to annotators.

    Raises:
        ValueError: If there are too few annotators to give every item the
            required number of *independent* ratings without showing one person
            both members of a pair.
    """
    if ratings_per_item < 2:
        raise ValueError(
            f"ratings_per_item must be at least 2; got {ratings_per_item}. "
            "Limitation L16 requires at least two independent ratings."
        )
    if len(annotators) < ratings_per_item:
        raise ValueError(
            f"{len(annotators)} annotators cannot supply {ratings_per_item} "
            "independent ratings per item"
        )

    instructions = _instructions()
    rng = random.Random(seed)

    packets: Dict[str, List[AnnotationTask]] = {a: [] for a in annotators}
    task_to_item: Dict[str, str] = {}
    # Which pair_ids each annotator has already seen, in either polarity.
    seen_pairs: Dict[str, set] = {a: set() for a in annotators}

    for passage in sorted(passages, key=lambda p: p.item_id):
        eligible = [a for a in annotators if passage.pair_id not in seen_pairs[a]]
        if len(eligible) < ratings_per_item:
            raise ValueError(
                f"item {passage.item_id}: only {len(eligible)} annotators have "
                f"not already seen pair {passage.pair_id}, but "
                f"{ratings_per_item} independent ratings are required. Recruit "
                "more annotators, or lower ratings_per_item."
            )
        # Prefer the least-loaded eligible annotators, breaking ties randomly so
        # allocation does not correlate with item order.
        rng.shuffle(eligible)
        eligible.sort(key=lambda a: len(packets[a]))
        chosen = eligible[:ratings_per_item]

        for annotator in chosen:
            task_id = hashlib.sha256(
                f"{seed}:{annotator}:{passage.item_id}".encode("utf-8")
            ).hexdigest()[:16]
            packets[annotator].append(
                AnnotationTask(
                    task_id=task_id,
                    item_id=passage.item_id,
                    passage=passage.passage,
                    instructions=instructions,
                )
            )
            task_to_item[task_id] = passage.item_id
            seen_pairs[annotator].add(passage.pair_id)

    # Shuffle each packet so a single annotator does not see items in id order,
    # which would group topics and lengths together.
    for annotator in packets:
        rng.shuffle(packets[annotator])

    return packets, task_to_item


def annotator_agreement(
    responses: Mapping[str, Mapping[str, str]]
) -> Dict[str, Any]:
    """Inter-annotator agreement over the non-ambiguous responses.

    Ambiguous responses are excluded from alpha and counted separately: they are
    a statement that the *item* is bad, not that the annotator disagreed.

    Args:
        responses: Mapping from annotator to ``{item_id: "yes"|"no"|"ambiguous"}``.

    Returns:
        Krippendorff's alpha, the ambiguity rate, and per-item disagreement.
    """
    from benchmarks.metrics import krippendorff_alpha

    item_ids = sorted({i for per_annotator in responses.values() for i in per_annotator})
    annotators = sorted(responses)

    matrix: List[List[Optional[float]]] = []
    ambiguous_counts: Dict[str, int] = {}
    disputed: List[str] = []

    for item_id in item_ids:
        row: List[Optional[float]] = []
        seen: List[str] = []
        for annotator in annotators:
            value = responses[annotator].get(item_id)
            if value is None:
                row.append(None)
                continue
            if value == AMBIGUOUS:
                ambiguous_counts[item_id] = ambiguous_counts.get(item_id, 0) + 1
                row.append(None)
                continue
            seen.append(value)
            row.append(1.0 if value == "yes" else 0.0)
        matrix.append(row)
        if len(set(seen)) > 1:
            disputed.append(item_id)

    n_items = len(item_ids)
    return {
        "n_items": n_items,
        "n_annotators": len(annotators),
        "krippendorff_alpha": krippendorff_alpha(matrix),
        "n_items_disputed": len(disputed),
        "dispute_rate": len(disputed) / n_items if n_items else float("nan"),
        "n_items_marked_ambiguous": len(ambiguous_counts),
        "ambiguity_rate": len(ambiguous_counts) / n_items if n_items else float("nan"),
        "disputed_item_ids": disputed[:100],
        "ambiguous_item_ids": sorted(ambiguous_counts)[:100],
        "note": (
            "Ambiguous responses are excluded from alpha rather than coerced to "
            "a class. They report that the item is unusable, which is what the "
            "ambiguity pilot is for."
        ),
    }


def adjudicate(
    responses: Mapping[str, Mapping[str, str]],
    adjudicator_rulings: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Resolve each item to a label, a discard, or a referral.

    Unanimous non-ambiguous items resolve immediately. Everything else needs an
    adjudicator ruling; where one is absent the item is *referred*, never
    guessed.

    Args:
        responses: Mapping from annotator to ``{item_id: response}``.
        adjudicator_rulings: Optional ``{item_id: "yes"|"no"|"discard"}``.

    Returns:
        Resolved labels, discards, referrals, and counts.
    """
    rulings = dict(adjudicator_rulings or {})
    item_ids = sorted({i for per_annotator in responses.values() for i in per_annotator})

    labels: Dict[str, bool] = {}
    discarded: List[str] = []
    referred: List[str] = []
    resolved_by_adjudicator = 0

    for item_id in item_ids:
        votes = [
            per_annotator[item_id]
            for per_annotator in responses.values()
            if item_id in per_annotator
        ]
        decisive = [v for v in votes if v != AMBIGUOUS]

        if decisive and len(set(decisive)) == 1 and len(decisive) == len(votes):
            labels[item_id] = decisive[0] == "yes"
            continue

        ruling = rulings.get(item_id)
        if ruling is None:
            referred.append(item_id)
        elif ruling == "discard":
            discarded.append(item_id)
            resolved_by_adjudicator += 1
        else:
            labels[item_id] = ruling == "yes"
            resolved_by_adjudicator += 1

    n = len(item_ids)
    return {
        "labels": labels,
        "n_items": n,
        "n_resolved": len(labels),
        "n_resolved_unanimously": len(labels) - resolved_by_adjudicator,
        "n_resolved_by_adjudicator": resolved_by_adjudicator,
        "n_discarded": len(discarded),
        "n_referred": len(referred),
        "discarded_item_ids": discarded,
        "referred_item_ids": referred[:100],
        "complete": not referred,
        "note": (
            "Items are referred, never guessed. A corpus with referrals "
            "outstanding cannot be frozen: balance.freeze_split requires an "
            "adjudicated label for every item."
        ),
    }


def ambiguity_pilot_report(
    agreement: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    *,
    alpha_target: float = 0.80,
    max_discard_rate: float = 0.20,
) -> Dict[str, Any]:
    """Decide whether the construction templates are good enough to scale.

    ``ADVCONSISTENCY_DATACARD.md`` §4: if the pilot discard rate exceeds 20%, the
    templates are producing ambiguous items and the phenomenon definition needs
    narrowing *before* scaling. Scaling first wastes expensive expert time.

    Args:
        agreement: Output of :func:`annotator_agreement`.
        adjudication: Output of :func:`adjudicate`.
        alpha_target: Minimum acceptable Krippendorff's alpha.
        max_discard_rate: Maximum acceptable discard rate.

    Returns:
        A go/no-go verdict with the reason stated.
    """
    n = adjudication.get("n_items", 0) or 1
    discard_rate = adjudication.get("n_discarded", 0) / n
    alpha = agreement.get("krippendorff_alpha", float("nan"))

    alpha_ok = alpha == alpha and alpha >= alpha_target
    discard_ok = discard_rate <= max_discard_rate

    if not alpha_ok:
        verdict = (
            f"NO-GO: Krippendorff's alpha {alpha:.3f} is below the {alpha_target} "
            "target. Labels are uncertain at roughly the size of the effect the "
            "benchmark is meant to measure. Revise the construction templates; "
            "do not relabel the existing items."
        )
    elif not discard_ok:
        verdict = (
            f"NO-GO: discard rate {discard_rate:.1%} exceeds {max_discard_rate:.0%}. "
            "The templates are producing ambiguous items. Narrow the phenomenon "
            "definition before scaling."
        )
    else:
        verdict = (
            f"GO: alpha {alpha:.3f} clears {alpha_target}, discard rate "
            f"{discard_rate:.1%} is within {max_discard_rate:.0%}. Scale "
            "construction and freeze the hidden split."
        )

    return {
        "krippendorff_alpha": alpha,
        "alpha_target": alpha_target,
        "alpha_passed": alpha_ok,
        "discard_rate": round(discard_rate, 4),
        "max_discard_rate": max_discard_rate,
        "discard_rate_passed": discard_ok,
        "ambiguity_rate": agreement.get("ambiguity_rate"),
        "passed": alpha_ok and discard_ok,
        "verdict": verdict,
    }
