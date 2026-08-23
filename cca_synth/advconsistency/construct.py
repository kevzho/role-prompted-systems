"""Paired-passage construction for AdvConsistency (specification section 6).

The construction protocol, in the order section 6 requires:

1. The phenomenon is defined independently of any role -- the definition lives in
   ``benchmarks.advconsistency.PHENOMENON_DEFINITION`` and is imported here
   rather than restated, so the two cannot drift.
2. Every item belongs to a **minimal pair**: two passages differing in exactly
   one clause, one consistent and one carrying a seeded contradiction. The pair
   is the unit of analysis, which removes passage difficulty as a confound.
3. **Positive controls** are passages where an apparent local oddity is
   legitimately resolved by wider context. Flagging one is a false alarm. Without
   them a flag-everything system scores perfectly.
4. Four factors are balanced **by construction**: contradiction position, topic,
   passage length, and contradiction distance.

What this module does *not* do is decide whether a constructed item is any good.
That requires domain experts (§6, and ``ADVCONSISTENCY_DATACARD.md`` §3), and
the pipeline is deliberately arranged so that unannotated items cannot reach a
frozen split: :func:`freeze_split` refuses without adjudicated labels.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "POSITIONS",
    "LENGTH_BUCKETS",
    "DISTANCES",
    "TOPICS",
    "TARGET_CONTROL_FRACTION",
    "CandidatePassage",
    "PassagePair",
    "make_pair",
    "assign_cells",
    "write_candidates",
    "read_candidates",
]

#: Where in the passage the contradicting clause sits.
POSITIONS: Tuple[str, ...] = ("early", "middle", "late")

#: Passage length buckets, in words. Long passages are where the
#: ``local_claim_checker``'s predicted *harm* (degraded long-range integration)
#: should show up, so the long bucket is not optional.
LENGTH_BUCKETS: Mapping[str, Tuple[int, int]] = {
    "short": (150, 250),
    "medium": (250, 450),
    "long": (450, 800),
}

#: How far the contradicting fact sits from the fact it contradicts. This is the
#: axis that separates local checking from global integration.
DISTANCES: Tuple[str, ...] = ("adjacent", "same_paragraph", "cross_paragraph")

#: Six topics, balanced equally, so a topic-specific effect cannot masquerade as
#: a general one.
TOPICS: Tuple[str, ...] = (
    "natural_science",
    "history",
    "geography",
    "technology",
    "economics",
    "biography",
)

#: Fraction of items that are positive controls. At 25%, a flag-everything
#: strategy scores at most 0.75 -- below any threshold worth reporting.
TARGET_CONTROL_FRACTION = 0.25


@dataclass
class CandidatePassage:
    """One passage awaiting expert annotation.

    Attributes:
        item_id: Stable identifier.
        pair_id: Shared by both members of a minimal pair.
        passage: The passage text.
        contains_contradiction: Construction-time intent. **Not** a label --
            :func:`freeze_split` uses adjudicated expert labels, and a mismatch
            between intent and label is itself reportable.
        is_positive_control: Apparent local oddity resolved by wider context.
        contradiction_position: One of :data:`POSITIONS`, or ``None``.
        contradiction_distance: One of :data:`DISTANCES`, or ``None``.
        passage_length_bucket: One of :data:`LENGTH_BUCKETS`.
        topic: One of :data:`TOPICS`.
        requires_long_range: Whether resolving it needs cross-paragraph
            integration.
        source_licence: Licence of any adapted source material.
        altered_clause: The clause that differs between pair members, recorded
            so an adjudicator can check that exactly one thing changed.
    """

    item_id: str
    pair_id: str
    passage: str
    contains_contradiction: bool
    is_positive_control: bool = False
    contradiction_position: Optional[str] = None
    contradiction_distance: Optional[str] = None
    passage_length_bucket: str = "short"
    topic: str = "natural_science"
    requires_long_range: bool = False
    source_licence: str = "CC-BY-4.0"
    altered_clause: str = ""

    def __post_init__(self) -> None:
        if self.contradiction_position is not None and self.contradiction_position not in POSITIONS:
            raise ValueError(
                f"contradiction_position must be one of {POSITIONS}; "
                f"got {self.contradiction_position!r}"
            )
        if self.contradiction_distance is not None and self.contradiction_distance not in DISTANCES:
            raise ValueError(
                f"contradiction_distance must be one of {DISTANCES}; "
                f"got {self.contradiction_distance!r}"
            )
        if self.passage_length_bucket not in LENGTH_BUCKETS:
            raise ValueError(
                f"passage_length_bucket must be one of {tuple(LENGTH_BUCKETS)}; "
                f"got {self.passage_length_bucket!r}"
            )
        if self.topic not in TOPICS:
            raise ValueError(f"topic must be one of {TOPICS}; got {self.topic!r}")
        if self.contains_contradiction and self.contradiction_position is None:
            raise ValueError(
                f"{self.item_id}: a passage carrying a contradiction must record "
                "where it sits; position balance cannot be verified otherwise"
            )
        if self.is_positive_control and self.contains_contradiction:
            raise ValueError(
                f"{self.item_id}: a positive control is a passage with NO real "
                "contradiction whose local oddity is resolved by context. "
                "Marking one as containing a contradiction defeats its purpose."
            )

    @property
    def word_count(self) -> int:
        """Length of the passage in whitespace-separated tokens.

        Returns:
            Word count.
        """
        return len(self.passage.split())

    def length_bucket_matches(self) -> bool:
        """Whether the passage's actual length matches its declared bucket.

        Returns:
            ``True`` when the word count falls inside the declared range.
        """
        low, high = LENGTH_BUCKETS[self.passage_length_bucket]
        return low <= self.word_count <= high


@dataclass
class PassagePair:
    """A minimal pair: two passages differing in exactly one clause.

    Attributes:
        pair_id: Shared identifier.
        consistent: The member with no seeded contradiction.
        contradictory: The member carrying it.
    """

    pair_id: str
    consistent: CandidatePassage
    contradictory: CandidatePassage

    def __post_init__(self) -> None:
        if self.consistent.pair_id != self.pair_id or self.contradictory.pair_id != self.pair_id:
            raise ValueError(f"pair {self.pair_id}: members carry mismatched pair_ids")
        if self.consistent.contains_contradiction:
            raise ValueError(f"pair {self.pair_id}: the consistent member claims a contradiction")
        if not self.contradictory.contains_contradiction:
            raise ValueError(f"pair {self.pair_id}: the contradictory member claims none")
        for factor in ("topic", "passage_length_bucket"):
            a, b = getattr(self.consistent, factor), getattr(self.contradictory, factor)
            if a != b:
                raise ValueError(
                    f"pair {self.pair_id}: members differ on {factor} ({a} vs {b}); "
                    "a minimal pair must differ in exactly one clause"
                )

    def as_list(self) -> List[CandidatePassage]:
        """Both members.

        Returns:
            ``[consistent, contradictory]``.
        """
        return [self.consistent, self.contradictory]


def _pair_id(seed: str) -> str:
    """Derive a stable pair identifier.

    Args:
        seed: Any stable string.

    Returns:
        A 12-character hex identifier.
    """
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def make_pair(
    base_passage: str,
    original_clause: str,
    contradicting_clause: str,
    *,
    topic: str,
    length_bucket: str,
    position: str,
    distance: str,
    seed: str,
    source_licence: str = "CC-BY-4.0",
) -> PassagePair:
    """Build a minimal pair by substituting exactly one clause.

    Substitution rather than independent authoring is the point: it is what makes
    the pair *minimal*. Writing two passages by hand would let incidental
    differences in style, length, or difficulty confound the contrast.

    Args:
        base_passage: Passage text containing ``original_clause`` verbatim.
        original_clause: The clause present in the consistent member.
        contradicting_clause: The clause that replaces it, contradicting a fact
            stated elsewhere in the passage.
        topic: One of :data:`TOPICS`.
        length_bucket: One of :data:`LENGTH_BUCKETS`.
        position: One of :data:`POSITIONS`.
        distance: One of :data:`DISTANCES`.
        seed: Stable string used to derive the pair id.
        source_licence: Licence of adapted source material.

    Returns:
        The constructed pair.

    Raises:
        ValueError: If ``original_clause`` does not appear exactly once in
            ``base_passage``, or if the substitution changes nothing.
    """
    occurrences = base_passage.count(original_clause)
    if occurrences != 1:
        raise ValueError(
            f"original_clause must appear exactly once in the passage; found "
            f"{occurrences}. A substitution that hits zero or several sites is "
            "not a minimal pair."
        )
    if original_clause == contradicting_clause:
        raise ValueError("contradicting_clause is identical to original_clause")

    pair = _pair_id(seed)
    altered = base_passage.replace(original_clause, contradicting_clause)

    common = dict(
        pair_id=pair,
        passage_length_bucket=length_bucket,
        topic=topic,
        source_licence=source_licence,
        requires_long_range=distance == "cross_paragraph",
    )
    return PassagePair(
        pair_id=pair,
        consistent=CandidatePassage(
            item_id=f"advc-{pair}-c",
            passage=base_passage,
            contains_contradiction=False,
            altered_clause=original_clause,
            **common,
        ),
        contradictory=CandidatePassage(
            item_id=f"advc-{pair}-x",
            passage=altered,
            contains_contradiction=True,
            contradiction_position=position,
            contradiction_distance=distance,
            altered_clause=contradicting_clause,
            **common,
        ),
    )


def assign_cells(n_pairs: int) -> List[Dict[str, str]]:
    """Produce a balanced assignment of construction cells.

    Rather than constructing items and checking balance afterwards -- which
    yields an unbalanceable corpus and a lot of wasted expert time -- the design
    is generated first and passages are written to fill it.

    The cell space is ``TOPICS x LENGTH_BUCKETS x POSITIONS x DISTANCES``
    (6 x 3 x 3 x 3 = 162). Cells are dealt round-robin so any prefix of the
    returned list is close to balanced, which matters because construction is
    usually interrupted before the target is reached.

    Args:
        n_pairs: Number of pairs to plan.

    Returns:
        One cell specification per pair.

    Raises:
        ValueError: If ``n_pairs`` is not positive.
    """
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be positive; got {n_pairs}")

    cells: List[Dict[str, str]] = []
    for topic in TOPICS:
        for bucket in LENGTH_BUCKETS:
            for position in POSITIONS:
                for distance in DISTANCES:
                    cells.append(
                        {
                            "topic": topic,
                            "length_bucket": bucket,
                            "position": position,
                            "distance": distance,
                        }
                    )

    # Emitting the nested loops in their natural order would exhaust topic 0's
    # 27 cells before touching topic 1, so a run stopped at 60 pairs would cover
    # two topics and a third of a third. Instead, order greedily: repeatedly take
    # whichever remaining cell uses the currently least-used levels. That makes
    # *every* prefix close to balanced on all four factors simultaneously, which
    # is what matters -- construction is routinely interrupted before target.
    factors = ("topic", "length_bucket", "position", "distance")
    used: Dict[str, Counter] = {f: Counter() for f in factors}
    remaining = list(cells)
    ordered: List[Dict[str, str]] = []

    while remaining:
        best_index = min(
            range(len(remaining)),
            # Sum of current usage across the cell's four levels; ties break on
            # the cell's position, keeping the ordering deterministic.
            key=lambda i: (
                sum(used[f][remaining[i][f]] for f in factors),
                i,
            ),
        )
        chosen = remaining.pop(best_index)
        ordered.append(chosen)
        for factor in factors:
            used[factor][chosen[factor]] += 1

    plan: List[Dict[str, str]] = []
    while len(plan) < n_pairs:
        plan.append(dict(ordered[len(plan) % len(ordered)]))
    return plan


def write_candidates(passages: Sequence[CandidatePassage], path: str) -> Dict[str, Any]:
    """Write candidate passages to JSONL.

    Args:
        passages: Passages to write.
        path: Destination path.

    Returns:
        A summary with counts and the content hash.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    digest = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as handle:
        for passage in passages:
            line = json.dumps(asdict(passage), sort_keys=True)
            handle.write(line + "\n")
            digest.update(line.encode("utf-8"))

    n_controls = sum(1 for p in passages if p.is_positive_control)
    return {
        "path": path,
        "n_passages": len(passages),
        "n_pairs": len({p.pair_id for p in passages}),
        "n_positive_controls": n_controls,
        "control_fraction": n_controls / len(passages) if passages else 0.0,
        "sha256": digest.hexdigest(),
    }


def read_candidates(path: str) -> List[CandidatePassage]:
    """Read candidate passages from JSONL.

    Args:
        path: Source path.

    Returns:
        The passages.
    """
    out: List[CandidatePassage] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(CandidatePassage(**json.loads(line)))
    return out
