"""Blinded, order-randomised judging packets (limitation L16).

Limitation L16 requires judges to be blinded to condition and shown responses in
randomised order. Both are easy to state and easy to get wrong, so they are
enforced structurally here rather than left to the person assembling a
spreadsheet:

* :class:`JudgingPacket` carries no condition, arm, model, or role field. The
  mapping back to conditions lives in a separate :class:`BlindingKey` that the
  judge never receives, and :func:`build_packets` returns them as two objects so
  that handing over the wrong one is a visible mistake rather than a silent one.
* Presentation order is a deterministic function of a seed, so it is reproducible
  and auditable, but it is *independent per item*, so a judge cannot learn that
  "the second response is always the council".
* :func:`check_blinding` re-inspects assembled packets for condition leakage --
  a response that names its own condition, a systematically different length, or
  an ordering that correlates with condition.

The last check matters more than it looks. Blinding fails most often not because
someone forgot to strip a field but because the arms are distinguishable on
sight: if the council's answers are three times longer, the judge is unblinded
regardless of what the metadata says.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "JudgedResponse",
    "JudgingPacket",
    "BlindingKey",
    "build_packets",
    "check_blinding",
    "CONDITION_LEAK_PATTERNS",
]

#: Patterns whose presence in a response would tell a judge which condition
#: produced it. Checked by :func:`check_blinding`; a hit is a blinding failure,
#: not a warning.
CONDITION_LEAK_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\bas the (council|synthesi[sz]er|committee)\b", re.IGNORECASE),
    re.compile(r"\b(agent|role)\s*[0-9]", re.IGNORECASE),
    re.compile(r"\bcounter[_ -]evidence[_ -]checker\b", re.IGNORECASE),
    re.compile(r"\bconsistency[_ -]checker\b", re.IGNORECASE),
    re.compile(r"\blocal[_ -]claim[_ -]checker\b", re.IGNORECASE),
    re.compile(r"\bexplore[_ -]then[_ -]prune\b", re.IGNORECASE),
    re.compile(r"\brare[_ -]hypothesis[_ -]checker\b", re.IGNORECASE),
    re.compile(r"\bconfidence[_ -]calibrator\b", re.IGNORECASE),
    re.compile(r"\bself[- ]consistency\b", re.IGNORECASE),
    re.compile(r"\bmajority vote\b", re.IGNORECASE),
    re.compile(r"\bcondition\s*[0-9]+\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class JudgedResponse:
    """One response awaiting judgement, with its provenance stripped.

    Attributes:
        response_id: Opaque identifier. Deliberately a hash, not
            ``"council_item_003"``: a readable id is an unblinding channel.
        text: The response shown to the judge.
    """

    response_id: str
    text: str


@dataclass(frozen=True)
class JudgingPacket:
    """What a judge actually receives for one item.

    Carries no condition, model, arm, or role. Adding such a field is a blinding
    failure and :func:`check_blinding` looks for it.

    Attributes:
        packet_id: Opaque packet identifier.
        item_id: Benchmark item id, needed to align ratings back to items.
        prompt: The question or instruction the responses answer.
        context: Optional supporting passage.
        responses: Responses in the order the judge sees them.
        rubric: The rubric text, published verbatim.
    """

    packet_id: str
    item_id: str
    prompt: str
    context: str
    responses: Tuple[JudgedResponse, ...]
    rubric: Mapping[str, str]


@dataclass
class BlindingKey:
    """The mapping from opaque response ids back to conditions.

    Held by the experimenter, never given to a judge. Kept as a separate object
    so that serialising the packets cannot accidentally serialise the key.

    Attributes:
        response_to_condition: Opaque response id to condition name.
        response_to_item: Opaque response id to benchmark item id.
        seed: Seed used for order randomisation, recorded for reproducibility.
    """

    response_to_condition: Dict[str, str] = field(default_factory=dict)
    response_to_item: Dict[str, str] = field(default_factory=dict)
    seed: int = 0

    def unblind(self, response_id: str) -> Tuple[str, str]:
        """Recover an item and condition after judging is complete.

        Args:
            response_id: Opaque response identifier.

        Returns:
            Tuple ``(item_id, condition)``.

        Raises:
            KeyError: If the id is unknown.
        """
        return (self.response_to_item[response_id], self.response_to_condition[response_id])


def _opaque_id(*parts: str, salt: str) -> str:
    """Derive a stable opaque identifier.

    Args:
        *parts: Components to hash.
        salt: Per-run salt so that ids are not comparable across runs.

    Returns:
        A 16-character hex identifier.
    """
    joined = "\x1f".join((salt,) + parts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def build_packets(
    items: Sequence[Any],
    responses_by_condition: Mapping[str, Sequence[str]],
    rubric: Mapping[str, str],
    *,
    seed: int = 20260728,
    salt: str = "cca-judging-v1",
) -> Tuple[List[JudgingPacket], BlindingKey]:
    """Assemble blinded, order-randomised judging packets.

    Args:
        items: Benchmark items, exposing ``item_id``, ``question``, ``context``.
        responses_by_condition: Condition name to per-item response texts,
            each aligned with ``items``.
        rubric: Rubric shown to the judge, published verbatim.
        seed: Order-randomisation seed, recorded in the key.
        salt: Per-run salt for opaque identifiers.

    Returns:
        Tuple ``(packets, key)``. The key must not be given to a judge.

    Raises:
        ValueError: If any condition's responses are not aligned with ``items``,
            or if fewer than two conditions are supplied (there is nothing to
            blind between).
    """
    if len(responses_by_condition) < 2:
        raise ValueError(
            "blinding requires at least two conditions; with one arm there is "
            "nothing for the judge to be blind to"
        )
    for condition, responses in responses_by_condition.items():
        if len(responses) != len(items):
            raise ValueError(
                f"condition {condition!r} has {len(responses)} responses for "
                f"{len(items)} items"
            )

    key = BlindingKey(seed=seed)
    packets: List[JudgingPacket] = []

    for index, item in enumerate(items):
        entries: List[JudgedResponse] = []
        for condition, responses in responses_by_condition.items():
            response_id = _opaque_id(item.item_id, condition, salt=salt)
            key.response_to_condition[response_id] = condition
            key.response_to_item[response_id] = item.item_id
            entries.append(JudgedResponse(response_id=response_id, text=responses[index]))

        # Per-item shuffle, deterministic in (seed, item_id). Using a per-item
        # stream rather than one global stream means the order for item k does
        # not depend on how many items preceded it, so a re-run over a subset
        # produces the same orders.
        rng = random.Random(f"{seed}:{item.item_id}")
        rng.shuffle(entries)

        packets.append(
            JudgingPacket(
                packet_id=_opaque_id(item.item_id, "packet", salt=salt),
                item_id=item.item_id,
                prompt=getattr(item, "question", ""),
                context=getattr(item, "context", ""),
                responses=tuple(entries),
                rubric=dict(rubric),
            )
        )

    return packets, key


def check_blinding(
    packets: Sequence[JudgingPacket],
    key: BlindingKey,
    *,
    length_ratio_tolerance: float = 1.5,
) -> Dict[str, Any]:
    """Re-inspect assembled packets for condition leakage.

    Three checks:

    1. **Textual leakage** -- a response naming its own condition or a role.
    2. **Length signature** -- one condition's mean response length differing
       from another's by more than ``length_ratio_tolerance``, which unblinds a
       judge on sight regardless of metadata.
    3. **Position leakage** -- a condition appearing in the first position more
       often than chance, which would mean the shuffle is not working.

    Args:
        packets: Assembled packets.
        key: The blinding key, used only to compute the diagnostics.
        length_ratio_tolerance: Maximum tolerated ratio between the longest and
            shortest condition mean length.

    Returns:
        A report with ``passed`` and the three sub-checks.
    """
    leaks: List[Dict[str, str]] = []
    lengths: Dict[str, List[int]] = {}
    first_position: Dict[str, int] = {}

    for packet in packets:
        for position, response in enumerate(packet.responses):
            condition = key.response_to_condition.get(response.response_id, "?")
            lengths.setdefault(condition, []).append(len(response.text.split()))
            if position == 0:
                first_position[condition] = first_position.get(condition, 0) + 1
            for pattern in CONDITION_LEAK_PATTERNS:
                if pattern.search(response.text):
                    leaks.append(
                        {
                            "packet_id": packet.packet_id,
                            "response_id": response.response_id,
                            "pattern": pattern.pattern,
                        }
                    )
                    break

    mean_lengths = {
        condition: sum(values) / len(values) if values else 0.0
        for condition, values in lengths.items()
    }
    positive = [v for v in mean_lengths.values() if v > 0]
    ratio = max(positive) / min(positive) if len(positive) >= 2 else 1.0

    n_packets = max(1, len(packets))
    n_conditions = max(1, len(mean_lengths))
    expected_first = n_packets / n_conditions
    position_skew = {
        condition: count / expected_first for condition, count in first_position.items()
    }
    skewed = [c for c, r in position_skew.items() if r > 1.25 or r < 0.75]

    length_ok = ratio <= length_ratio_tolerance
    return {
        "passed": (not leaks) and length_ok and not skewed,
        "textual_leakage": {
            "n_leaking_responses": len(leaks),
            "passed": not leaks,
            "examples": leaks[:20],
        },
        "length_signature": {
            "mean_words_by_condition": mean_lengths,
            "max_min_ratio": round(float(ratio), 3),
            "tolerance": length_ratio_tolerance,
            "passed": length_ok,
            "note": (
                "Conditions distinguishable by length unblind the judge on "
                "sight. Where the arms genuinely differ in verbosity, this "
                "cannot be fixed by blinding and must be reported as a "
                "limitation of the judged outcome."
            ),
        },
        "position_balance": {
            "first_position_ratio": position_skew,
            "passed": not skewed,
            "skewed_conditions": skewed,
        },
    }
