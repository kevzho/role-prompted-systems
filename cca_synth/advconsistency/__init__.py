"""AdvConsistency construction toolkit (specification section 6).

The benchmark itself does not exist yet and cannot be built without domain
experts. What this package supplies is everything *around* the expert judgement:
paired-passage construction, balanced cell planning, blinded annotation packets,
agreement statistics, adjudication, the ambiguity pilot's go/no-go rule, and a
freeze that refuses on any section 6 violation.

The division of labour is deliberate. Every step a machine can do correctly is
done here, so that expensive expert time is spent only on the one thing experts
are for — deciding whether a passage genuinely contradicts itself.

Typical flow::

    from advconsistency import (
        assign_cells, make_pair, balance_report,
        build_annotation_packets, annotator_agreement,
        adjudicate, ambiguity_pilot_report, freeze_split,
    )

    cells = assign_cells(n_pairs=75)              # balanced design first
    pairs = [make_pair(...) for cell in cells]    # authors fill the design
    passages = [p for pair in pairs for p in pair.as_list()]

    assert balance_report(passages)["passed"]
    packets, join = build_annotation_packets(passages, annotators, ratings_per_item=3)
    # ... experts annotate ...
    agreement = annotator_agreement(responses)
    ruling = adjudicate(responses, adjudicator_rulings)
    pilot = ambiguity_pilot_report(agreement, ruling)
    if pilot["passed"]:
        manifest = freeze_split(splits, ruling["labels"])

``freeze_split`` is the gate: it raises unless the four balanced factors hold,
positive controls are present, length buckets are honest, no minimal pair is
split across two splits, and every item carries an adjudicated expert label.
"""

from __future__ import annotations

from .annotate import (
    AMBIGUOUS,
    AnnotationTask,
    adjudicate,
    ambiguity_pilot_report,
    annotator_agreement,
    build_annotation_packets,
)
from .balance import (
    DEFAULT_TOLERANCE,
    BalanceError,
    balance_report,
    control_report,
    freeze_split,
    length_report,
    pair_integrity_report,
)
from .construct import (
    DISTANCES,
    LENGTH_BUCKETS,
    POSITIONS,
    TARGET_CONTROL_FRACTION,
    TOPICS,
    CandidatePassage,
    PassagePair,
    assign_cells,
    make_pair,
    read_candidates,
    write_candidates,
)

__all__ = [
    "AMBIGUOUS",
    "AnnotationTask",
    "BalanceError",
    "CandidatePassage",
    "DEFAULT_TOLERANCE",
    "DISTANCES",
    "LENGTH_BUCKETS",
    "POSITIONS",
    "PassagePair",
    "TARGET_CONTROL_FRACTION",
    "TOPICS",
    "adjudicate",
    "ambiguity_pilot_report",
    "annotator_agreement",
    "assign_cells",
    "balance_report",
    "build_annotation_packets",
    "control_report",
    "freeze_split",
    "length_report",
    "make_pair",
    "pair_integrity_report",
    "read_candidates",
    "write_candidates",
]
