"""Blinded judging with a human audit gate (limitation L16).

Five requirements, all enforced here rather than described:

1. Judges are blinded to condition and see responses in randomised order
   (:mod:`.blinding`).
2. Every subjective primary outcome carries at least two independent human
   ratings, with inter-rater agreement reported (:func:`collect_ratings`).
3. The rubric is written and published verbatim (:mod:`.rubrics`, plus
   ``benchmarks.creative_divergence.RUBRIC``).
4. A stratified sample of model-judge decisions is audited against human
   judgement and the agreement rate reported (:mod:`.audit`).
5. Judge disagreement is reported as a secondary outcome for TruthfulQA
   (``benchmarks.metrics.judge_disagreement``).

The gate that makes this more than paperwork is
:func:`audit.gate_primary_outcome`: a model judge whose audited error could
exceed the benchmark's smallest effect size of interest is refused as a primary
outcome, because it cannot resolve the effect being claimed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .audit import (
    MIN_AUDIT_ITEMS,
    gate_primary_outcome,
    judge_human_agreement,
    stratified_audit_sample,
)
from .blinding import (
    CONDITION_LEAK_PATTERNS,
    BlindingKey,
    JudgedResponse,
    JudgingPacket,
    build_packets,
    check_blinding,
)
from .rubrics import RUBRICS, TRUTHFULQA_RUBRIC, rubric_for

__all__ = [
    "BlindingKey",
    "JudgedResponse",
    "JudgingPacket",
    "build_packets",
    "check_blinding",
    "CONDITION_LEAK_PATTERNS",
    "MIN_AUDIT_ITEMS",
    "stratified_audit_sample",
    "judge_human_agreement",
    "gate_primary_outcome",
    "RUBRICS",
    "TRUTHFULQA_RUBRIC",
    "rubric_for",
    "collect_ratings",
    "judging_report",
]


def collect_ratings(
    packets: Sequence[JudgingPacket],
    key: BlindingKey,
    ratings: Mapping[str, Mapping[str, Sequence[Optional[float]]]],
    *,
    min_human_raters: int = 2,
) -> Dict[str, Any]:
    """Unblind collected ratings and organise them by condition and dimension.

    Args:
        packets: The packets that were judged.
        key: The blinding key.
        ratings: Mapping from rater id to ``{response_id: {dimension: score}}``
            style input, flattened as ``{rater_id: {response_id: [scores]}}``
            where the score list is ordered by rubric dimension.
        min_human_raters: Minimum independent ratings per response.

    Returns:
        Mapping from condition to per-dimension per-item rating matrices, in the
        shape ``benchmarks.metrics.rubric_means`` and ``krippendorff_alpha``
        consume, plus a coverage report.

    Raises:
        ValueError: If any response received fewer than ``min_human_raters``
            ratings. Limitation L16's "at least two independent human ratings"
            is not satisfiable by averaging over items.
    """
    rater_ids = sorted(ratings)
    by_response: Dict[str, List[Optional[Sequence[Optional[float]]]]] = {}
    for packet in packets:
        for response in packet.responses:
            by_response[response.response_id] = [
                ratings[rater].get(response.response_id) for rater in rater_ids
            ]

    undercovered = {
        response_id: sum(1 for r in rows if r is not None)
        for response_id, rows in by_response.items()
    }
    thin = {rid: n for rid, n in undercovered.items() if n < min_human_raters}
    if thin:
        raise ValueError(
            f"{len(thin)} responses carry fewer than {min_human_raters} ratings "
            f"(minimum observed: {min(thin.values())}). Limitation L16 requires "
            "at least two independent human ratings for every subjective "
            "primary outcome."
        )

    return {
        "rater_ids": rater_ids,
        "n_responses": len(by_response),
        "ratings_by_response": by_response,
        "response_to_condition": dict(key.response_to_condition),
        "response_to_item": dict(key.response_to_item),
        "min_ratings_per_response": min(undercovered.values()) if undercovered else 0,
    }


def judging_report(
    blinding_check: Mapping[str, Any],
    coverage: Mapping[str, Any],
    audits: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Assemble the judging section of the run artifact.

    Args:
        blinding_check: Output of :func:`check_blinding`.
        coverage: Output of :func:`collect_ratings`.
        audits: Per-benchmark output of :func:`judge_human_agreement`.
        gates: Per-benchmark output of :func:`gate_primary_outcome`.

    Returns:
        JSON-serialisable report, including which primary outcomes are refused.
    """
    refused = sorted(name for name, gate in gates.items() if not gate.get("permitted"))
    return {
        "blinding": blinding_check,
        "coverage": {
            "n_raters": len(coverage.get("rater_ids", [])),
            "n_responses": coverage.get("n_responses", 0),
            "min_ratings_per_response": coverage.get("min_ratings_per_response", 0),
        },
        "model_judge_audits": dict(audits),
        "primary_outcome_gates": dict(gates),
        "refused_primary_outcomes": refused,
        "passed": bool(blinding_check.get("passed")) and not refused,
        "note": (
            "A benchmark listed under refused_primary_outcomes may still be "
            "reported, but only as a secondary outcome and only with the "
            "refusal reason quoted next to it."
        ),
    }
