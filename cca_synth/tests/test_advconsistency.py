"""Tests for the AdvConsistency construction toolkit (P8, specification §6).

The properties pinned here are the ones that would produce a benchmark that
*looks* fine and measures nothing: unbalanced factors, missing positive
controls, a minimal pair straddling the train/test boundary, or items frozen on
construction intent rather than expert judgement.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advconsistency import (  # noqa: E402
    AMBIGUOUS,
    BalanceError,
    CandidatePassage,
    LENGTH_BUCKETS,
    POSITIONS,
    TOPICS,
    adjudicate,
    ambiguity_pilot_report,
    annotator_agreement,
    assign_cells,
    balance_report,
    build_annotation_packets,
    control_report,
    freeze_split,
    make_pair,
    pair_integrity_report,
)


def _passage(**kwargs) -> CandidatePassage:
    """Build a passage with sane defaults."""
    defaults = dict(
        item_id="advc-0001-c",
        pair_id="0001",
        passage=" ".join(["word"] * 200),
        contains_contradiction=False,
        passage_length_bucket="short",
        topic="natural_science",
    )
    defaults.update(kwargs)
    return CandidatePassage(**defaults)


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------


def test_contradictory_passage_must_record_its_position():
    """Position balance cannot be verified if position is not recorded."""
    with pytest.raises(ValueError, match="where it sits"):
        _passage(contains_contradiction=True, contradiction_position=None)


def test_positive_control_cannot_contain_a_contradiction():
    """A control is a passage with no real contradiction; that is the point."""
    with pytest.raises(ValueError, match="defeats its purpose"):
        _passage(
            contains_contradiction=True,
            contradiction_position="early",
            is_positive_control=True,
        )


def test_make_pair_requires_a_unique_substitution_site():
    """A substitution hitting zero or several sites is not a minimal pair."""
    passage = "The tower is red. The tower is red."
    with pytest.raises(ValueError, match="exactly once"):
        make_pair(
            passage, "The tower is red.", "The tower is blue.",
            topic="history", length_bucket="short", position="early",
            distance="adjacent", seed="s",
        )


def test_make_pair_differs_in_exactly_one_clause():
    """The two members must differ only where the clause was swapped."""
    base = "The summit is 3,000 metres high. Climbers need oxygen above 2,500 metres."
    pair = make_pair(
        base,
        "The summit is 3,000 metres high.",
        "The summit is 300 metres high.",
        topic="geography", length_bucket="short", position="early",
        distance="adjacent", seed="s1",
    )
    assert pair.consistent.passage == base
    assert "300 metres" in pair.contradictory.passage
    assert pair.consistent.pair_id == pair.contradictory.pair_id
    assert pair.contradictory.contains_contradiction is True
    assert pair.consistent.contains_contradiction is False
    # Everything except the clause is held constant.
    assert pair.consistent.topic == pair.contradictory.topic
    assert pair.consistent.passage_length_bucket == pair.contradictory.passage_length_bucket


def test_cross_paragraph_distance_marks_long_range():
    """The long-range subset is what measures the predicted trade-off."""
    pair = make_pair(
        "A B C. D E F.", "A B C.", "A B X.",
        topic="history", length_bucket="short", position="late",
        distance="cross_paragraph", seed="s2",
    )
    assert pair.contradictory.requires_long_range is True


def test_assign_cells_is_balanced_in_any_prefix():
    """Construction is usually interrupted; a partial run must stay balanced."""
    plan = assign_cells(60)
    topics = [c["topic"] for c in plan]
    counts = [topics.count(t) for t in TOPICS]
    assert max(counts) - min(counts) <= 1


# ---------------------------------------------------------------------------
# Balance gates
# ---------------------------------------------------------------------------


def _balanced_corpus() -> list:
    """A corpus balanced across all four factors, with controls."""
    passages = []
    n = 0
    for topic in TOPICS:
        for bucket in LENGTH_BUCKETS:
            for position, distance in zip(POSITIONS, ("adjacent", "same_paragraph", "cross_paragraph")):
                low, _ = LENGTH_BUCKETS[bucket]
                text = " ".join(["word"] * (low + 5))
                pair_id = f"p{n:04d}"
                passages.append(
                    CandidatePassage(
                        item_id=f"advc-{pair_id}-c", pair_id=pair_id, passage=text,
                        contains_contradiction=False, passage_length_bucket=bucket,
                        topic=topic,
                    )
                )
                passages.append(
                    CandidatePassage(
                        item_id=f"advc-{pair_id}-x", pair_id=pair_id, passage=text,
                        contains_contradiction=True, contradiction_position=position,
                        contradiction_distance=distance, passage_length_bucket=bucket,
                        topic=topic,
                    )
                )
                n += 1
    # Positive controls at the declared 25% target: with 2n contradictory/
    # consistent passages already present, adding ~2n/3 controls lands the
    # control fraction near 0.25. Spread evenly over topics and buckets so they
    # do not themselves unbalance the design.
    n_controls = round(len(passages) / 3)
    buckets = list(LENGTH_BUCKETS)
    for i in range(n_controls):
        pair_id = f"ctl{i:03d}"
        bucket = buckets[i % len(buckets)]
        low, _ = LENGTH_BUCKETS[bucket]
        passages.append(
            CandidatePassage(
                item_id=f"advc-{pair_id}-c", pair_id=pair_id,
                passage=" ".join(["word"] * (low + 5)),
                contains_contradiction=False, is_positive_control=True,
                passage_length_bucket=bucket, topic=TOPICS[i % len(TOPICS)],
            )
        )
    return passages


def _distinct_pair_passages(n: int) -> list:
    """n passages, each from a *different* pair.

    Allocation refuses to give one annotator both members of a pair, so a
    fixture drawn from few pairs cannot be spread across few annotators. Tests
    that only care about packet *content* use this instead.
    """
    return [
        CandidatePassage(
            item_id=f"advc-solo{i:03d}-c", pair_id=f"solo{i:03d}",
            passage=" ".join(["word"] * 200), contains_contradiction=False,
            passage_length_bucket="short", topic=TOPICS[i % len(TOPICS)],
        )
        for i in range(n)
    ]


def test_balanced_corpus_passes_every_gate():
    """The happy path must be reachable or the gates are untestable."""
    passages = _balanced_corpus()
    assert balance_report(passages)["passed"], balance_report(passages)["failing_factors"]


def test_topic_imbalance_is_caught():
    """A topic-specific effect must not masquerade as a general one."""
    passages = _balanced_corpus()
    passages += [
        CandidatePassage(
            item_id=f"advc-extra{i}-c", pair_id=f"extra{i}",
            passage=" ".join(["word"] * 200), contains_contradiction=False,
            passage_length_bucket="short", topic="history",
        )
        for i in range(60)
    ]
    report = balance_report(passages)
    assert not report["passed"]
    assert "topic" in report["failing_factors"]


def test_position_balance_uses_contradictory_passages_only():
    """Consistent passages have no position and would dilute the check."""
    report = balance_report(_balanced_corpus())
    assert report["n_with_contradiction"] < report["n_passages"]
    assert report["factors"]["contradiction_position"]["n"] == report["n_with_contradiction"]


def test_missing_positive_controls_are_caught():
    """Without controls, a flag-everything system scores perfectly."""
    passages = [p for p in _balanced_corpus() if not p.is_positive_control]
    report = control_report(passages)
    assert not report["passed"]
    assert report["flag_everything_accuracy"] == pytest.approx(0.5)


def test_length_bucket_mismatch_is_caught():
    """A 'long' passage that is 200 words invalidates the long-range subset."""
    from advconsistency import length_report

    bad = _passage(passage_length_bucket="long")  # 200 words, declared 450-800
    report = length_report([bad])
    assert not report["passed"]
    assert report["n_mismatched"] == 1


def test_split_pair_is_caught():
    """A pair straddling the boundary leaks almost the whole answer."""
    a = _passage(item_id="advc-9-c", pair_id="9")
    b = _passage(item_id="advc-9-x", pair_id="9", contains_contradiction=True,
                 contradiction_position="early")
    report = pair_integrity_report({"train": [a], "test": [b]})
    assert not report["passed"]
    assert report["n_broken_pairs"] == 1


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------


def test_freeze_refuses_without_expert_labels():
    """Construction intent is not a label; §6 requires expert adjudication."""
    passages = _balanced_corpus()
    with pytest.raises(BalanceError, match="no adjudicated expert label"):
        freeze_split({"train": passages}, adjudicated_labels={})


def test_freeze_reports_every_failure_at_once():
    """One pass should be enough to fix everything."""
    passages = [p for p in _balanced_corpus() if not p.is_positive_control][:8]
    with pytest.raises(BalanceError) as excinfo:
        freeze_split({"train": passages}, adjudicated_labels={})
    message = str(excinfo.value)
    assert "positive controls" in message
    assert "annotation" in message


def test_freeze_succeeds_and_hashes_the_hidden_split():
    """A valid corpus freezes, and the test split is marked non-public."""
    passages = _balanced_corpus()
    labels = {p.item_id: p.contains_contradiction for p in passages}
    # Split by pair so no pair straddles the boundary.
    pairs = sorted({p.pair_id for p in passages})
    train_pairs = set(pairs[: len(pairs) // 2])
    splits = {
        "train": [p for p in passages if p.pair_id in train_pairs],
        "test": [p for p in passages if p.pair_id not in train_pairs],
    }
    manifest = freeze_split(splits, labels)
    assert len(manifest["splits"]["test"]["sha256"]) == 64
    assert manifest["splits"]["test"]["public"] is False
    assert manifest["splits"]["train"]["public"] is True
    assert manifest["label_vs_intent_disagreements"] == 0


def test_freeze_reports_expert_overrules_of_construction_intent():
    """Experts disagreeing with the constructor is a publishable rate."""
    passages = _balanced_corpus()
    labels = {p.item_id: p.contains_contradiction for p in passages}
    flipped = passages[0].item_id
    labels[flipped] = not labels[flipped]
    manifest = freeze_split({"train": passages}, labels)
    assert manifest["label_vs_intent_disagreements"] == 1
    assert flipped in manifest["disagreeing_item_ids"]


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def test_no_annotator_sees_both_members_of_a_pair():
    """Seeing both members makes the task trivial and the statistic meaningless."""
    passages = _balanced_corpus()
    annotators = [f"expert{i}" for i in range(6)]
    packets, join = build_annotation_packets(passages, annotators, ratings_per_item=3)

    by_item = {p.item_id: p for p in passages}
    for annotator, tasks in packets.items():
        seen_pairs = [by_item[t.item_id].pair_id for t in tasks]
        assert len(seen_pairs) == len(set(seen_pairs)), (
            f"{annotator} received both members of a pair"
        )


def test_every_item_receives_the_required_independent_ratings():
    """L16's two-rating minimum is per item, not on average."""
    passages = _balanced_corpus()
    packets, _ = build_annotation_packets(
        passages, [f"e{i}" for i in range(6)], ratings_per_item=3
    )
    counts = {}
    for tasks in packets.values():
        for task in tasks:
            counts[task.item_id] = counts.get(task.item_id, 0) + 1
    assert set(counts) == {p.item_id for p in passages}
    assert min(counts.values()) == 3


def test_annotator_view_hides_the_item_id():
    """Item ids encode pair membership via their -c/-x suffix."""
    packets, _ = build_annotation_packets(
        _distinct_pair_passages(12), ["a", "b"], ratings_per_item=2
    )
    view = packets["a"][0].for_annotator()
    assert "item_id" not in view
    assert set(view) == {"task_id", "passage", "instructions"}


def test_instructions_carry_the_canonical_definition():
    """The definition must not be restated and allowed to drift."""
    from benchmarks.advconsistency import PHENOMENON_DEFINITION

    packets, _ = build_annotation_packets(
        _distinct_pair_passages(6), ["a", "b"], ratings_per_item=2
    )
    instructions = packets["a"][0].instructions
    assert PHENOMENON_DEFINITION in instructions
    assert AMBIGUOUS in instructions


def test_too_few_annotators_raises():
    """Silently giving fewer ratings would break L16 invisibly."""
    with pytest.raises(ValueError, match="independent ratings"):
        build_annotation_packets(_balanced_corpus(), ["only_one"], ratings_per_item=3)


def test_ratings_per_item_below_two_is_refused():
    """L16 requires at least two independent ratings."""
    with pytest.raises(ValueError, match="at least 2"):
        build_annotation_packets(_balanced_corpus(), ["a", "b"], ratings_per_item=1)


# ---------------------------------------------------------------------------
# Agreement, adjudication, pilot
# ---------------------------------------------------------------------------


def test_ambiguous_is_excluded_from_alpha_not_coerced():
    """Coercing 'ambiguous' to a class manufactures agreement."""
    responses = {
        "a": {"i1": "yes", "i2": AMBIGUOUS, "i3": "no"},
        "b": {"i1": "yes", "i2": AMBIGUOUS, "i3": "no"},
    }
    report = annotator_agreement(responses)
    assert report["n_items_marked_ambiguous"] == 1
    assert report["ambiguity_rate"] == pytest.approx(1 / 3)
    assert report["n_items_disputed"] == 0


def test_disputed_items_are_counted():
    """Genuine disagreement is distinct from ambiguity."""
    responses = {"a": {"i1": "yes"}, "b": {"i1": "no"}}
    report = annotator_agreement(responses)
    assert report["n_items_disputed"] == 1


def test_adjudication_refers_rather_than_guesses():
    """An unresolved item must not acquire a label by default."""
    responses = {"a": {"i1": "yes"}, "b": {"i1": "no"}}
    ruling = adjudicate(responses)
    assert ruling["n_referred"] == 1
    assert ruling["complete"] is False
    assert "i1" not in ruling["labels"]


def test_adjudicator_can_discard():
    """Discards are counted, not silently dropped."""
    responses = {"a": {"i1": "yes"}, "b": {"i1": "no"}}
    ruling = adjudicate(responses, {"i1": "discard"})
    assert ruling["n_discarded"] == 1
    assert ruling["complete"] is True
    assert "i1" not in ruling["labels"]


def test_unanimous_items_resolve_without_an_adjudicator():
    """Adjudication is for disagreement, not a bottleneck on everything."""
    responses = {"a": {"i1": "yes", "i2": "no"}, "b": {"i1": "yes", "i2": "no"}}
    ruling = adjudicate(responses)
    assert ruling["labels"] == {"i1": True, "i2": False}
    assert ruling["n_resolved_unanimously"] == 2
    assert ruling["complete"] is True


def test_pilot_blocks_scaling_on_low_agreement():
    """Alpha below target means the labels themselves are uncertain."""
    verdict = ambiguity_pilot_report(
        {"krippendorff_alpha": 0.55, "ambiguity_rate": 0.05},
        {"n_items": 100, "n_discarded": 5},
    )
    assert not verdict["passed"]
    assert "NO-GO" in verdict["verdict"]
    assert "alpha" in verdict["verdict"]


def test_pilot_blocks_scaling_on_high_discard_rate():
    """Too many discards means the templates produce ambiguous items."""
    verdict = ambiguity_pilot_report(
        {"krippendorff_alpha": 0.92, "ambiguity_rate": 0.30},
        {"n_items": 100, "n_discarded": 35},
    )
    assert not verdict["passed"]
    assert "narrow the phenomenon definition" in verdict["verdict"].lower()


def test_pilot_passes_on_a_clean_pilot():
    """The go path must be reachable."""
    verdict = ambiguity_pilot_report(
        {"krippendorff_alpha": 0.88, "ambiguity_rate": 0.04},
        {"n_items": 150, "n_discarded": 9},
    )
    assert verdict["passed"]
    assert verdict["verdict"].startswith("GO")
