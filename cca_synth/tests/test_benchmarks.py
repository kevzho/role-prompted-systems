"""Tests for the P7, P9, and P10 additions.

The properties pinned here are the ones that, if broken, would let an
unsupportable number reach a results table:

* an adapter cannot exist without a harm metric (L13);
* a dataset cannot be evaluated without a pinned revision and content hash (L14);
* TruthfulQA and creative divergence refuse to produce a primary number without
  judge labels (L16);
* the sycophancy battery measures both directions, and says so when it cannot;
* contamination reports are lower bounds and say so;
* the judging pipeline detects condition leakage and refuses a model judge whose
  error could exceed the effect being measured.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import (  # noqa: E402
    ADAPTERS,
    BenchmarkItem,
    DatasetPin,
    HaluEvalAdapter,
    MathAimeAdapter,
    PinMismatchError,
    SycophancyAdapter,
    describe_all,
    get_adapter,
    sesoi_by_benchmark,
    verify_pin,
)
from benchmarks.metrics import (  # noqa: E402
    binary_classification,
    exact_match,
    krippendorff_alpha,
    sycophancy_outcomes,
    truthful_and_informative,
)
from contamination import (  # noqa: E402
    MATERIAL_OVERLAP_THRESHOLD,
    NGramIndex,
    contamination_report,
    dedup_within,
    hash_release_split,
    overlap_against_corpus,
    post_cutoff_partition,
    split_leakage_report,
)
from judging import (  # noqa: E402
    build_packets,
    check_blinding,
    gate_primary_outcome,
    judge_human_agreement,
    rubric_for,
    stratified_audit_sample,
)


# ---------------------------------------------------------------------------
# Registry and pins
# ---------------------------------------------------------------------------


def test_registry_covers_every_configured_benchmark():
    """Import-time check already ran; assert the invariant explicitly."""
    from config import BENCHMARKS

    assert set(ADAPTERS) == set(BENCHMARKS)


def test_every_adapter_declares_a_harm_metric():
    """Limitation L13: every domain needs an outcome that can get worse."""
    for name in ADAPTERS:
        adapter = get_adapter(name)
        assert adapter.harm_metrics, f"{name} declares no harm metric"
        assert adapter.primary_metric


def test_adapter_without_harm_metrics_cannot_be_defined():
    """The base class refuses the subclass, not merely the run."""
    from benchmarks.base import BenchmarkAdapter

    with pytest.raises(TypeError, match="harm_metrics"):

        class Bad(BenchmarkAdapter):  # noqa: D401
            name = "bad"
            primary_metric = "accuracy"
            harm_metrics = ()
            pin = DatasetPin(
                source="x", revision="v1", sha256="", licence="MIT",
                licence_url="u", provenance="p",
            )

            def load(self, path, *, verify=True):
                return []

            def normalize(self, text):
                return text

            def score(self, items, predictions, *, extras=None):
                return {}


def test_moving_alias_is_refused_as_a_revision():
    """"main" is not a version; section 4 requires an immutable identifier."""
    for alias in ("main", "master", "latest", "HEAD"):
        with pytest.raises(ValueError, match="moving alias"):
            DatasetPin(
                source="x", revision=alias, sha256="", licence="MIT",
                licence_url="u", provenance="p",
            )


def test_unhashed_dataset_is_refused():
    """A pin with no content hash cannot be evaluated against."""
    pin = DatasetPin(
        source="x", revision="v1", sha256="", licence="MIT",
        licence_url="u", provenance="p",
    )
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
        handle.write('{"a": 1}\n')
        path = handle.name
    try:
        with pytest.raises(PinMismatchError, match="no recorded sha256"):
            verify_pin(path, pin)
        # ...but the hash can be computed for recording.
        digest = verify_pin(path, pin, allow_missing_hash=True)
        assert len(digest) == 64
    finally:
        os.unlink(path)


def test_wrong_hash_is_refused():
    """Evaluated bytes must be the pinned bytes."""
    pin = DatasetPin(
        source="x", revision="v1", sha256="0" * 64, licence="MIT",
        licence_url="u", provenance="p",
    )
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
        handle.write('{"a": 1}\n')
        path = handle.name
    try:
        with pytest.raises(PinMismatchError, match="refusing to proceed"):
            verify_pin(path, pin)
    finally:
        os.unlink(path)


def test_describe_all_flags_unbuilt_and_unpinned_benchmarks():
    """The manifest says plainly which benchmarks are not ready."""
    described = describe_all()
    assert "advconsistency" in described["unbuilt"]
    # H11: the two scoped-study benchmarks are materialised and pinned
    # (benchmarks/materialize.py); the other five remain unpinned until theirs
    # are materialised in turn.
    assert set(described["unpinned_hashes"]) == set(ADAPTERS) - {
        "halueval",
        "truthfulqa",
    }
    assert set(described["requiring_human_judgement"]) == {
        "truthfulqa",
        "creative_divergence",
    }


def test_sesoi_is_declared_for_every_benchmark():
    """normalize.clustering_risk_report needs one per benchmark."""
    sesoi = sesoi_by_benchmark()
    assert set(sesoi) == set(ADAPTERS)
    assert all(v > 0 for v in sesoi.values())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_truthful_and_informative_is_the_conjunction():
    """A truthful refusal must not score."""
    scores = truthful_and_informative([True, True, False], [False, True, True])
    assert scores["truthful_and_informative"] == pytest.approx(1 / 3)
    assert scores["truthful"] == pytest.approx(2 / 3)
    assert scores["informative"] == pytest.approx(2 / 3)


def test_unjudged_items_are_excluded_not_zeroed():
    """A missing judge label is not a score of zero."""
    scores = truthful_and_informative([True, None], [True, None])
    assert scores["truthful_and_informative"] == 1.0
    assert scores["n_judged"] == 1.0
    assert scores["n_unjudged"] == 1.0


def test_sycophancy_measures_both_directions():
    """Retention and stubbornness are computed on disjoint denominators."""
    # Item 1: right, wrongly challenged, held  -> robust_correct
    # Item 2: right, wrongly challenged, flipped -> sycophantic_flip
    # Item 3: wrong, rightly challenged, updated -> appropriate update
    # Item 4: wrong, rightly challenged, refused -> stubborn_error
    out = sycophancy_outcomes(
        initial_correct=[True, True, False, False],
        final_correct=[True, False, True, False],
        challenge_was_correct=[False, False, True, True],
    )
    assert out["retention_under_wrong_challenge"] == pytest.approx(0.5)
    assert out["capitulation_when_user_wrong"] == pytest.approx(0.5)
    assert out["appropriate_update_when_user_right"] == pytest.approx(0.5)
    assert out["inappropriate_stubbornness"] == pytest.approx(0.5)
    assert out["n_right_challenged_wrongly"] == 2.0
    assert out["n_wrong_challenged_rightly"] == 2.0


def test_never_updating_does_not_score_perfectly():
    """The trivially gameable strategy is caught by the harm outcome."""
    n = 10
    out = sycophancy_outcomes(
        initial_correct=[True] * 5 + [False] * 5,
        final_correct=[True] * 5 + [False] * 5,   # never updates
        challenge_was_correct=[False] * 5 + [True] * 5,
    )
    assert out["retention_under_wrong_challenge"] == 1.0
    # ...but stubbornness is maximal, which is the point of measuring both.
    assert out["inappropriate_stubbornness"] == 1.0


def test_sycophancy_adapter_warns_when_an_arm_is_empty():
    """An unmeasured harm outcome is stated, not silently nan."""
    adapter = SycophancyAdapter()
    items = [
        BenchmarkItem(
            item_id="syc-1", benchmark="sycophancy", question="q",
            gold_answer="paris",
            metadata={"arm": "wrong_challenge", "challenge_was_correct": False},
        )
    ]
    out = adapter.score(items, ["paris"], extras={"initial_predictions": ["paris"]})
    assert out["retention_under_wrong_challenge"] == 1.0
    assert "warning_stubbornness" in out


def test_krippendorff_alpha_bounds():
    """Perfect agreement is 1.0; systematic disagreement is well below it."""
    assert krippendorff_alpha([[3, 3], [4, 4], [5, 5]]) == pytest.approx(1.0)
    disagreeing = krippendorff_alpha([[1, 5], [5, 1], [1, 5], [5, 1]])
    assert disagreeing < 0.0


def test_krippendorff_alpha_needs_two_units():
    """A single rated item cannot establish reliability."""
    import math

    assert math.isnan(krippendorff_alpha([[3, 3]]))


def test_binary_classification_confusion_counts():
    """Precision, recall, and the two error rates come out right."""
    out = binary_classification(
        [True, True, False, False], [True, False, True, False]
    )
    assert out["tp"] == 1 and out["fp"] == 1 and out["fn"] == 1 and out["tn"] == 1
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Adapters that must refuse to guess
# ---------------------------------------------------------------------------


def test_truthfulqa_refuses_to_score_without_judge_labels():
    """L16: an LLM judge cannot silently become the primary outcome."""
    adapter = get_adapter("truthfulqa")
    items = [BenchmarkItem(item_id="t-1", benchmark="truthfulqa", question="q")]
    with pytest.raises(ValueError, match="judged"):
        adapter.score(items, ["an answer"])


def test_creative_divergence_refuses_without_two_human_ratings():
    """L16: two independent ratings per subjective primary outcome."""
    adapter = get_adapter("creative_divergence")
    items = [BenchmarkItem(item_id="c-1", benchmark="creative_divergence", question="q")]
    with pytest.raises(ValueError, match="two independent ratings"):
        adapter.score(
            items,
            ["a response"],
            extras={
                "ratings": {
                    "originality": [[4.0, None]],
                    "usefulness": [[3.0, None]],
                    "feasibility": [[3.0, None]],
                    "safety": [[5.0, None]],
                }
            },
        )


def test_advconsistency_refuses_to_load_while_unbuilt():
    """A benchmark that does not exist cannot produce a number."""
    adapter = get_adapter("advconsistency")
    with pytest.raises(RuntimeError, match="not been constructed"):
        adapter.load("nonexistent.jsonl")


def test_halueval_excludes_unparseable_rather_than_scoring_them_wrong():
    """A format failure and a wrong answer are different failures."""
    adapter = HaluEvalAdapter()
    items = [
        BenchmarkItem(
            item_id=f"h-{i}", benchmark="halueval", question="q",
            gold_answer="yes", options=("no", "yes"), gold_option=1,
        )
        for i in range(4)
    ]
    out = adapter.score(items, ["yes", "yes", "banana", "no"])
    assert out["n_excluded_unparseable"] == 1.0
    assert out["invalid_format_rate"] == pytest.approx(0.25)
    # Accuracy is over the three parseable items: 2 of 3.
    assert out["accuracy"] == pytest.approx(2 / 3)


def test_math_scores_unparseable_as_wrong_and_says_so():
    """Producing no parseable answer is a failure at the task itself."""
    adapter = MathAimeAdapter()
    items = [
        BenchmarkItem(item_id=f"m-{i}", benchmark="math_aime", question="q",
                      gold_answer="42")
        for i in range(2)
    ]
    out = adapter.score(items, ["\\boxed{42}", ""])
    assert out["accuracy"] == pytest.approx(0.5)
    assert out["invalid_format_rate"] == pytest.approx(0.5)


def test_math_normalisation_matches_the_shared_implementation():
    """Two normalisers would drift; the adapter must delegate."""
    from normalize import normalize_answer

    adapter = MathAimeAdapter()
    for text in ("\\boxed{42}", "5.0", "\\dfrac{1}{2}", "1,000"):
        assert adapter.normalize(text) == normalize_answer(text, "math_aime")


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------


def _item(item_id: str, text: str) -> BenchmarkItem:
    """Minimal item for contamination tests."""
    return BenchmarkItem(item_id=item_id, benchmark="halueval", question=text)


def test_overlap_detects_an_exact_copy():
    """The obvious case must work before the subtle ones matter."""
    items = [_item("a", "the capital of france is paris and it is very large")]
    corpus = [("doc1", "the capital of france is paris and it is very large")]
    out = overlap_against_corpus(items, corpus, corpus_name="test")
    assert out["overlap_rate"] == 1.0
    assert out["matches"][0]["exact"] is True


def test_overlap_reports_itself_as_a_lower_bound():
    """A clean result must not read as proof of cleanliness."""
    items = [_item("a", "an entirely unrelated sentence about badgers")]
    out = overlap_against_corpus(items, [("d", "something else")], corpus_name="tiny")
    assert out["overlap_rate"] == 0.0
    assert "LOWER BOUND" in out["interpretation"]


def test_material_overlap_threshold_matches_the_smallest_sesoi():
    """The threshold is tied to what the study can detect, not picked freely."""
    assert MATERIAL_OVERLAP_THRESHOLD == pytest.approx(min(sesoi_by_benchmark().values()))


def test_split_leakage_detects_a_near_duplicate_across_splits():
    """A near-twin in calibration and test invalidates the paired design."""
    shared = "the mitochondria is the powerhouse of the cell and produces atp energy"
    splits = {
        "calibration": [_item("c1", shared)],
        "test": [_item("t1", shared + " efficiently")],
    }
    out = split_leakage_report(splits, min_similarity=0.7)
    assert not out["passed"]
    assert out["n_cross_split_pairs"] == 1


def test_dedup_within_finds_internal_duplicates():
    """A duplicated item is weighted twice and may straddle a split."""
    text = "how many legs does a spider have in total when fully grown"
    out = dedup_within([_item("a", text), _item("b", text), _item("c", "unrelated")])
    assert out["n_duplicate_pairs"] == 1
    assert out["n_items_in_a_duplicate_pair"] == 2


def test_post_cutoff_partition_requires_a_cutoff():
    """A post-cutoff subset with no cutoff is meaningless."""
    with pytest.raises(ValueError, match="at least one model cutoff"):
        post_cutoff_partition([], {})


def test_post_cutoff_marks_only_items_after_every_cutoff():
    """Cross-family claims need items unseen by all compared models."""
    items = [
        BenchmarkItem(item_id="x1", benchmark="temporal_qa", question="q",
                      metadata={"as_of": "2025-01-01"}),
        BenchmarkItem(item_id="x2", benchmark="temporal_qa", question="q",
                      metadata={"as_of": "2026-06-01"}),
    ]
    out = post_cutoff_partition(
        items, {"model_a": "2025-06-01", "model_b": "2026-01-01"}
    )
    assert out["n_post_cutoff_for_all_models"] == 1
    assert items[1].metadata["post_training_cutoff"] is True
    assert items[0].metadata["post_training_cutoff"] is False


def test_contamination_report_labels_the_main_table():
    """A contaminated benchmark is labelled where readers will see it."""
    report = contamination_report(
        {
            "halueval": {
                "overlap": {"overlap_rate": 0.12, "corpus_name": "c"},
                "leakage": {"passed": True, "n_cross_split_pairs": 0},
            },
            "math_aime": {
                "overlap": {"overlap_rate": 0.001, "corpus_name": "c"},
                "leakage": {"passed": True, "n_cross_split_pairs": 0},
            },
            "truthfulqa": {},
        }
    )
    assert "CONTAMINATED" in report["main_table_labels"]["halueval"]
    assert "math_aime" not in report["main_table_labels"]
    assert report["main_table_labels"]["truthfulqa"] == "CONTAMINATION UNCHECKED"
    assert report["n_unchecked"] == 1


def test_release_split_hash_is_order_independent():
    """Publishing the hash lets a replication confirm split identity."""
    a = hash_release_split([_item("b", "x"), _item("a", "y")])
    b = hash_release_split([_item("a", "y"), _item("b", "x")])
    assert a["split_sha256"] == b["split_sha256"]


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def _judge_items(n: int):
    """Items for judging tests."""
    return [
        BenchmarkItem(item_id=f"j-{i}", benchmark="truthfulqa", question=f"q{i}")
        for i in range(n)
    ]


def test_packets_carry_no_condition_field():
    """The packet the judge sees must not name the arm."""
    items = _judge_items(5)
    packets, key = build_packets(
        items,
        {"council": ["a"] * 5, "single": ["b"] * 5},
        rubric_for("truthfulqa"),
    )
    blob = json.dumps([p.__dict__ for p in packets], default=str)
    assert "council" not in blob and "single" not in blob
    # ...but the experimenter can still unblind.
    response_id = packets[0].responses[0].response_id
    assert key.unblind(response_id)[1] in ("council", "single")


def test_blinding_requires_at_least_two_conditions():
    """With one arm there is nothing to be blind to."""
    with pytest.raises(ValueError, match="at least two conditions"):
        build_packets(_judge_items(2), {"only": ["a", "b"]}, {})


def test_presentation_order_is_randomised_but_reproducible():
    """Same seed, same order; the order must not be constant across items."""
    items = _judge_items(40)
    responses = {"a": ["ra"] * 40, "b": ["rb"] * 40}
    first, key1 = build_packets(items, responses, {}, seed=11)
    second, _ = build_packets(items, responses, {}, seed=11)
    assert [p.responses[0].response_id for p in first] == [
        p.responses[0].response_id for p in second
    ]
    conditions_first = [
        key1.response_to_condition[p.responses[0].response_id] for p in first
    ]
    assert len(set(conditions_first)) == 2, "order is constant across items"


def test_blinding_check_catches_a_condition_naming_itself():
    """Textual leakage is a blinding failure, not a warning."""
    items = _judge_items(3)
    packets, key = build_packets(
        items,
        {"a": ["As the council, I conclude X"] * 3, "b": ["I conclude X"] * 3},
        {},
    )
    report = check_blinding(packets, key)
    assert not report["passed"]
    assert report["textual_leakage"]["n_leaking_responses"] == 3


def test_blinding_check_catches_a_length_signature():
    """Arms distinguishable by length unblind the judge on sight."""
    items = _judge_items(6)
    packets, key = build_packets(
        items,
        {"long": ["word " * 200] * 6, "short": ["word"] * 6},
        {},
    )
    report = check_blinding(packets, key)
    assert not report["passed"]
    assert not report["length_signature"]["passed"]


def test_audit_sample_is_stratified_over_the_decision():
    """A uniform sample would be dominated by the easy majority class."""
    decisions = [True] * 190 + [False] * 10
    sample = stratified_audit_sample(decisions, n=100)
    assert len(sample) == 100
    # The rare class must be represented far above its 5% population share
    # would give under proportional-only allocation with no floor.
    assert sum(1 for i in sample if not decisions[i]) >= 5


def test_model_judge_refused_when_its_error_could_exceed_the_effect():
    """A judge that cannot resolve the SESOI cannot establish it."""
    audit = judge_human_agreement([True] * 80 + [False] * 20, [True] * 100)
    gate = gate_primary_outcome("truthfulqa", sesoi=0.03, audit=audit)
    assert not gate["permitted"]
    assert "smallest effect size" in gate["reason"]


def test_model_judge_refused_without_any_audit():
    """No audit means no primary outcome, per L16."""
    gate = gate_primary_outcome("truthfulqa", sesoi=0.03, audit=None)
    assert not gate["permitted"]
    assert "No human audit sample" in gate["reason"]


def test_small_audit_is_refused():
    """Below the minimum, the interval is wider than any detectable effect."""
    audit = judge_human_agreement([True] * 20, [True] * 20)
    gate = gate_primary_outcome("truthfulqa", sesoi=0.03, audit=audit)
    assert not gate["permitted"]
    assert "below the" in gate["reason"]


def test_human_rated_outcome_bypasses_the_model_judge_gate():
    """Human raters are the primary outcome; the audit gate does not apply."""
    gate = gate_primary_outcome(
        "creative_divergence", sesoi=0.25, audit=None, judge_is_human=True
    )
    assert gate["permitted"]


def test_rubric_missing_for_a_judged_benchmark_raises():
    """A judged outcome with no published rubric violates L16."""
    with pytest.raises(KeyError, match="no published rubric"):
        rubric_for("halueval")


def test_collect_ratings_refuses_thin_coverage():
    """L16's "two independent human ratings" is per response, not on average."""
    from judging import collect_ratings

    items = _judge_items(3)
    packets, key = build_packets(
        items, {"a": ["ra"] * 3, "b": ["rb"] * 3}, {}, seed=5
    )
    ids = [r.response_id for p in packets for r in p.responses]
    # Rater 1 covers everything; rater 2 skips one response.
    ratings = {
        "rater1": {rid: [4.0] for rid in ids},
        "rater2": {rid: [3.0] for rid in ids[1:]},
    }
    with pytest.raises(ValueError, match="fewer than 2 ratings"):
        collect_ratings(packets, key, ratings)


def test_collect_ratings_unblinds_back_to_conditions():
    """Ratings must map back to arms, or they cannot be compared."""
    from judging import collect_ratings

    items = _judge_items(4)
    packets, key = build_packets(
        items, {"council": ["ra"] * 4, "single": ["rb"] * 4}, {}, seed=6
    )
    ids = [r.response_id for p in packets for r in p.responses]
    ratings = {r: {rid: [4.0] for rid in ids} for r in ("rater1", "rater2")}

    coverage = collect_ratings(packets, key, ratings)
    assert coverage["n_responses"] == 8
    assert coverage["min_ratings_per_response"] == 2
    assert set(coverage["response_to_condition"].values()) == {"council", "single"}


def test_judging_report_surfaces_refused_primary_outcomes():
    """A refused outcome must be visible in the artifact, not buried."""
    from judging import judging_report

    report = judging_report(
        blinding_check={"passed": True},
        coverage={"rater_ids": ["a", "b"], "n_responses": 10,
                  "min_ratings_per_response": 2},
        audits={"truthfulqa": {"agreement_rate": 0.8}},
        gates={
            "truthfulqa": {"permitted": False, "reason": "judge error too high"},
            "halueval": {"permitted": True, "reason": "n/a"},
        },
    )
    assert report["refused_primary_outcomes"] == ["truthfulqa"]
    assert report["passed"] is False


def test_creative_rubric_is_reachable_and_covers_all_dimensions():
    """The rubric shown to raters is the one the adapter validates against."""
    from benchmarks.creative_divergence import RUBRIC_DIMENSIONS

    rubric = rubric_for("creative_divergence")
    for dimension in RUBRIC_DIMENSIONS:
        assert dimension in rubric
    assert "_preamble" in rubric
