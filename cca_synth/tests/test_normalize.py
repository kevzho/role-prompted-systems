"""Tests for answer normalisation and clustering.

Order invariance is the one that matters most: if an answer's cluster depended
on which agent was listed first, the vote matrix would encode agent order and
the model's permutation invariance would be defeated at the data layer, below
where any of the model tests could see it.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalize import (  # noqa: E402
    EXACT_MATCH_BENCHMARKS,
    ClusteringResult,
    cluster_answers,
    clustering_agreement,
    clustering_risk_report,
    needs_adjudication,
    normalize_answer,
)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_mathematical_normalisation_collapses_presentation_variants() -> None:
    """Answers differing only in LaTeX presentation must normalise identically."""
    equivalent = [
        r"\boxed{42}",
        "42",
        "42.0",
        r"$42$",
        r"\text{42}",
        " 42 ",
        "42.",
    ]
    canonical = {normalize_answer(a, "math_aime") for a in equivalent}
    assert len(canonical) == 1, f"did not collapse: {canonical}"


def test_mathematical_normalisation_preserves_real_differences() -> None:
    """Genuinely different values must not be merged."""
    assert normalize_answer("42", "math_aime") != normalize_answer("43", "math_aime")
    assert normalize_answer("1/2", "math_aime") != normalize_answer("1/3", "math_aime")
    assert normalize_answer("-5", "math_aime") != normalize_answer("5", "math_aime")


def test_frac_variants_collapse() -> None:
    """dfrac and tfrac are presentation variants of frac."""
    forms = [r"\frac{1}{2}", r"\dfrac{1}{2}", r"\tfrac{1}{2}"]
    assert len({normalize_answer(f, "math_aime") for f in forms}) == 1


def test_free_form_normalisation_strips_articles_and_punctuation() -> None:
    """SQuAD-style normalisation for free text."""
    equivalent = ["The Eiffel Tower", "eiffel tower", "Eiffel Tower.", "  the  Eiffel Tower  "]
    assert len({normalize_answer(a, "truthfulqa") for a in equivalent}) == 1


def test_normalisation_is_idempotent() -> None:
    """Normalising twice must equal normalising once."""
    for benchmark in ("math_aime", "truthfulqa"):
        for raw in [r"\boxed{7}", "The Answer.", "  spaced  ", "42.0", "café"]:
            once = normalize_answer(raw, benchmark)
            assert normalize_answer(once, benchmark) == once, f"{raw!r} on {benchmark}"


def test_normalisation_rejects_non_strings() -> None:
    """A numeric answer must raise rather than be silently coerced."""
    with pytest.raises(TypeError):
        normalize_answer(42, "math_aime")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def test_clustering_is_order_invariant() -> None:
    """Cluster membership must not depend on the order agents are listed in.

    If it did, the vote matrix would encode agent order, defeating the model's
    permutation invariance at the data layer where no model test would see it.
    """
    answers = [
        "the Eiffel Tower in Paris",
        "Eiffel Tower Paris",
        "the Statue of Liberty",
        "Statue of Liberty",
    ]
    forward = cluster_answers(answers, "truthfulqa")
    backward = cluster_answers(list(reversed(answers)), "truthfulqa")

    # Compare the *partition*, not the labels, since cluster numbering is
    # arbitrary and reversing the input legitimately renumbers.
    def partition(result: ClusteringResult, texts):
        groups = {}
        for text, option in zip(texts, result.option_ids):
            groups.setdefault(option, set()).add(text)
        return {frozenset(g) for g in groups.values()}

    assert partition(forward, answers) == partition(backward, list(reversed(answers)))


def test_semantically_identical_distinct_texts_merge() -> None:
    """Six agents phrasing one answer differently must yield one option."""
    answers = [
        "The Eiffel Tower",
        "the eiffel tower",
        "Eiffel Tower",
        "the Eiffel Tower.",
    ]
    result = cluster_answers(answers, "truthfulqa")
    assert result.n_options == 1
    assert len(set(result.option_ids)) == 1


def test_distinct_answers_stay_distinct() -> None:
    """Unrelated answers must not be merged."""
    answers = ["Paris", "London", "Berlin"]
    result = cluster_answers(answers, "truthfulqa")
    assert result.n_options == 3
    assert len(set(result.option_ids)) == 3


def test_exact_benchmarks_cluster_deterministically() -> None:
    """Exact-match benchmarks use identity on the canonical form."""
    result = cluster_answers([r"\boxed{42}", "42.0", "43"], "math_aime")
    assert result.exact
    assert result.n_options == 2
    assert result.option_ids[0] == result.option_ids[1] != result.option_ids[2]
    assert not result.borderline_pairs


def test_empty_answers_map_to_no_option() -> None:
    """An agent that produced no answer must not create a phantom option."""
    result = cluster_answers(["Paris", "", "Paris"], "truthfulqa")
    assert result.option_ids[1] == -1
    assert result.n_options == 1


def test_borderline_pairs_are_flagged_not_merged() -> None:
    """Near-miss clusters must surface for adjudication rather than merging."""
    # Overlapping but not identical: shares two of three tokens.
    result = cluster_answers(
        ["carbon dioxide gas", "carbon dioxide", "helium"],
        "truthfulqa",
        merge_threshold=0.95,
        adjudication_threshold=0.50,
    )
    assert result.n_options >= 2
    assert needs_adjudication(result), "no borderline pair flagged"


def test_adjudication_thresholds_must_be_ordered() -> None:
    """An adjudication band above the merge threshold is incoherent."""
    with pytest.raises(ValueError, match="must not exceed"):
        cluster_answers(["a"], "truthfulqa", merge_threshold=0.5, adjudication_threshold=0.9)


# ---------------------------------------------------------------------------
# Error measurement and the go/no-go gate
# ---------------------------------------------------------------------------


def test_clustering_agreement_is_label_invariant() -> None:
    """Agreement must depend on the partition, not on cluster numbering."""
    predicted = [[0, 0, 1], [2, 2, 2]]
    human = [[7, 7, 9], [1, 1, 1]]
    result = clustering_agreement(predicted, human)
    assert result["pairwise_agreement"] == pytest.approx(1.0)
    assert result["error_rate"] == pytest.approx(0.0)


def test_clustering_agreement_detects_a_split() -> None:
    """Splitting a true cluster must reduce agreement."""
    predicted = [[0, 1, 2]]
    human = [[0, 0, 0]]
    result = clustering_agreement(predicted, human)
    assert result["pairwise_agreement"] == pytest.approx(0.0)
    assert result["n_pairs"] == 3.0


def test_clustering_agreement_rejects_misaligned_input() -> None:
    """Mismatched shapes must raise rather than silently truncate."""
    with pytest.raises(ValueError):
        clustering_agreement([[0, 1]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        clustering_agreement([[0, 1, 2]], [[0, 1]])


def test_risk_report_blocks_a_benchmark_when_error_exceeds_sesoi() -> None:
    """The go/no-go gate must actually block, in those words."""
    report = clustering_risk_report(
        error_rate_by_benchmark={"truthfulqa": 0.08, "temporal_qa": 0.01, "math_aime": 0.0},
        sesoi_by_benchmark={"truthfulqa": 0.03, "temporal_qa": 0.03, "math_aime": 0.03},
    )
    assert report["benchmarks"]["truthfulqa"]["conclusion_permitted"] is False
    assert "BLOCKED" in report["benchmarks"]["truthfulqa"]["verdict"]
    assert report["benchmarks"]["temporal_qa"]["conclusion_permitted"] is True
    assert report["n_blocked"] == 1


def test_risk_report_never_blocks_exact_benchmarks() -> None:
    """Exact normalisation contributes no clustering error by construction."""
    report = clustering_risk_report(
        error_rate_by_benchmark={b: 0.0 for b in EXACT_MATCH_BENCHMARKS},
        sesoi_by_benchmark={b: 0.03 for b in EXACT_MATCH_BENCHMARKS},
    )
    assert report["n_blocked"] == 0
    for benchmark in EXACT_MATCH_BENCHMARKS:
        assert report["benchmarks"][benchmark]["exact_match_benchmark"]
