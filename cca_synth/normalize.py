"""Answer normalisation and clustering into a shared per-item option space.

The synthesiser consumes a vote matrix over a shared option space.  Nothing
produced that space until this module: agents emit *text*, and turning six texts
into "these three endorsed option 2" is a real NLP component with its own error
rate.  Left unmeasured, that error sits underneath every downstream number.

Two regimes, and the distinction decides what a benchmark can support:

**Exact-match benchmarks** (MATH, AIME, and any multiple-choice dataset) have an
official normalisation.  Clustering is then deterministic and its error rate is
essentially zero, which is why ``PILOT_FINDINGS`` should start with one of them.

**Free-form benchmarks** (TruthfulQA generative, FreshQA) have no such thing.
:func:`cluster_answers` applies a conservative lexical clustering here, and the
honest position is stated in :func:`clustering_risk_report`: this is a
*placeholder* whose disagreement rate against human labels must be measured
before any conclusion is drawn on those benchmarks.  Where measured clustering
error exceeds a benchmark's smallest effect size of interest, that benchmark
cannot support a conclusion, and the report says so in those words.

Deliberately no embedding model is used.  A learned semantic clusterer would be
better and is the right eventual answer, but it would introduce a second
uncontrolled component whose failures are correlated with the agents' own, and
``PREREGISTRATION.md`` §10.3 registers clustering error as a declared threat
rather than something to paper over.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "EXACT_MATCH_BENCHMARKS",
    "FREE_FORM_BENCHMARKS",
    "normalize_answer",
    "cluster_answers",
    "ClusteringResult",
    "needs_adjudication",
    "clustering_agreement",
    "clustering_risk_report",
]

#: Benchmarks with an official, deterministic answer normalisation.  Clustering
#: on these is exact and carries no measurable error.
EXACT_MATCH_BENCHMARKS: Tuple[str, ...] = ("math_aime", "halueval", "advconsistency")

#: Benchmarks whose answers are free text.  Clustering here is approximate and
#: its error rate must be measured before results are interpreted.
FREE_FORM_BENCHMARKS: Tuple[str, ...] = (
    "truthfulqa",
    "temporal_qa",
    "sycophancy",
    "creative_divergence",
)

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_LATEX_WRAPPERS = re.compile(r"\\(?:text|mathrm|mbox|left|right|!|,|;|\s)")
_BOXED = re.compile(r"\\boxed\s*{(.+?)}", re.DOTALL)
_TRAILING_ZEROS = re.compile(r"^(-?\d+)\.0+$")
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


def _strip_accents(text: str) -> str:
    """Remove combining marks so equivalent spellings collapse.

    Args:
        text: Input string.

    Returns:
        The string with combining characters removed.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _unwrap_braces(text: str) -> str:
    """Strip brace pairs left orphaned by removing a LaTeX wrapper.

    ``\\text{42}`` becomes ``{42}`` once the wrapper is gone, which would
    otherwise fail to match a bare ``42``.  Only a brace pair enclosing the
    *entire* string is removed, and only when it is genuinely balanced, so
    ``\\frac{1}{2}`` (which does not start with a brace) and ``{1}{2}`` (two
    groups, not one) are both left alone.

    Args:
        text: Possibly brace-wrapped text.

    Returns:
        The text with redundant outer braces removed.
    """
    while len(text) >= 2 and text[0] == "{" and text[-1] == "}":
        depth = 0
        for index, character in enumerate(text):
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                # The opening brace closes before the end, so the string is not
                # one enclosing group; leave it as is.
                if depth == 0 and index != len(text) - 1:
                    return text
        if depth != 0:
            return text
        text = text[1:-1]
    return text


def _normalize_mathematical(text: str) -> str:
    """Apply MATH/AIME-style normalisation.

    Follows the conventions the official evaluators use: extract ``\\boxed{}``
    when present, drop LaTeX presentation wrappers, remove separators and units
    that do not change the value, and canonicalise trivially equal numeric forms
    such as ``5.0`` and ``5``.

    Args:
        text: Raw answer text.

    Returns:
        The canonical form.
    """
    boxed = _BOXED.search(text)
    if boxed:
        text = boxed.group(1)
    text = _LATEX_WRAPPERS.sub("", text)
    text = _unwrap_braces(text)
    text = text.replace("$", "").replace("\\%", "").replace("%", "")
    text = text.replace(",", "").replace(" ", "")
    text = text.rstrip(".")
    # dfrac/tfrac are presentation variants of frac and must not split a cluster.
    text = re.sub(r"\\[dt]frac", r"\\frac", text)
    trailing = _TRAILING_ZEROS.match(text)
    if trailing:
        text = trailing.group(1)
    if _NUMBER.match(text):
        try:
            value = float(text)
            text = str(int(value)) if value.is_integer() else repr(value)
        except ValueError:
            pass
    return text.lower()


def _normalize_free_form(text: str) -> str:
    """Apply SQuAD-style normalisation to a free-text answer.

    Args:
        text: Raw answer text.

    Returns:
        Lowercased, accent-stripped, article- and punctuation-free text.
    """
    text = _strip_accents(text.lower())
    text = _PUNCTUATION.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_answer(text: str, benchmark: str) -> str:
    """Canonicalise one answer for its benchmark.

    Args:
        text: Raw answer text as produced by an agent.
        benchmark: Benchmark identifier from ``config.BENCHMARKS``.

    Returns:
        The canonical string; equal strings denote the same answer.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    text = text.strip()
    if not text:
        return ""
    if benchmark in EXACT_MATCH_BENCHMARKS:
        return _normalize_mathematical(text)
    return _normalize_free_form(text)


def _token_set(text: str) -> frozenset:
    """Token set of a normalised string, used for lexical overlap.

    Args:
        text: Normalised text.

    Returns:
        Frozen set of tokens.
    """
    return frozenset(text.split())


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two token sets.

    Args:
        a: First token set.
        b: Second token set.

    Returns:
        Similarity in ``[0, 1]``; ``1.0`` when both are empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class ClusteringResult:
    """Outcome of clustering one item's candidate answers.

    Attributes:
        option_ids: Cluster index per input answer, in input order.
        representatives: Canonical text per cluster, indexed by option id.
        n_options: Number of distinct clusters.
        borderline_pairs: Index pairs whose similarity fell in the adjudication
            band, i.e. close enough to be arguably the same answer but not close
            enough to merge automatically.
        exact: Whether clustering was deterministic (exact-match benchmark).
    """

    option_ids: List[int]
    representatives: List[str]
    n_options: int
    borderline_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    exact: bool = True


def cluster_answers(
    answers: Sequence[str],
    benchmark: str,
    merge_threshold: float = 0.85,
    adjudication_threshold: float = 0.60,
) -> ClusteringResult:
    """Group candidate answers into a shared option space.

    Clustering is *order invariant*: an answer's cluster must not depend on
    which agent happened to be listed first, or the vote matrix would encode
    agent order and the model's permutation invariance would be defeated at the
    data layer.  This is achieved by clustering canonical forms in sorted order
    and mapping back, and it is enforced by test.

    Args:
        answers: Raw answer texts, one per agent (empty string for no answer).
        benchmark: Benchmark identifier.
        merge_threshold: Jaccard similarity at or above which two free-form
            answers are treated as the same option.
        adjudication_threshold: Similarity at or above which a pair is flagged
            as borderline for human review, without being merged.

    Returns:
        A :class:`ClusteringResult`.

    Raises:
        ValueError: If the thresholds are inverted.
    """
    if adjudication_threshold > merge_threshold:
        raise ValueError("adjudication_threshold must not exceed merge_threshold")

    canonical = [normalize_answer(a, benchmark) for a in answers]
    exact = benchmark in EXACT_MATCH_BENCHMARKS

    # Exact benchmarks: identity on the canonical form, nothing further.
    if exact:
        representatives: List[str] = []
        index_of: Dict[str, int] = {}
        for form in sorted(set(c for c in canonical if c)):
            index_of[form] = len(representatives)
            representatives.append(form)
        return ClusteringResult(
            option_ids=[index_of.get(c, -1) for c in canonical],
            representatives=representatives,
            n_options=len(representatives),
            exact=True,
        )

    # Free-form: greedy agglomeration over *sorted* canonical forms, so the
    # result cannot depend on input order.
    distinct = sorted(set(c for c in canonical if c))
    token_sets = {form: _token_set(form) for form in distinct}
    clusters: List[List[str]] = []
    borderline: List[Tuple[int, int, float]] = []

    for form in distinct:
        placed = False
        for cluster in clusters:
            similarity = max(_jaccard(token_sets[form], token_sets[m]) for m in cluster)
            if similarity >= merge_threshold:
                cluster.append(form)
                placed = True
                break
        if not placed:
            clusters.append([form])

    # Shortest member is the representative: it is the least likely to carry
    # incidental qualifiers.
    representatives = [min(c, key=len) for c in clusters]
    form_to_option = {form: i for i, cluster in enumerate(clusters) for form in cluster}
    option_ids = [form_to_option.get(c, -1) for c in canonical]

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            similarity = max(
                _jaccard(token_sets[a], token_sets[b])
                for a in clusters[i]
                for b in clusters[j]
            )
            if similarity >= adjudication_threshold:
                borderline.append((i, j, float(similarity)))

    return ClusteringResult(
        option_ids=option_ids,
        representatives=representatives,
        n_options=len(clusters),
        borderline_pairs=borderline,
        exact=False,
    )


def needs_adjudication(result: ClusteringResult) -> bool:
    """Whether an item should be routed to a human adjudicator.

    Args:
        result: A clustering result.

    Returns:
        ``True`` when any cluster pair fell in the borderline band.
    """
    return bool(result.borderline_pairs)


def clustering_agreement(
    predicted: Sequence[Sequence[int]], human: Sequence[Sequence[int]]
) -> Dict[str, float]:
    """Measure clustering agreement against human labels.

    Agreement is computed pairwise rather than by comparing cluster ids
    directly, because cluster numbering is arbitrary: what matters is whether
    two answers were placed together, not what the group was called.  This is
    the Rand index restricted to within-item pairs.

    Args:
        predicted: Per-item cluster assignments from :func:`cluster_answers`.
        human: Per-item gold cluster assignments, same shape.

    Returns:
        Mapping with ``pairwise_agreement``, ``error_rate``, ``n_items``, and
        ``n_pairs``.

    Raises:
        ValueError: If the two sequences are not aligned.
    """
    if len(predicted) != len(human):
        raise ValueError(f"length mismatch: {len(predicted)} vs {len(human)}")

    agree = total = 0
    for predicted_item, human_item in zip(predicted, human):
        if len(predicted_item) != len(human_item):
            raise ValueError("per-item assignments are not aligned")
        n = len(predicted_item)
        for i in range(n):
            for j in range(i + 1, n):
                same_predicted = predicted_item[i] == predicted_item[j]
                same_human = human_item[i] == human_item[j]
                agree += int(same_predicted == same_human)
                total += 1

    agreement = agree / total if total else float("nan")
    return {
        "pairwise_agreement": float(agreement),
        "error_rate": float(1.0 - agreement) if total else float("nan"),
        "n_items": float(len(predicted)),
        "n_pairs": float(total),
    }


def clustering_risk_report(
    error_rate_by_benchmark: Mapping[str, float],
    sesoi_by_benchmark: Mapping[str, float],
) -> Dict[str, Any]:
    """Decide, per benchmark, whether clustering error permits a conclusion.

    ``PREREGISTRATION.md`` §10.3 registers this as a declared threat: where
    measured clustering error exceeds a benchmark's smallest effect size of
    interest, an effect of that size cannot be distinguished from clustering
    noise, and no conclusion is drawn for that benchmark.

    Args:
        error_rate_by_benchmark: Measured clustering error per benchmark.
        sesoi_by_benchmark: Smallest effect size of interest per benchmark.

    Returns:
        A per-benchmark go/no-go report with an explicit verdict string.
    """
    verdicts: Dict[str, Any] = {}
    for benchmark, error in error_rate_by_benchmark.items():
        sesoi = sesoi_by_benchmark.get(benchmark, float("nan"))
        exact = benchmark in EXACT_MATCH_BENCHMARKS
        blocked = (not exact) and error >= sesoi
        verdicts[benchmark] = {
            "clustering_error_rate": float(error),
            "sesoi": float(sesoi),
            "exact_match_benchmark": exact,
            "conclusion_permitted": not blocked,
            "verdict": (
                "Exact normalisation; clustering contributes no measurable error."
                if exact
                else (
                    f"BLOCKED: clustering error {error:.3f} meets or exceeds the "
                    f"smallest effect size of interest {sesoi:.3f}. An effect of "
                    "that size cannot be distinguished from clustering noise, so "
                    "no conclusion is drawn for this benchmark."
                    if blocked
                    else f"Clustering error {error:.3f} is below the SESOI "
                    f"{sesoi:.3f}; conclusions permitted with the error reported."
                )
            ),
        }
    return {
        "benchmarks": verdicts,
        "n_blocked": sum(1 for v in verdicts.values() if not v["conclusion_permitted"]),
        "note": (
            "Answer clustering is a component with its own error rate, not a "
            "free preprocessing step. Its error is reported alongside every "
            "free-form benchmark result."
        ),
    }
