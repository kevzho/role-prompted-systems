"""Contamination and temporal-validity checks (limitation L15).

Limitation L15 is that the study has no contamination plan. Four things are
required, and this module implements all four:

1. **Deduplicate benchmark items against public corpora as far as practical**,
   and report the overlap rate found.
2. **Hash and publish release splits**, keeping private held-out items.
3. **Use post-training-cutoff items where temporal validity matters** --
   especially ``temporal_qa``, where the ``rare_hypothesis_checker``'s predicted
   advantage lives.
4. **Separate closed-book from retrieval-assisted performance.**

The honest limits, stated up front because a contamination check that overstates
its own coverage is worse than none:

* No public pretraining corpus is available to diff against directly. What is
  practical is (a) n-gram overlap against any corpus the user *can* materialise,
  (b) exact and near-duplicate detection *within* and *across* benchmark splits,
  and (c) canary-string and memorisation probes. Overlap measured against a
  corpus you have is a lower bound on contamination, never an upper bound.
* A zero overlap rate against the corpora checked means "not found in what was
  checked". It does not mean "not in the training data", and
  :func:`contamination_report` says so in those words in every artifact.

Per ``REMAINING_PROMPTS.md`` P9, findings are a first-class result: any benchmark
with material overlap is labelled in the main results table, not footnoted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    "MATERIAL_OVERLAP_THRESHOLD",
    "normalize_for_overlap",
    "shingles",
    "jaccard",
    "NGramIndex",
    "OverlapMatch",
    "dedup_within",
    "overlap_against_corpus",
    "split_leakage_report",
    "post_cutoff_partition",
    "hash_release_split",
    "contamination_report",
]

#: Overlap at or above which a benchmark's results are labelled contaminated in
#: the main table. Chosen to match the smallest SESOI in the pre-registration
#: (0.03): if more than that fraction of items may be memorised, a real effect
#: of SESOI size cannot be separated from recall of seen items.
MATERIAL_OVERLAP_THRESHOLD = 0.03

#: Shingle width for near-duplicate detection. Thirteen tokens is the width used
#: by several pretraining-decontamination pipelines; long enough that natural
#: language rarely collides by chance, short enough to survive light editing.
_SHINGLE_N = 13

#: Second, shorter width, emitted alongside the primary one.
#:
#: Without it, near-duplicate detection silently fails on short text. A 12-token
#: question and its 13-token near-twin each collapse to a *single* whole-text
#: shingle; those two shingles differ, so their Jaccard similarity is 0 and the
#: pair is reported as unrelated. Benchmark questions are routinely under 13
#: tokens, so this was not an edge case -- it was the common case, and it would
#: have under-reported contamination on exactly the items most likely to be
#: memorised. Emitting both widths means short texts still share sub-grams.
_SHORT_SHINGLE_N = 5

_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")


def normalize_for_overlap(text: str) -> str:
    """Canonicalise text for overlap comparison.

    Deliberately more aggressive than answer normalisation: contamination
    survives casing, punctuation, and whitespace changes, so a check that does
    not is measuring the wrong thing.

    Args:
        text: Raw text.

    Returns:
        Lowercased, punctuation-free, whitespace-collapsed text.
    """
    text = _NON_WORD.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", text).strip()


def shingles(text: str, n: int = _SHINGLE_N) -> Set[str]:
    """Token n-gram shingles of a normalised string.

    Shingles are emitted at *two* widths -- the primary ``n`` and
    :data:`_SHORT_SHINGLE_N` -- and unioned. The short width is what makes
    near-duplicate detection work on text shorter than ``n`` tokens, where a
    single-width scheme degenerates to one whole-text shingle and can only ever
    report exact matches.

    Args:
        text: Text to shingle; normalised internally.
        n: Primary shingle width in tokens.

    Returns:
        Set of shingles at both widths.
    """
    tokens = normalize_for_overlap(text).split()
    if not tokens:
        return set()

    grams: Set[str] = set()
    for width in {n, _SHORT_SHINGLE_N}:
        if len(tokens) <= width:
            grams.add(" ".join(tokens))
        else:
            grams.update(
                " ".join(tokens[i : i + width])
                for i in range(len(tokens) - width + 1)
            )
    return grams


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two shingle sets.

    Args:
        a: First set.
        b: Second set.

    Returns:
        Similarity in ``[0, 1]``; ``0.0`` when either is empty.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class OverlapMatch:
    """One detected overlap between a benchmark item and a corpus document.

    Attributes:
        item_id: The benchmark item.
        source_id: Identifier of the matching corpus document.
        similarity: Jaccard similarity over shingles.
        shared_shingles: Number of shingles in common.
        exact: Whether the normalised texts were identical.
    """

    item_id: str
    source_id: str
    similarity: float
    shared_shingles: int
    exact: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """Return the match as a plain mapping.

        Returns:
            JSON-serialisable mapping.
        """
        return {
            "item_id": self.item_id,
            "source_id": self.source_id,
            "similarity": round(float(self.similarity), 4),
            "shared_shingles": int(self.shared_shingles),
            "exact": bool(self.exact),
        }


class NGramIndex:
    """Inverted shingle index over a reference corpus.

    Built once and queried per item, because the alternative -- comparing every
    item against every document -- is quadratic and does not finish on any
    corpus large enough to be worth checking.
    """

    def __init__(self, n: int = _SHINGLE_N) -> None:
        """Create an empty index.

        Args:
            n: Shingle width.
        """
        self.n = n
        self._postings: Dict[str, Set[str]] = {}
        self._sizes: Dict[str, int] = {}
        self._exact: Dict[str, str] = {}

    def add(self, doc_id: str, text: str) -> None:
        """Index one document.

        Args:
            doc_id: Stable document identifier.
            text: Document text.
        """
        grams = shingles(text, self.n)
        self._sizes[doc_id] = len(grams)
        self._exact[hashlib.sha256(normalize_for_overlap(text).encode()).hexdigest()] = doc_id
        for gram in grams:
            self._postings.setdefault(gram, set()).add(doc_id)

    def add_many(self, documents: Iterable[Tuple[str, str]]) -> int:
        """Index many documents.

        Args:
            documents: Pairs of ``(doc_id, text)``.

        Returns:
            Number of documents indexed.
        """
        count = 0
        for doc_id, text in documents:
            self.add(doc_id, text)
            count += 1
        return count

    def __len__(self) -> int:
        """Number of indexed documents.

        Returns:
            Document count.
        """
        return len(self._sizes)

    def query(self, text: str, min_similarity: float = 0.8) -> List[Tuple[str, float, int]]:
        """Find documents overlapping a query text.

        Args:
            text: Query text.
            min_similarity: Jaccard threshold for a reported match.

        Returns:
            List of ``(doc_id, similarity, shared_shingles)``, best first.
        """
        grams = shingles(text, self.n)
        if not grams:
            return []
        shared: Dict[str, int] = {}
        for gram in grams:
            for doc_id in self._postings.get(gram, ()):  # noqa: B007
                shared[doc_id] = shared.get(doc_id, 0) + 1

        results: List[Tuple[str, float, int]] = []
        for doc_id, count in shared.items():
            union = len(grams) + self._sizes[doc_id] - count
            similarity = count / union if union else 0.0
            if similarity >= min_similarity:
                results.append((doc_id, similarity, count))
        results.sort(key=lambda row: (-row[1], row[0]))
        return results

    def exact_match(self, text: str) -> Optional[str]:
        """Find a document whose normalised text is identical.

        Args:
            text: Query text.

        Returns:
            The matching document id, or ``None``.
        """
        digest = hashlib.sha256(normalize_for_overlap(text).encode()).hexdigest()
        return self._exact.get(digest)


def dedup_within(
    items: Sequence[Any], min_similarity: float = 0.9
) -> Dict[str, Any]:
    """Find near-duplicate items inside one benchmark.

    Internal duplication is its own problem: a duplicated item is effectively
    weighted twice, and if its two copies land in different splits it leaks
    train into test regardless of any external contamination.

    Args:
        items: Benchmark items exposing ``item_id``, ``question``, and
            ``context``.
        min_similarity: Jaccard threshold.

    Returns:
        Mapping with duplicate pairs, the duplicate rate, and counts.
    """
    index = NGramIndex()
    texts: Dict[str, str] = {}
    for item in items:
        text = f"{getattr(item, 'question', '')} {getattr(item, 'context', '')}"
        texts[item.item_id] = text
        index.add(item.item_id, text)

    pairs: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for item_id, text in texts.items():
        for other_id, similarity, shared in index.query(text, min_similarity):
            if other_id == item_id:
                continue
            key = tuple(sorted((item_id, other_id)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                OverlapMatch(
                    item_id=key[0],
                    source_id=key[1],
                    similarity=similarity,
                    shared_shingles=shared,
                    exact=similarity >= 0.999,
                ).as_dict()
            )

    duplicated = {i for pair in pairs for i in (pair["item_id"], pair["source_id"])}
    return {
        "n_items": len(texts),
        "n_duplicate_pairs": len(pairs),
        "n_items_in_a_duplicate_pair": len(duplicated),
        "duplicate_rate": len(duplicated) / len(texts) if texts else float("nan"),
        "pairs": pairs[:500],
        "pairs_truncated": len(pairs) > 500,
    }


def overlap_against_corpus(
    items: Sequence[Any],
    corpus: Iterable[Tuple[str, str]],
    *,
    min_similarity: float = 0.8,
    corpus_name: str = "unnamed",
) -> Dict[str, Any]:
    """Measure benchmark overlap against a reference corpus.

    Args:
        items: Benchmark items exposing ``item_id``, ``question``, ``context``.
        corpus: Pairs of ``(doc_id, text)`` from a corpus the caller can
            materialise -- a public dataset dump, a scrape of the benchmark's
            own source pages, or previously released model development prompts.
        min_similarity: Jaccard threshold for a reported match.
        corpus_name: Recorded in the report so the reader knows what was checked.

    Returns:
        Mapping with the overlap rate, the matches, and an explicit statement of
        what a zero rate does and does not mean.
    """
    index = NGramIndex()
    n_documents = index.add_many(corpus)

    matches: List[Dict[str, Any]] = []
    contaminated: Set[str] = set()
    for item in items:
        text = f"{getattr(item, 'question', '')} {getattr(item, 'context', '')}"
        exact = index.exact_match(text)
        if exact is not None:
            matches.append(
                OverlapMatch(item.item_id, exact, 1.0, -1, exact=True).as_dict()
            )
            contaminated.add(item.item_id)
            continue
        hits = index.query(text, min_similarity)
        if hits:
            doc_id, similarity, shared = hits[0]
            matches.append(
                OverlapMatch(item.item_id, doc_id, similarity, shared).as_dict()
            )
            contaminated.add(item.item_id)

    rate = len(contaminated) / len(items) if items else float("nan")
    return {
        "corpus_name": corpus_name,
        "n_corpus_documents": n_documents,
        "n_items": len(items),
        "n_contaminated_items": len(contaminated),
        "overlap_rate": rate,
        "material": bool(rate == rate and rate >= MATERIAL_OVERLAP_THRESHOLD),
        "threshold": MATERIAL_OVERLAP_THRESHOLD,
        "matches": matches[:500],
        "matches_truncated": len(matches) > 500,
        "interpretation": (
            f"Overlap measured against '{corpus_name}' only. A rate of "
            f"{rate:.4f} means this fraction of items was found in the corpus "
            "that was checked. It is a LOWER BOUND on contamination and says "
            "nothing about corpora that were not checked, including any "
            "model's actual pretraining data."
        ),
    }


def split_leakage_report(
    splits: Mapping[str, Sequence[Any]], min_similarity: float = 0.9
) -> Dict[str, Any]:
    """Detect near-duplicate items spanning two different splits.

    This is the failure that silently invalidates a calibration split: fit a
    threshold on an item, evaluate on its near-twin, and the reported selective
    accuracy is optimistic by an unknown amount.

    Args:
        splits: Mapping from split name to its items.
        min_similarity: Jaccard threshold.

    Returns:
        Mapping with cross-split matches and a pass/fail verdict.
    """
    index = NGramIndex()
    owner: Dict[str, str] = {}
    for split_name, items in splits.items():
        for item in items:
            text = f"{getattr(item, 'question', '')} {getattr(item, 'context', '')}"
            index.add(item.item_id, text)
            owner[item.item_id] = split_name

    crossings: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for split_name, items in splits.items():
        for item in items:
            text = f"{getattr(item, 'question', '')} {getattr(item, 'context', '')}"
            for other_id, similarity, shared in index.query(text, min_similarity):
                if other_id == item.item_id or owner[other_id] == split_name:
                    continue
                key = tuple(sorted((item.item_id, other_id)))
                if key in seen:
                    continue
                seen.add(key)
                crossings.append(
                    {
                        "item_a": key[0],
                        "split_a": owner[key[0]],
                        "item_b": key[1],
                        "split_b": owner[key[1]],
                        "similarity": round(float(similarity), 4),
                        "shared_shingles": int(shared),
                    }
                )

    return {
        "n_cross_split_pairs": len(crossings),
        "pairs": crossings[:500],
        "pairs_truncated": len(crossings) > 500,
        "passed": not crossings,
        "note": (
            "Any cross-split near-duplicate invalidates the paired design for "
            "the affected items: a threshold fitted on one copy is evaluated on "
            "the other. Remove the duplicates and re-split before reporting."
        ),
    }


def post_cutoff_partition(
    items: Sequence[Any], cutoffs: Mapping[str, str]
) -> Dict[str, Any]:
    """Partition items by whether they postdate each model's training cutoff.

    Specification section 4 requires post-cutoff items where temporal validity
    matters. This tags items in place -- setting ``post_training_cutoff`` in
    metadata, which the ``temporal_qa`` and ``math_aime`` adapters already read.

    Args:
        items: Items carrying an ``as_of`` or ``year`` field in metadata.
        cutoffs: Mapping from model identifier to its ISO training-cutoff date.

    Returns:
        Per-model counts of post-cutoff items, plus the overall partition.

    Raises:
        ValueError: If ``cutoffs`` is empty. A post-cutoff partition with no
            cutoff is meaningless.
    """
    if not cutoffs:
        raise ValueError(
            "post_cutoff_partition needs at least one model cutoff; "
            "record the exact snapshot cutoffs in the run manifest"
        )

    def item_date(item: Any) -> Optional[str]:
        """Best available ISO date for an item."""
        metadata = getattr(item, "metadata", {}) or {}
        as_of = metadata.get("as_of")
        if as_of:
            return str(as_of)
        year = metadata.get("year")
        if year:
            return f"{int(year):04d}-12-31"
        return None

    per_model: Dict[str, Dict[str, Any]] = {}
    for model, cutoff in cutoffs.items():
        after: List[str] = []
        undated = 0
        for item in items:
            date = item_date(item)
            if date is None:
                undated += 1
                continue
            if date > cutoff:
                after.append(item.item_id)
        per_model[model] = {
            "cutoff": cutoff,
            "n_post_cutoff": len(after),
            "post_cutoff_fraction": len(after) / len(items) if items else float("nan"),
            "n_undated_items": undated,
            "item_ids": after[:1000],
        }

    # An item is "safe for all" only if it postdates every model's cutoff; that
    # is the subset on which a cross-family comparison is contamination-resistant.
    safe_for_all: Set[str] = set()
    if per_model:
        sets = [set(v["item_ids"]) for v in per_model.values()]
        safe_for_all = set.intersection(*sets) if sets else set()

    for item in items:
        metadata = getattr(item, "metadata", None)
        if isinstance(metadata, dict):
            metadata["post_training_cutoff"] = item.item_id in safe_for_all

    return {
        "per_model": per_model,
        "n_items": len(items),
        "n_post_cutoff_for_all_models": len(safe_for_all),
        "post_cutoff_for_all_fraction": (
            len(safe_for_all) / len(items) if items else float("nan")
        ),
        "note": (
            "Only items postdating EVERY compared model's cutoff support a "
            "contamination-resistant cross-family claim (RQ5). Items dated "
            "before some model's cutoff are reported separately."
        ),
    }


def hash_release_split(items: Sequence[Any]) -> Dict[str, Any]:
    """Hash a split for publication, per limitation L15.

    Publishing the hash lets a third party confirm they hold the same split
    without the private held-out items being released.

    Args:
        items: Items in the split.

    Returns:
        Mapping with the split hash, the item count, and the per-item id hashes.
    """
    ids = sorted(str(getattr(item, "item_id", "")) for item in items)
    joined = "\n".join(ids).encode("utf-8")
    return {
        "n_items": len(ids),
        "split_sha256": hashlib.sha256(joined).hexdigest(),
        "first_item_id": ids[0] if ids else "",
        "last_item_id": ids[-1] if ids else "",
        "note": (
            "The hash covers item ids in sorted order. Publish it alongside "
            "results so a replication can confirm split identity without "
            "requiring release of any private held-out item."
        ),
    }


def contamination_report(
    per_benchmark: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float = MATERIAL_OVERLAP_THRESHOLD,
) -> Dict[str, Any]:
    """Assemble the first-class contamination result.

    ``REMAINING_PROMPTS.md`` P9 requires this to be a result, not an appendix
    note: any benchmark showing material overlap is labelled in the main table.
    The returned ``main_table_labels`` mapping is what the report generator
    consumes to do that.

    Args:
        per_benchmark: Mapping from benchmark name to the outputs of
            :func:`overlap_against_corpus`, :func:`dedup_within`, and
            :func:`split_leakage_report`, under keys ``overlap``, ``dedup``, and
            ``leakage``. Any subset may be present.
        threshold: Overlap rate at or above which a benchmark is labelled.

    Returns:
        A report with per-benchmark verdicts and the labels the main results
        table must carry.
    """
    verdicts: Dict[str, Any] = {}
    labels: Dict[str, str] = {}
    corpora_checked: Set[str] = set()

    for benchmark, findings in per_benchmark.items():
        overlap = findings.get("overlap") or {}
        dedup = findings.get("dedup") or {}
        leakage = findings.get("leakage") or {}
        rate = overlap.get("overlap_rate", float("nan"))
        if overlap.get("corpus_name"):
            corpora_checked.add(str(overlap["corpus_name"]))

        material = bool(rate == rate and rate >= threshold)
        leaked = bool(leakage and not leakage.get("passed", True))
        unchecked = not overlap

        if unchecked:
            label = "CONTAMINATION UNCHECKED"
        elif material and leaked:
            label = f"CONTAMINATED ({rate:.1%}) + SPLIT LEAKAGE"
        elif material:
            label = f"CONTAMINATED ({rate:.1%} overlap)"
        elif leaked:
            label = "SPLIT LEAKAGE"
        else:
            label = ""

        if label:
            labels[benchmark] = label

        verdicts[benchmark] = {
            "overlap_rate": rate,
            "material_overlap": material,
            "duplicate_rate": dedup.get("duplicate_rate", float("nan")),
            "cross_split_pairs": leakage.get("n_cross_split_pairs", 0),
            "checked": not unchecked,
            "conclusion_permitted": not (material or leaked),
            "verdict": (
                "Not checked against any corpus; no contamination claim can be "
                "made in either direction."
                if unchecked
                else (
                    f"Material overlap ({rate:.4f}) at or above the {threshold} "
                    "threshold. Results for this benchmark are labelled in the "
                    "main table and cannot be read as evidence about "
                    "generalisation."
                    if material
                    else f"Overlap {rate:.4f} below the {threshold} threshold in "
                    "the corpora checked."
                )
            ),
        }

    return {
        "threshold": threshold,
        "benchmarks": verdicts,
        "main_table_labels": labels,
        "n_material": sum(1 for v in verdicts.values() if v["material_overlap"]),
        "n_unchecked": sum(1 for v in verdicts.values() if not v["checked"]),
        "corpora_checked": sorted(corpora_checked),
        "limits": (
            "Overlap was measured only against the corpora named in "
            "'corpora_checked'. No public pretraining corpus was diffed "
            "against, because none is available. Every rate here is a LOWER "
            "BOUND. A benchmark not labelled as contaminated has not been shown "
            "to be clean; it has been shown not to overlap what was checked."
        ),
    }
