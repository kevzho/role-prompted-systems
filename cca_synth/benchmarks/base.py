"""Core types for benchmark adapters.

Specification section 5 names a primary outcome and a set of secondary and harm
outcomes for each domain.  Section 4 additionally requires frozen dataset
versions, separated development/calibration/validation/test sets, and disclosure
of retrieval and tool access per condition.  This module supplies the types that
make those requirements structural rather than aspirational:

* :class:`DatasetPin` refuses to describe a dataset without a revision and a
  content hash, so "we used TruthfulQA" cannot appear in a manifest without
  saying *which* TruthfulQA;
* :class:`BenchmarkAdapter` requires a primary metric, secondary metrics, *and*
  harm metrics, so an adapter cannot be written that reports only good news;
* :func:`split_items` reuses ``dataset.hash_split`` so that an item's split is a
  deterministic function of its id and cannot drift as the corpus grows.

Datasets are not downloaded here.  An adapter is given a path to a materialised
local copy and verifies its hash against the pin.  That keeps the network out of
the evaluation path and makes an unpinned run fail loudly instead of quietly
evaluating on whatever the hub currently serves.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "BenchmarkItem",
    "DatasetPin",
    "BenchmarkAdapter",
    "ToolAccess",
    "split_items",
    "file_sha256",
    "verify_pin",
    "PinMismatchError",
]


class PinMismatchError(RuntimeError):
    """Raised when a materialised dataset does not match its declared pin."""


@dataclass(frozen=True)
class ToolAccess:
    """Tool availability for one condition, disclosed per specification section 4.

    Attributes:
        retrieval: A retrieval index was available.
        browsing: Live web browsing was available.
        calculator: A calculator tool was available.
        code_execution: Code execution was available.
    """

    retrieval: bool = False
    browsing: bool = False
    calculator: bool = False
    code_execution: bool = False

    def as_dict(self) -> Dict[str, bool]:
        """Return the disclosure as a plain mapping.

        Returns:
            Mapping suitable for a manifest.
        """
        return {
            "retrieval": self.retrieval,
            "browsing": self.browsing,
            "calculator": self.calculator,
            "code_execution": self.code_execution,
        }

    @property
    def closed_book(self) -> bool:
        """Whether the condition is closed-book.

        Returns:
            ``True`` when no external knowledge source was reachable.
        """
        return not (self.retrieval or self.browsing)


@dataclass(frozen=True)
class DatasetPin:
    """An exact, checkable identification of one dataset version.

    Limitation L14 requires frozen versions; limitation L15 requires hashed
    release splits.  A moving alias such as ``main`` is rejected: the whole
    point is that the artifact identifies the bytes that were evaluated.

    Attributes:
        source: Hub identifier, URL, or ``"constructed"`` for datasets built by
            this project.
        revision: Immutable revision -- a commit SHA, a version tag, or a release
            date.  ``"main"``, ``"latest"``, and ``"master"`` are refused.
        sha256: Hash of the materialised local file.  Empty only for
            ``constructed`` datasets whose hash is produced at build time.
        licence: SPDX identifier or the licence's short name.
        licence_url: Where the licence text lives.
        provenance: One-line statement of where the items came from.
        cutoff_sensitive: Whether temporal validity matters for this dataset,
            i.e. whether items may postdate a model's training cutoff.
        personal_data: Whether the corpus contains personal data, which triggers
            the handling rules in ``ETHICS_AND_DATA_GOVERNANCE.md``.
    """

    source: str
    revision: str
    sha256: str
    licence: str
    licence_url: str
    provenance: str
    cutoff_sensitive: bool = False
    personal_data: bool = False

    def __post_init__(self) -> None:
        if not self.revision:
            raise ValueError("DatasetPin.revision is required; an unpinned dataset cannot be evaluated")
        if self.revision.lower() in {"main", "master", "latest", "head"}:
            raise ValueError(
                f"DatasetPin.revision={self.revision!r} is a moving alias; "
                "specification section 4 requires an immutable identifier"
            )
        if not self.licence:
            raise ValueError("DatasetPin.licence is required")

    def as_dict(self) -> Dict[str, Any]:
        """Return the pin as a manifest-ready mapping.

        Returns:
            JSON-serialisable mapping.
        """
        return {
            "source": self.source,
            "revision": self.revision,
            "sha256": self.sha256,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "provenance": self.provenance,
            "cutoff_sensitive": self.cutoff_sensitive,
            "personal_data": self.personal_data,
        }


@dataclass(frozen=True)
class BenchmarkItem:
    """One evaluation item, in the form the orchestrator dispatches.

    Attributes:
        item_id: Stable identifier; determines the split via
            ``dataset.hash_split`` and must not change between runs.
        benchmark: Benchmark identifier from ``config.BENCHMARKS``.
        question: The question or instruction shown to every agent.
        context: Optional passage or supporting text.
        gold_answer: Canonical gold answer text; empty when the item has no
            single correct answer (creative divergence).
        options: Fixed option texts when the item is multiple-choice.
        gold_option: Index into ``options``; ``-1`` when not applicable.
        metadata: Per-benchmark extras -- timestamps, contradiction position,
            challenge protocol state, and so on.  Never carries personal data.
    """

    item_id: str
    benchmark: str
    question: str
    context: str = ""
    gold_answer: str = ""
    options: Tuple[str, ...] = ()
    gold_option: int = -1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gold_option >= 0 and self.gold_option >= len(self.options):
            raise ValueError(
                f"item {self.item_id}: gold_option {self.gold_option} is out of "
                f"range for {len(self.options)} options"
            )


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Hash a file's contents.

    Args:
        path: File to hash.
        chunk_size: Read size in bytes.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pin(path: str, pin: DatasetPin, *, allow_missing_hash: bool = False) -> str:
    """Check a materialised dataset against its pin.

    Args:
        path: Local file the adapter is about to read.
        pin: The declared pin.
        allow_missing_hash: Permit a pin whose ``sha256`` is empty, returning the
            computed hash so it can be recorded.  Used when first materialising a
            constructed dataset.

    Returns:
        The computed hash.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        PinMismatchError: If the hash differs from the pin, or the pin carries no
            hash and ``allow_missing_hash`` is ``False``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist; materialise {pin.source}@{pin.revision} first"
        )
    computed = file_sha256(path)
    if not pin.sha256:
        if allow_missing_hash:
            return computed
        raise PinMismatchError(
            f"{pin.source}@{pin.revision} has no recorded sha256. Record "
            f"{computed} in the pin before evaluating on it."
        )
    if computed != pin.sha256:
        raise PinMismatchError(
            f"{path} hashes to {computed} but the pin for "
            f"{pin.source}@{pin.revision} declares {pin.sha256}. The evaluated "
            "bytes are not the pinned bytes; refusing to proceed."
        )
    return computed


def split_items(
    items: Sequence[BenchmarkItem],
    salt: str,
    fractions: Sequence[float],
) -> Dict[str, List[BenchmarkItem]]:
    """Assign items to the four splits section 4 requires.

    Delegates to ``dataset.hash_split`` so that benchmark items and candidate
    records land in the *same* split for the same ``item_id``.  Any other
    splitter would leak calibration items into the test set as soon as the two
    layers disagreed.

    Args:
        items: Items to split.
        salt: Split salt, frozen in ``PREREGISTRATION.md`` section 9.
        fractions: Four fractions for train/calibration/validation/test.

    Returns:
        Mapping from split name to items.
    """
    from dataset import SPLIT_NAMES, hash_split

    buckets: Dict[str, List[BenchmarkItem]] = {name: [] for name in SPLIT_NAMES}
    for item in items:
        buckets[hash_split(item.item_id, salt, fractions)].append(item)
    return buckets


class BenchmarkAdapter(ABC):
    """One evaluation domain from the specification section 5 table.

    Subclasses must declare a primary metric, secondary metrics, and *harm*
    metrics.  The last is not optional: limitation L13 requires that a role's
    target benefit be weighed against the regressions it causes, and an adapter
    with no harm outcome makes that impossible for its domain.
    """

    #: Benchmark identifier; must appear in ``config.BENCHMARKS``.
    name: str = ""

    #: Frozen dataset identification.
    pin: DatasetPin

    #: Name of the dataset-native primary outcome.
    primary_metric: str = ""

    #: Secondary outcomes from the section 5 table.
    secondary_metrics: Tuple[str, ...] = ()

    #: Harm outcomes from the section 5 table.  Empty is a bug, not a default.
    harm_metrics: Tuple[str, ...] = ()

    #: Whether the primary outcome requires human judgement (section 5 /
    #: limitation L16).  When true, the judging pipeline must supply labels and
    #: an LLM judge may not be used as the primary outcome without a human audit
    #: sample quantifying its error rate.
    requires_human_judgement: bool = False

    #: Smallest effect size of interest, from ``PREREGISTRATION.md`` section 2.
    #: Compared against measured clustering error to decide whether the
    #: benchmark can support a conclusion at all.
    sesoi: float = 0.03

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only enforce on concrete adapters, so intermediate bases stay usable.
        if getattr(cls, "__abstractmethods__", None):
            return
        if not cls.name:
            raise TypeError(f"{cls.__name__} must declare a benchmark name")
        if not cls.primary_metric:
            raise TypeError(f"{cls.__name__} must declare a primary_metric")
        if not cls.harm_metrics:
            raise TypeError(
                f"{cls.__name__} declares no harm_metrics; limitation L13 "
                "requires every domain to carry an outcome that can get worse"
            )

    @abstractmethod
    def load(self, path: str, *, verify: bool = True) -> List[BenchmarkItem]:
        """Materialise items from a local file.

        Args:
            path: Local copy of the pinned dataset.
            verify: Check the file against :attr:`pin`.

        Returns:
            All items, unsplit.
        """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Apply this benchmark's official answer normalisation.

        Args:
            text: Raw answer text.

        Returns:
            Canonical form.
        """

    @abstractmethod
    def score(
        self,
        items: Sequence[BenchmarkItem],
        predictions: Sequence[str],
        *,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score predictions, returning primary, secondary, and harm outcomes.

        Args:
            items: Items in the order they were evaluated.
            predictions: Predicted answer texts, aligned with ``items``.
            extras: Per-benchmark side channels -- judge labels, token counts,
                challenge-round responses.

        Returns:
            Mapping containing at least ``primary``, plus every name declared in
            :attr:`secondary_metrics` and :attr:`harm_metrics`.
        """

    def load_splits(
        self,
        path: str,
        salt: str = "cca-synth-v1",
        fractions: Sequence[float] = (0.50, 0.15, 0.15, 0.20),
        *,
        verify: bool = True,
    ) -> Dict[str, List[BenchmarkItem]]:
        """Load and split in one step.

        Args:
            path: Local copy of the pinned dataset.
            salt: Split salt, frozen in the pre-registration.
            fractions: Four split fractions.
            verify: Check the file against :attr:`pin`.

        Returns:
            Mapping from split name to items.
        """
        return split_items(self.load(path, verify=verify), salt, fractions)

    def describe(self) -> Dict[str, Any]:
        """Manifest entry for this benchmark.

        Returns:
            JSON-serialisable description including the pin, the outcomes, and
            whether human judgement is required.
        """
        return {
            "benchmark": self.name,
            "primary_metric": self.primary_metric,
            "secondary_metrics": list(self.secondary_metrics),
            "harm_metrics": list(self.harm_metrics),
            "requires_human_judgement": self.requires_human_judgement,
            "sesoi": self.sesoi,
            "pin": self.pin.as_dict(),
        }


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file, reporting the offending line on a parse error.

    Args:
        path: File to read.

    Returns:
        One dict per non-empty line.

    Raises:
        json.JSONDecodeError: With the line number attached.
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise json.JSONDecodeError(
                    f"{error.msg} (in {path} at line {line_number})",
                    error.doc,
                    error.pos,
                ) from error
    return out
