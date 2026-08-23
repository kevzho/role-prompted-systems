"""Benchmark adapters for the specification section 5 outcome table.

One adapter per domain. Each declares a pinned dataset version, a licence, the
dataset-native primary metric, the official answer normalisation, and -- not
optionally -- the secondary and *harm* outcomes named in the section 5 table.

The registry below is the single place that maps a ``config.BENCHMARKS``
identifier onto an adapter, so a benchmark cannot be evaluated without one.

Nothing here downloads data. Adapters read a local materialised copy and verify
its hash against the declared pin, which is what makes limitation L14's "no
prompt, model, or decoding versions are frozen" answerable for the data layer:
the artifact records which bytes were evaluated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from .advconsistency import AdvConsistencyAdapter
from .base import (
    BenchmarkAdapter,
    BenchmarkItem,
    DatasetPin,
    PinMismatchError,
    ToolAccess,
    file_sha256,
    read_jsonl,
    split_items,
    verify_pin,
)
from .creative_divergence import RUBRIC, RUBRIC_DIMENSIONS, CreativeDivergenceAdapter
from .halueval import HALUEVAL_TASKS, HaluEvalAdapter
from .math_aime import MathAimeAdapter
from .sycophancy import SycophancyAdapter
from .temporal_qa import TemporalQAAdapter
from .truthfulqa import TruthfulQAAdapter

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkItem",
    "DatasetPin",
    "PinMismatchError",
    "ToolAccess",
    "file_sha256",
    "read_jsonl",
    "split_items",
    "verify_pin",
    "AdvConsistencyAdapter",
    "CreativeDivergenceAdapter",
    "HaluEvalAdapter",
    "MathAimeAdapter",
    "SycophancyAdapter",
    "TemporalQAAdapter",
    "TruthfulQAAdapter",
    "HALUEVAL_TASKS",
    "RUBRIC",
    "RUBRIC_DIMENSIONS",
    "ADAPTERS",
    "get_adapter",
    "describe_all",
    "sesoi_by_benchmark",
]

#: Benchmark identifier to adapter class.  Keys must match ``config.BENCHMARKS``
#: exactly; :func:`_check_registry_covers_config` enforces that at import time.
ADAPTERS: Mapping[str, type] = {
    "truthfulqa": TruthfulQAAdapter,
    "halueval": HaluEvalAdapter,
    "math_aime": MathAimeAdapter,
    "advconsistency": AdvConsistencyAdapter,
    "sycophancy": SycophancyAdapter,
    "temporal_qa": TemporalQAAdapter,
    "creative_divergence": CreativeDivergenceAdapter,
}


def _check_registry_covers_config() -> None:
    """Fail at import if the registry and ``config.BENCHMARKS`` disagree.

    A benchmark named in the config but missing from the registry would be
    silently unevaluable; one in the registry but not the config would never be
    reached. Both are configuration errors that should surface immediately.

    Raises:
        RuntimeError: On any mismatch.
    """
    from config import BENCHMARKS

    configured = set(BENCHMARKS)
    registered = set(ADAPTERS)
    if configured != registered:
        missing = sorted(configured - registered)
        extra = sorted(registered - configured)
        raise RuntimeError(
            "benchmark registry does not match config.BENCHMARKS; "
            f"missing adapters: {missing}; unknown adapters: {extra}"
        )


_check_registry_covers_config()


def get_adapter(name: str, **kwargs: Any) -> BenchmarkAdapter:
    """Instantiate the adapter for one benchmark.

    Args:
        name: Benchmark identifier from ``config.BENCHMARKS``.
        **kwargs: Adapter-specific options, e.g. HaluEval's ``task``.

    Returns:
        A constructed adapter.

    Raises:
        KeyError: On an unknown benchmark.
    """
    if name not in ADAPTERS:
        raise KeyError(f"unknown benchmark {name!r}; expected one of {sorted(ADAPTERS)}")
    return ADAPTERS[name](**kwargs)  # type: ignore[abstract]


def describe_all(**per_benchmark_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Manifest section describing every benchmark.

    Args:
        **per_benchmark_kwargs: Optional per-benchmark constructor arguments,
            keyed by benchmark name.

    Returns:
        Mapping from benchmark name to its description, plus a roll-up of which
        benchmarks require human judgement and which are still unbuilt.
    """
    described: Dict[str, Any] = {}
    for name in ADAPTERS:
        kwargs = dict(per_benchmark_kwargs.get(name, {}))
        described[name] = get_adapter(name, **kwargs).describe()
    return {
        "benchmarks": described,
        "requiring_human_judgement": sorted(
            n for n, d in described.items() if d["requires_human_judgement"]
        ),
        "unbuilt": sorted(
            n for n, d in described.items() if d["pin"]["revision"] == "UNBUILT"
        ),
        "unpinned_hashes": sorted(
            n for n, d in described.items() if not d["pin"]["sha256"]
        ),
        "note": (
            "A benchmark listed under 'unpinned_hashes' has no recorded content "
            "hash and will be refused by verify_pin. Record the hash of the "
            "materialised file before evaluating on it."
        ),
    }


def sesoi_by_benchmark() -> Dict[str, float]:
    """Smallest effect size of interest per benchmark.

    Feeds ``normalize.clustering_risk_report``, which blocks conclusions on any
    benchmark whose measured answer-clustering error exceeds its SESOI.

    Returns:
        Mapping from benchmark name to SESOI.
    """
    return {name: float(ADAPTERS[name].sesoi) for name in ADAPTERS}
