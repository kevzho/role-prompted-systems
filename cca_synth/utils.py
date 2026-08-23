"""Infrastructure for ``cca_synth``: seeding, distribution, metrics, statistics.

Contents, in dependency order:

* **Reproducibility** -- :func:`seed_everything` seeds Python, NumPy, and every
  torch generator (CPU, all CUDA devices, MPS) in one call, plus per-worker and
  per-rank derivation so that dataloader shuffling is reproducible without being
  *identical* across workers.
* **Device and distribution** -- :func:`resolve_device` and
  :func:`setup_distributed` give a single code path across CUDA, MPS, and CPU,
  single-process or ``torchrun``.
* **Metrics** -- accuracy, expected calibration error, Brier score, selective
  accuracy, and the full risk-coverage curve with AURC.
* **Statistics** -- bootstrap intervals, paired bootstrap contrasts,
  Benjamini-Hochberg FDR control, and pairwise error correlation, matching
  ``RESEARCH_SPECIFICATION.md`` section 8.
* **Governance** -- :class:`AuditRecordWriter` enforces the per-field allowlist
  and hidden-reasoning denylist from ``ETHICS_AND_DATA_GOVERNANCE.md``.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import math
import os
import platform
import random
import re
import socket
import sys
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.distributed as dist

from config import (
    AUDIT_FIELD_ALLOWLIST,
    AUDIT_FIELD_DENY_PATTERNS,
    CLINICAL_TRACEABILITY,
    COALITION_METRICS,
    Config,
    config_hash,
    config_to_dict,
)

__all__ = [
    "seed_everything",
    "seed_worker",
    "torch_generator",
    "resolve_device",
    "amp_settings",
    "DistState",
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "barrier",
    "all_gather_object",
    "reduce_scalar",
    "get_logger",
    "Timer",
    "MeanTracker",
    "accuracy",
    "expected_calibration_error",
    "brier_score",
    "selective_accuracy",
    "coalition_item_scores",
    "risk_coverage_curve",
    "area_under_risk_coverage",
    "fit_selective_threshold",
    "bootstrap_ci",
    "paired_bootstrap_diff",
    "benjamini_hochberg",
    "pairwise_error_correlation",
    "coalition_to_index",
    "index_to_coalition",
    "shapley_from_values",
    "AuditRecordWriter",
    "write_json",
    "append_jsonl",
    "save_checkpoint",
    "load_checkpoint",
    "build_manifest",
    "format_table",
    "human_bytes",
]


LOGGER_NAME = "cca_synth"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    rank: int = 0,
    rank_offset: bool = False,
) -> int:
    """Seed every RNG this project can reach.

    Seeds ``PYTHONHASHSEED``, :mod:`random`, :mod:`numpy`, torch CPU, all CUDA
    devices, and MPS.  Model initialisation must be identical across ranks, so
    ``rank_offset`` defaults to ``False``; pass ``True`` only for stream-local
    randomness such as augmentation or perturbation sampling.

    Args:
        seed: Base seed.
        deterministic: Request deterministic cuDNN kernels and, where the
            installed torch supports it, deterministic algorithms globally.
            Non-deterministic ops then raise instead of silently varying.
        rank: Process rank, used when ``rank_offset`` is set.
        rank_offset: Add ``rank`` to the seed to decorrelate per-rank streams.

    Returns:
        The effective seed actually applied.
    """
    effective = int(seed) + (int(rank) if rank_offset else 0)
    os.environ["PYTHONHASHSEED"] = str(effective)
    random.seed(effective)
    np.random.seed(effective % (2**32 - 1))
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        with contextlib.suppress(Exception):
            torch.mps.manual_seed(effective)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
    return effective


def seed_worker(worker_id: int) -> None:  # pragma: no cover - runs in subprocess
    """``worker_init_fn`` that derives a distinct, reproducible per-worker seed.

    Torch already assigns each worker ``base_seed + worker_id``; this propagates
    that value into NumPy and :mod:`random`, which torch does not seed.

    Args:
        worker_id: Index of the dataloader worker.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int) -> torch.Generator:
    """Return a CPU :class:`torch.Generator` seeded with ``seed``.

    Args:
        seed: Seed value.

    Returns:
        A generator suitable for ``DataLoader(generator=...)``.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


# ---------------------------------------------------------------------------
# Device and distribution
# ---------------------------------------------------------------------------


def resolve_device(preference: str = "auto", local_rank: int = 0) -> torch.device:
    """Choose an execution device.

    Args:
        preference: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
        local_rank: CUDA ordinal to bind when several GPUs are visible.

    Returns:
        The selected :class:`torch.device`.

    Raises:
        RuntimeError: If an explicit preference is unavailable.
    """
    preference = preference.lower()
    has_cuda = torch.cuda.is_available()
    has_mps = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )

    if preference == "auto":
        if has_cuda:
            return torch.device(f"cuda:{local_rank}")
        if has_mps:
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda":
        if not has_cuda:
            raise RuntimeError("device='cuda' requested but CUDA is unavailable")
        return torch.device(f"cuda:{local_rank}")
    if preference == "mps":
        if not has_mps:
            raise RuntimeError("device='mps' requested but MPS is unavailable")
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"unknown device preference '{preference}'")


def amp_settings(
    device: torch.device, enabled: bool, dtype_name: str
) -> Tuple[bool, torch.dtype, bool]:
    """Decide the autocast configuration for ``device``.

    CUDA gets bfloat16 when the hardware supports it, else float16 with a
    :class:`torch.amp.GradScaler`.  MPS autocast is disabled by default: its
    reduced-precision reductions are not consistent enough for the calibration
    metrics this project reports, and the model is far too small to benefit.
    CPU autocast is likewise disabled.

    Args:
        device: Target device.
        enabled: Whether the user requested mixed precision.
        dtype_name: ``"bfloat16"`` or ``"float16"``.

    Returns:
        Tuple ``(use_amp, dtype, needs_grad_scaler)``.
    """
    requested = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    if not enabled or device.type != "cuda":
        return False, torch.float32, False
    if requested is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return True, torch.float16, True
    return True, requested, requested is torch.float16


@dataclass
class DistState:
    """Resolved distributed-execution state.

    Attributes:
        enabled: Whether a process group is initialised.
        rank: Global rank.
        local_rank: Rank within the node.
        world_size: Total processes.
        device: Device bound to this process.
        backend: Process-group backend in use, or ``""``.
    """

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str

    @property
    def is_main(self) -> bool:
        """True on rank 0, or in any single-process run."""
        return self.rank == 0


_DIST_STATE: Optional[DistState] = None


def setup_distributed(cfg: Config) -> DistState:
    """Initialise the process group when launched under ``torchrun``.

    Reads ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` and picks NCCL for CUDA, Gloo
    otherwise.  A single-process run returns an ``enabled=False`` state and
    touches no collective, so every downstream call site works unmodified.

    Args:
        cfg: Root configuration (``cfg.dist`` is consulted).

    Returns:
        The resolved :class:`DistState`, also cached module-globally.
    """
    global _DIST_STATE

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = resolve_device(cfg.dist.device, local_rank=local_rank)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        if cfg.dist.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    if world_size <= 1 or dist.is_initialized():
        backend = dist.get_backend() if dist.is_initialized() else ""
        _DIST_STATE = DistState(
            enabled=dist.is_initialized(),
            rank=rank,
            local_rank=local_rank,
            world_size=world_size if dist.is_initialized() else 1,
            device=device,
            backend=backend,
        )
        return _DIST_STATE

    if cfg.dist.backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"
    else:
        backend = cfg.dist.backend

    dist.init_process_group(
        backend=backend,
        timeout=_dt.timedelta(minutes=cfg.dist.timeout_minutes),
        world_size=world_size,
        rank=rank,
    )
    _DIST_STATE = DistState(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )
    return _DIST_STATE


def cleanup_distributed() -> None:
    """Destroy the process group if one is active."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    """True when this process should write artifacts to disk."""
    if _DIST_STATE is not None:
        return _DIST_STATE.is_main
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier() -> None:
    """Synchronise all ranks; a no-op outside distributed runs."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_gather_object(payload: Any) -> List[Any]:
    """Gather an arbitrary picklable object from every rank.

    Used for per-item evaluation records, which are small and irregular; a
    tensor collective would require padding and a second length exchange.

    Args:
        payload: Any picklable object.

    Returns:
        A list of length ``world_size`` ordered by rank; ``[payload]`` outside
        distributed runs.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [payload]
    gathered: List[Any] = [None] * dist.get_world_size()  # type: ignore[list-item]
    dist.all_gather_object(gathered, payload)
    return gathered


def reduce_scalar(value: float, device: torch.device, op: str = "mean") -> float:
    """All-reduce a Python scalar across ranks.

    Args:
        value: Local scalar.
        device: Device to stage the reduction tensor on.
        op: ``"mean"`` or ``"sum"``.

    Returns:
        The reduced scalar, or ``value`` outside distributed runs.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    staging = "cpu" if device.type == "mps" else device
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=staging)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if op == "mean":
        tensor /= dist.get_world_size()
    return float(tensor.item())


# ---------------------------------------------------------------------------
# Logging and timing
# ---------------------------------------------------------------------------


def get_logger(name: str = LOGGER_NAME, log_file: Optional[str] = None) -> logging.Logger:
    """Return a rank-aware logger.

    Non-main ranks are raised to ``WARNING`` so that a multi-GPU run does not
    interleave six copies of every line.

    Args:
        name: Logger name.
        log_file: Optional path for an additional file handler (main rank only).

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if is_main_process() else logging.WARNING)
    logger.propagate = False

    if log_file and is_main_process():
        already = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == os.path.abspath(log_file)
            for h in logger.handlers
        )
        if not already:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
            )
            logger.addHandler(file_handler)
    return logger


class Timer:
    """Context manager measuring wall-clock seconds.

    Example:
        >>> with Timer() as t:
        ...     _ = sum(range(10))
        >>> t.elapsed >= 0.0
        True
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.start


class MeanTracker:
    """Streaming mean of named scalars, with distributed reduction on demand."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, float] = {}

    def update(self, values: Mapping[str, float], weight: float = 1.0) -> None:
        """Accumulate a batch of named scalars.

        Args:
            values: Mapping from metric name to scalar value.
            weight: Sample weight, normally the batch size.
        """
        for key, value in values.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            self._sums[key] = self._sums.get(key, 0.0) + float(value) * weight
            self._counts[key] = self._counts.get(key, 0.0) + weight

    def compute(self, device: Optional[torch.device] = None) -> Dict[str, float]:
        """Return the current means, reduced across ranks when ``device`` is given.

        Args:
            device: Device to stage collectives on; ``None`` skips reduction.

        Returns:
            Mapping from metric name to mean value.
        """
        out: Dict[str, float] = {}
        for key in sorted(self._sums):
            total, count = self._sums[key], self._counts[key]
            if device is not None:
                total = reduce_scalar(total, device, op="sum")
                count = reduce_scalar(count, device, op="sum")
            out[key] = total / count if count > 0 else float("nan")
        return out

    def reset(self) -> None:
        """Clear all accumulators."""
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Plain top-1 accuracy over items with a known label.

    Args:
        predictions: Integer predictions, shape ``(N,)``.
        labels: Integer gold labels, shape ``(N,)``; ``-1`` marks unknown.

    Returns:
        Accuracy over labelled items, or ``nan`` when none are labelled.
    """
    mask = labels >= 0
    if not mask.any():
        return float("nan")
    return float((predictions[mask] == labels[mask]).mean())


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
    binning: str = "equal_mass",
) -> float:
    """Expected calibration error.

    Equal-mass binning is the default because equal-width bins are dominated by
    a single saturated high-confidence bin for well-trained models, which
    understates miscalibration.

    Args:
        confidences: Predicted probability of the selected answer, ``(N,)``.
        correct: Binary correctness indicators, ``(N,)``.
        n_bins: Number of bins.
        binning: ``"equal_mass"`` or ``"equal_width"``.

    Returns:
        ECE in ``[0, 1]``, or ``nan`` for an empty input.
    """
    confidences = np.asarray(confidences, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    n = confidences.size
    if n == 0:
        return float("nan")

    if binning == "equal_mass":
        order = np.argsort(confidences, kind="stable")
        chunks = np.array_split(order, min(n_bins, n))
        error = 0.0
        for chunk in chunks:
            if chunk.size == 0:
                continue
            gap = abs(correct[chunk].mean() - confidences[chunk].mean())
            error += (chunk.size / n) * gap
        return float(error)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            in_bin |= confidences == 0.0
        if not in_bin.any():
            continue
        gap = abs(correct[in_bin].mean() - confidences[in_bin].mean())
        error += (in_bin.sum() / n) * gap
    return float(error)


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score against one-hot targets.

    Args:
        probabilities: Row-stochastic array, shape ``(N, K)``.
        labels: Integer labels, shape ``(N,)``; ``-1`` marks unknown.

    Returns:
        Mean squared error over the probability simplex, lower is better.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    mask = labels >= 0
    if not mask.any():
        return float("nan")
    probs = probabilities[mask]
    onehot = np.zeros_like(probs)
    onehot[np.arange(probs.shape[0]), labels[mask]] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def selective_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    abstained: np.ndarray,
    abstain_credit: float = 0.0,
) -> Tuple[float, float]:
    """Item score under an abstention policy, plus coverage.

    The specification requires selective accuracy *and* coverage to be reported
    together: an abstain-always policy trivially maximises accuracy-on-answered.

    Args:
        predictions: Integer predictions, ``(N,)``.
        labels: Integer gold labels, ``(N,)``.
        abstained: Boolean abstention decisions, ``(N,)``.
        abstain_credit: Score awarded for abstaining, predeclared per section 8.

    Returns:
        Tuple ``(mean_item_score, coverage)``.
    """
    scores = item_scores(predictions, labels, abstained, abstain_credit)
    coverage = float(1.0 - np.asarray(abstained, dtype=np.float64).mean())
    return float(np.nanmean(scores)), coverage


def item_scores(
    predictions: np.ndarray,
    labels: np.ndarray,
    abstained: np.ndarray,
    abstain_credit: float = 0.0,
) -> np.ndarray:
    """Per-item scores forming the characteristic function ``v(.)``.

    Keeping this at item granularity is what allows section 8's paired designs:
    the same items and seeds are reused across coalitions, and every interval is
    computed by resampling items rather than re-running the model.

    Args:
        predictions: Integer predictions, ``(N,)``.
        labels: Integer gold labels, ``(N,)``; ``-1`` items score ``nan``.
        abstained: Boolean abstention decisions, ``(N,)``.
        abstain_credit: Score for an abstention.

    Returns:
        Float array ``(N,)`` of item scores.
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    abstained = np.asarray(abstained, dtype=bool)
    scores = np.where(predictions == labels, 1.0, 0.0)
    scores = np.where(abstained, float(abstain_credit), scores)
    return np.where(labels >= 0, scores, np.nan)


def coalition_item_scores(
    metric: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    abstained: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    abstain_credit: float = 0.0,
) -> np.ndarray:
    """Per-item characteristic-function values under a named metric.

    ``selective_accuracy`` is the pre-registered default and scores an abstention
    at ``abstain_credit``.  ``accuracy`` ignores the abstention decision and
    scores the argmax, which isolates how much of a coalition's value comes from
    answering well versus from declining to answer.  ``negative_brier`` is a
    proper scoring rule, rescaled from the multiclass Brier score's ``[0, 2]``
    onto ``[0, 1]`` so that it composes with the others.

    Args:
        metric: One of :data:`COALITION_METRICS`.
        predictions: Integer predictions, ``(N,)``.
        labels: Integer gold labels, ``(N,)``; ``-1`` items score ``nan``.
        abstained: Boolean abstention decisions, ``(N,)``.
        probabilities: ``(N, K)`` predictive distribution; required for
            ``negative_brier``.
        abstain_credit: Score awarded for an abstention.

    Returns:
        Float array ``(N,)`` of item scores.

    Raises:
        ValueError: On an unknown metric, or a missing ``probabilities``
            argument for a metric that needs one.
    """
    if metric not in COALITION_METRICS:
        raise ValueError(
            f"unknown coalition_metric {metric!r}; expected one of {COALITION_METRICS}"
        )
    labels = np.asarray(labels)
    valid = labels >= 0

    if metric == "selective_accuracy":
        return item_scores(predictions, labels, abstained, abstain_credit)

    if metric == "accuracy":
        scores = np.where(np.asarray(predictions) == labels, 1.0, 0.0)
        return np.where(valid, scores, np.nan)

    if probabilities is None:
        raise ValueError("negative_brier requires the predictive distribution")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    onehot = np.zeros_like(probabilities)
    safe = np.where(valid, labels, 0)
    onehot[np.arange(probabilities.shape[0]), safe] = 1.0
    brier = ((probabilities - onehot) ** 2).sum(axis=1)
    scores = 1.0 - brier / 2.0
    scores = np.where(np.asarray(abstained, dtype=bool), float(abstain_credit), scores)
    return np.where(valid, scores, np.nan)


def risk_coverage_curve(
    confidences: np.ndarray, correct: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve obtained by sweeping the confidence threshold.

    Args:
        confidences: Selection scores, ``(N,)``; higher means more likely kept.
        correct: Binary correctness indicators, ``(N,)``.

    Returns:
        Tuple ``(coverage, risk)`` of shape ``(N,)`` each, coverage ascending.
    """
    confidences = np.asarray(confidences, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    n = confidences.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(-confidences, kind="stable")
    sorted_correct = correct[order]
    cumulative_errors = np.cumsum(1.0 - sorted_correct)
    counts = np.arange(1, n + 1, dtype=np.float64)
    return counts / n, cumulative_errors / counts


def area_under_risk_coverage(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better).

    Args:
        confidences: Selection scores, ``(N,)``.
        correct: Binary correctness indicators, ``(N,)``.

    Returns:
        AURC, or ``nan`` for empty input.
    """
    coverage, risk = risk_coverage_curve(confidences, correct)
    if coverage.size == 0:
        return float("nan")
    return float(np.trapezoid(risk, coverage) / max(coverage[-1], 1e-12))


def fit_selective_threshold(
    confidences: np.ndarray, target_coverage: float
) -> float:
    """Pick the confidence threshold achieving ``target_coverage``.

    Fitted on the calibration split only, never on validation or test.

    Args:
        confidences: Selection scores from the calibration split, ``(N,)``.
        target_coverage: Desired fraction of answered items in ``(0, 1]``.

    Returns:
        The threshold; items with confidence below it abstain.
    """
    confidences = np.asarray(confidences, dtype=np.float64).ravel()
    if confidences.size == 0:
        return 0.0
    target_coverage = float(np.clip(target_coverage, 1e-6, 1.0))
    quantile = float(np.clip(1.0 - target_coverage, 0.0, 1.0))
    return float(np.quantile(confidences, quantile))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
    statistic: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict[str, float]:
    """Percentile bootstrap interval for a statistic of item-level values.

    Args:
        values: Item-level observations, ``(N,)``; ``nan`` entries are dropped.
        n_bootstrap: Number of resamples.
        ci_level: Two-sided confidence level.
        seed: RNG seed for the resampler.
        statistic: Callable applied to each resample; defaults to the mean.

    Returns:
        Mapping with ``point``, ``lo``, ``hi``, ``se``, and ``n``.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[~np.isnan(values)]
    n = values.size
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "n": 0.0}
    func = statistic or (lambda arr: float(np.mean(arr)))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    draws = np.array([func(values[idx]) for idx in indices], dtype=np.float64)
    alpha = (1.0 - ci_level) / 2.0
    return {
        "point": float(func(values)),
        "lo": float(np.quantile(draws, alpha)),
        "hi": float(np.quantile(draws, 1.0 - alpha)),
        "se": float(draws.std(ddof=1)) if n_bootstrap > 1 else float("nan"),
        "n": float(n),
    }


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired item-level bootstrap contrast ``mean(a) - mean(b)``.

    Section 8 requires paired item-level methods for the pre-registered primary
    contrasts.  Both arms must be evaluated on the same items in the same order.

    Args:
        a: Item scores for the treatment arm, ``(N,)``.
        b: Item scores for the control arm, ``(N,)``.
        n_bootstrap: Number of resamples.
        ci_level: Two-sided confidence level.
        seed: RNG seed.

    Returns:
        Mapping with ``diff``, ``lo``, ``hi``, ``p_two_sided``, ``p_superior``,
        and ``n``.

    Raises:
        ValueError: If the two arms have different lengths.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    n = a.size
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_two_sided": float("nan"), "p_superior": float("nan"), "n": 0.0}

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    draws = a[indices].mean(axis=1) - b[indices].mean(axis=1)
    observed = float(a.mean() - b.mean())
    alpha = (1.0 - ci_level) / 2.0
    p_superior = float((draws > 0).mean())
    # Two-sided achieved significance level from the bootstrap distribution,
    # centred at zero: the fraction of resamples at least as extreme as 0.
    centred = draws - observed
    p_two = float((np.abs(centred) >= abs(observed)).mean())
    return {
        "diff": observed,
        "lo": float(np.quantile(draws, alpha)),
        "hi": float(np.quantile(draws, 1.0 - alpha)),
        "p_two_sided": p_two,
        "p_superior": p_superior,
        "n": float(n),
    }


def benjamini_hochberg(
    p_values: Sequence[float], alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up FDR control.

    Applied to the agent-by-benchmark secondary family, as section 8 requires.

    Args:
        p_values: Raw p-values.
        alpha: Target false discovery rate.

    Returns:
        Tuple ``(rejected, q_values)`` of shape ``(m,)``.
    """
    p = np.asarray(list(p_values), dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    scaled = ranked * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q <= alpha, q


def pairwise_error_correlation(error_matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation between agents' item-level error vectors.

    Limitation L7 warns that six instances of one base model may share failure
    modes; this matrix is the direct measurement.  Constant rows (an agent that
    never errs, or always does) yield ``nan`` rather than a spurious zero.

    Args:
        error_matrix: Binary error indicators, shape ``(N, R)``.

    Returns:
        Correlation matrix, shape ``(R, R)``.
    """
    errors = np.asarray(error_matrix, dtype=np.float64)
    if errors.ndim != 2:
        raise ValueError(f"expected a 2-D (N, R) matrix, got shape {errors.shape}")
    n_roles = errors.shape[1]
    out = np.full((n_roles, n_roles), np.nan, dtype=np.float64)
    stds = errors.std(axis=0)
    for i in range(n_roles):
        for j in range(n_roles):
            if stds[i] < 1e-12 or stds[j] < 1e-12:
                continue
            out[i, j] = float(np.corrcoef(errors[:, i], errors[:, j])[0, 1])
    return out


def coalition_to_index(coalition: Sequence[int]) -> int:
    """Encode a binary coalition mask as an integer bitset.

    Args:
        coalition: Sequence of 0/1 flags, one per role.

    Returns:
        Integer in ``[0, 2**R)`` with bit ``r`` set when role ``r`` is present.
    """
    index = 0
    for role_index, active in enumerate(coalition):
        if active:
            index |= 1 << role_index
    return index


def index_to_coalition(index: int, n_roles: int) -> Tuple[int, ...]:
    """Decode a bitset back into a coalition mask.

    Args:
        index: Integer bitset.
        n_roles: Number of roles.

    Returns:
        Tuple of 0/1 flags of length ``n_roles``.
    """
    return tuple((index >> r) & 1 for r in range(n_roles))


def shapley_from_values(values: np.ndarray, n_roles: int) -> np.ndarray:
    """Exact Shapley values from a full table of coalition values.

    Implements the standard weighting

    ``phi_i = sum_{S subset A\\{i}} |S|!(|A|-|S|-1)!/|A|! [v(S+i) - v(S)]``

    exhaustively; with six roles the 64-entry table is trivially enumerable, so
    no sampling approximation is needed.

    Args:
        values: Coalition values indexed by bitset, shape ``(2**n_roles,)``.
        n_roles: Number of roles.

    Returns:
        Shapley values, shape ``(n_roles,)``.

    Raises:
        ValueError: If ``values`` has the wrong length.
    """
    values = np.asarray(values, dtype=np.float64)
    expected = 2**n_roles
    if values.shape[0] != expected:
        raise ValueError(f"expected {expected} coalition values, got {values.shape[0]}")

    factorial = [math.factorial(k) for k in range(n_roles + 1)]
    phi = np.zeros(n_roles, dtype=np.float64)
    for role in range(n_roles):
        bit = 1 << role
        for subset in range(expected):
            if subset & bit:
                continue
            size = bin(subset).count("1")
            weight = factorial[size] * factorial[n_roles - size - 1] / factorial[n_roles]
            phi[role] += weight * (values[subset | bit] - values[subset])
    return phi


# ---------------------------------------------------------------------------
# Governance: the audit trail
# ---------------------------------------------------------------------------


class AuditRecordWriter:
    """JSONL writer enforcing the ethics addendum's logging rules.

    Three rules are enforced, and all three *raise* rather than redact, because
    a silently dropped field hides a bug at the call site:

    1. every key must appear in ``AUDIT_FIELD_ALLOWLIST``;
    2. no key may contain a substring from ``AUDIT_FIELD_DENY_PATTERNS``, which
       covers chain-of-thought, prompts, and credentials;
    3. no *value* may be one of the clinical identifiers in
       ``CLINICAL_TRACEABILITY``, keeping the artifact function-first.

    The class is a context manager and is a no-op on non-main ranks.
    """

    def __init__(self, path: Optional[str], enabled: bool = True, max_records: int = 0):
        """Open the writer.

        Args:
            path: Destination JSONL path; ``None`` disables writing.
            enabled: Master switch.
            max_records: Stop after this many records; ``0`` means unlimited.
        """
        self.path = path
        self.enabled = bool(enabled and path and is_main_process())
        self.max_records = int(max_records)
        self.count = 0
        self._handle = None
        self._deny = re.compile("|".join(re.escape(p) for p in AUDIT_FIELD_DENY_PATTERNS))
        self._clinical = {v.lower() for v in CLINICAL_TRACEABILITY.values()}
        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)  # type: ignore[arg-type]
            self._handle = open(path, "w", encoding="utf-8")  # type: ignore[arg-type]

    def __enter__(self) -> "AuditRecordWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def validate(self, record: Mapping[str, Any], _path: str = "") -> None:
        """Recursively validate a record against all three rules.

        Args:
            record: Candidate audit record.
            _path: Internal dotted-path accumulator for error messages.

        Raises:
            ValueError: On a denied key, an unknown key, or a clinical value.
        """
        for key, value in record.items():
            dotted = f"{_path}{key}"
            lowered = str(key).lower()
            if self._deny.search(lowered):
                raise ValueError(
                    f"audit field '{dotted}' matches a prohibited pattern; the "
                    "ethics addendum forbids logging hidden reasoning, prompts, "
                    "or credentials"
                )
            if key not in AUDIT_FIELD_ALLOWLIST:
                raise ValueError(
                    f"audit field '{dotted}' is not in AUDIT_FIELD_ALLOWLIST; "
                    "add it deliberately in config.py after a governance review"
                )
            if isinstance(value, str) and value.lower() in self._clinical:
                raise ValueError(
                    f"audit field '{dotted}' carries the clinical identifier "
                    f"'{value}'; use the functional role label instead"
                )
            if isinstance(value, Mapping):
                self.validate(value, _path=f"{dotted}.")
            elif isinstance(value, (list, tuple)):
                for element in value:
                    if isinstance(element, Mapping):
                        self.validate(element, _path=f"{dotted}[].")
                    elif isinstance(element, str) and element.lower() in self._clinical:
                        raise ValueError(
                            f"audit field '{dotted}[]' carries the clinical "
                            f"identifier '{element}'; use the functional label"
                        )

    def write(self, record: Mapping[str, Any]) -> bool:
        """Validate and append one record.

        Args:
            record: The audit record.

        Returns:
            ``True`` if written, ``False`` if the writer is disabled or full.
        """
        if not self.enabled or self._handle is None:
            return False
        if self.max_records and self.count >= self.max_records:
            return False
        self.validate(record)
        self._handle.write(json.dumps(record, sort_keys=True, default=_json_default))
        self._handle.write("\n")
        self.count += 1
        return True

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None


def _json_default(obj: Any) -> Any:
    """Fallback encoder for NumPy and torch scalars.

    Args:
        obj: Object that :mod:`json` could not serialise.

    Returns:
        A JSON-safe primitive.

    Raises:
        TypeError: If the object is genuinely unserialisable.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if math.isnan(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def write_json(payload: Any, path: str) -> None:
    """Write ``payload`` as indented JSON (main rank only).

    Args:
        payload: JSON-serialisable object.
        path: Destination path; parent directories are created.
    """
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def append_jsonl(payload: Mapping[str, Any], path: str) -> None:
    """Append one JSON object as a line (main rank only).

    Args:
        payload: JSON-serialisable mapping.
        path: Destination JSONL path.
    """
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default))
        handle.write("\n")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    cfg: Config,
    *,
    epoch: int,
    step: int,
    metrics: Mapping[str, float],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    ema_state: Optional[Mapping[str, torch.Tensor]] = None,
    trainer_state: Optional[Mapping[str, Any]] = None,
    rng_state: Optional[Mapping[str, torch.Tensor]] = None,
) -> None:
    """Persist a checkpoint (main rank only).

    The config is stored as a JSON *string* rather than a nested object so that
    the file loads cleanly under ``torch.load(weights_only=True)``.  For the same
    reason ``trainer_state`` is JSON-encoded: it carries the loop bookkeeping a
    resume needs (best metric, early-stop counter, per-epoch history), none of
    which is a tensor.

    Args:
        path: Destination ``.pt`` path.
        model: Model, unwrapped from DDP if necessary.
        cfg: Configuration to embed.
        epoch: Epoch index.
        step: Global optimiser step.
        metrics: Validation metrics at save time.
        optimizer: Optional optimiser state.
        scheduler: Optional scheduler state.
        scaler: Optional AMP scaler state.
        ema_state: Optional EMA shadow weights.
        trainer_state: Optional JSON-serialisable training-loop bookkeeping.
        rng_state: Optional byte-tensor generator states, so that a resumed run
            samples the same coalitions the uninterrupted run would have.
    """
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    payload: Dict[str, Any] = {
        "model": {k: v.detach().cpu() for k, v in module.state_dict().items()},
        "config_json": json.dumps(config_to_dict(cfg), sort_keys=True),
        "config_hash": config_hash(cfg),
        "epoch": int(epoch),
        "step": int(step),
        "metrics": {k: float(v) for k, v in metrics.items()},
        # str(): torch.__version__ is a TorchVersion instance, which the
        # weights_only=True unpickler refuses to reconstruct.
        "torch_version": str(torch.__version__),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler"] = scaler.state_dict()
    if ema_state is not None:
        payload["ema"] = {k: v.detach().cpu() for k, v in ema_state.items()}
    if trainer_state is not None:
        payload["trainer_state_json"] = json.dumps(trainer_state, sort_keys=True, default=float)
    if rng_state is not None:
        payload["rng_state"] = {k: v.detach().cpu() for k, v in rng_state.items()}
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Source ``.pt`` path.
        map_location: Torch map location.

    Returns:
        The checkpoint dictionary, with ``config`` re-parsed from JSON.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if "config_json" in payload:
        payload["config"] = json.loads(payload["config_json"])
    if "trainer_state_json" in payload:
        payload["trainer_state"] = json.loads(payload["trainer_state_json"])
    return payload


def build_manifest(cfg: Config, dist_state: DistState, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the execution manifest demanded by specification section 4.

    Records software versions, hardware, seeds, device, world size, the config
    hash, data provenance, and the mandatory disclaimer, so that a run can be
    reconstructed from the artifact alone.

    Args:
        cfg: Root configuration.
        dist_state: Resolved distributed state.
        extra: Additional key-value pairs to merge in.

    Returns:
        A JSON-serialisable manifest.
    """
    from config import DISCLAIMER  # local import keeps the module import graph flat

    manifest: Dict[str, Any] = {
        "run_id": cfg.run_id,
        "config_hash": config_hash(cfg),
        "config": config_to_dict(cfg),
        "disclaimer": DISCLAIMER,
        "data_provenance": cfg.data.provenance,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seed": cfg.train.seed,
        "world_size": dist_state.world_size,
        "device": str(dist_state.device),
        "backend": dist_state.backend,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "mps_available": bool(
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ),
        },
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def format_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Optional[Sequence[str]] = None,
    float_format: str = "{:.4f}",
) -> str:
    """Render rows as a fixed-width ASCII table for console reports.

    Args:
        rows: Sequence of mappings sharing a key set.
        columns: Column order; inferred from the first row when omitted.
        float_format: Format string applied to floats.

    Returns:
        The rendered table, or ``"(no rows)"``.
    """
    if not rows:
        return "(no rows)"
    cols = list(columns) if columns else list(rows[0].keys())

    def render(value: Any) -> str:
        if isinstance(value, float):
            return "nan" if math.isnan(value) else float_format.format(value)
        return str(value)

    widths = {c: len(c) for c in cols}
    rendered: List[Dict[str, str]] = []
    for row in rows:
        cells = {c: render(row.get(c, "")) for c in cols}
        for c in cols:
            widths[c] = max(widths[c], len(cells[c]))
        rendered.append(cells)

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    rule = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(r[c].ljust(widths[c]) for c in cols) for r in rendered)
    return f"{header}\n{rule}\n{body}"


def human_bytes(n: float) -> str:
    """Format a byte count with a binary unit suffix.

    Args:
        n: Number of bytes.

    Returns:
        A short human-readable string such as ``"12.4 MiB"``.
    """
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"
