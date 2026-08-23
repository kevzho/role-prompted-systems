"""Configuration objects, vocabularies, and governance constants for ``cca_synth``.

This module is the single source of truth for

* the **functional** role vocabulary (``ROLES``) required by
  ``ETHICS_AND_DATA_GOVERNANCE.md`` -- clinical identifiers appear *only* in
  :data:`CLINICAL_TRACEABILITY`, are never emitted into logs, prompts, tensors,
  or user-facing strings, and exist purely so that a reader of the July 2026
  concept paper can map old names onto new ones;
* the structured-record vocabularies (evidence statuses, failure flags,
  benchmarks, adjudication rules) that pin down the JSON schema of
  ``RESEARCH_SPECIFICATION.md`` section 7;
* a nested, hashable, serialisable :class:`Config` tree covering data, model,
  optimisation, evaluation, and distributed execution.

Nothing in this module imports :mod:`torch`, so it can be loaded by analysis
scripts, notebooks, and CI linters without a GPU stack present.
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import functools
import hashlib
import json
import os
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ROLES",
    "ROLE_TO_INDEX",
    "N_ROLES",
    "CLINICAL_TRACEABILITY",
    "DISCLAIMER",
    "EVIDENCE_STATUSES",
    "EVIDENCE_STATUS_TO_INDEX",
    "EVIDENCE_PAD_INDEX",
    "FAILURE_FLAGS",
    "FLAG_TO_INDEX",
    "BENCHMARKS",
    "BENCHMARK_TO_INDEX",
    "COALITION_METRICS",
    "ADJUDICATION_RULES",
    "AUDIT_FIELD_ALLOWLIST",
    "AUDIT_FIELD_DENY_PATTERNS",
    "DataConfig",
    "ModelConfig",
    "LossConfig",
    "OptimConfig",
    "TrainConfig",
    "EvalConfig",
    "DistConfig",
    "Config",
    "config_to_dict",
    "config_from_dict",
    "apply_overrides",
    "config_hash",
    "add_config_cli_arguments",
    "build_config_from_args",
]


# ---------------------------------------------------------------------------
# Governance constants
# ---------------------------------------------------------------------------

#: Public, function-first role identifiers.  The index of a role in this tuple
#: is its permanent slot index in every tensor produced by :mod:`dataset`.
#: Ordering is fixed for checkpoint compatibility; the model is nevertheless
#: provably invariant to permutations of this axis (see ``model.py``).
ROLES: Tuple[str, ...] = (
    "counter_evidence_checker",
    "consistency_checker",
    "local_claim_checker",
    "explore_then_prune",
    "rare_hypothesis_checker",
    "confidence_calibrator",
)

ROLE_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ROLES)}
N_ROLES: int = len(ROLES)

#: Traceability only.  ``ETHICS_AND_DATA_GOVERNANCE.md`` requires function-first
#: names in code, interfaces, tables, and demonstrations.  This mapping exists so
#: that a reader of the original concept paper can locate the corresponding
#: section; it must never be used to label a tensor, a log line, a metric, a
#: prompt, or any product-facing string.  ``utils.AuditRecordWriter`` actively
#: rejects records containing these values.
CLINICAL_TRACEABILITY: Dict[str, str] = {
    "counter_evidence_checker": "Sentinel",
    "consistency_checker": "Auditor",
    "local_claim_checker": "Savant",
    "explore_then_prune": "Oscillator",
    "rare_hypothesis_checker": "Dissenter",
    "confidence_calibrator": "Challenger",
}

#: Required verbatim disclaimer.  Emitted into every run manifest and printed at
#: the head of every report.
DISCLAIMER: str = (
    "These roles are functional AI processing constraints. They are not "
    "diagnostic models, simulations of lived experience, or claims that mental "
    "health conditions confer a single cognitive mechanism."
)


# ---------------------------------------------------------------------------
# Structured-record vocabularies (RESEARCH_SPECIFICATION.md section 7)
# ---------------------------------------------------------------------------

#: ``evidence_status`` enum from the candidate-record schema.  Index 4 is the
#: padding class used for absent claims and is masked out of every loss.
EVIDENCE_STATUSES: Tuple[str, ...] = (
    "supported",
    "unsupported",
    "contradicted",
    "uncertain",
)
EVIDENCE_STATUS_TO_INDEX: Dict[str, int] = {
    s: i for i, s in enumerate(EVIDENCE_STATUSES)
}
EVIDENCE_PAD_INDEX: int = len(EVIDENCE_STATUSES)

#: Closed vocabulary for ``failure_flags``.  A closed vocabulary is deliberate:
#: an open free-text flag field would re-introduce the unbounded logging that
#: ``ETHICS_AND_DATA_GOVERNANCE.md`` prohibits.
FAILURE_FLAGS: Tuple[str, ...] = (
    "contradiction",
    "calculation_error",
    "unsupported_claim",
    "format_invalid",
    "refusal",
    "truncated_output",
    "injection_suspected",
    "privacy_sensitive",
)
FLAG_TO_INDEX: Dict[str, int] = {f: i for i, f in enumerate(FAILURE_FLAGS)}

#: Evaluation domains from ``RESEARCH_SPECIFICATION.md`` section 5.
BENCHMARKS: Tuple[str, ...] = (
    "truthfulqa",
    "halueval",
    "math_aime",
    "advconsistency",
    "sycophancy",
    "temporal_qa",
    "creative_divergence",
)
BENCHMARK_TO_INDEX: Dict[str, int] = {b: i for i, b in enumerate(BENCHMARKS)}

#: Item-level characteristic functions ``v(.)`` selectable via
#: ``eval.coalition_metric``.  All are oriented so higher is better and all are
#: defined per item, which section 8's paired bootstrap and the Shapley
#: computation both require.  ``selective_accuracy`` is the pre-registered
#: default; the others exist so that a reported result can be checked against a
#: characteristic function that does not depend on the abstention policy
#: (``accuracy``) or on a hard decision at all (``negative_brier``).
COALITION_METRICS: Tuple[str, ...] = (
    "selective_accuracy",
    "accuracy",
    "negative_brier",
)

#: Section 7.4 requires "the rule used for every accepted or rejected material
#: claim".  The synthesiser emits a distribution over this closed rule set for
#: each claim, so the audit record contains a discrete, greppable justification
#: rather than free-form text.
ADJUDICATION_RULES: Tuple[str, ...] = (
    "accepted_evidence_supported",
    "accepted_corroborated_by_independent_agent",
    "rejected_contradicted_by_counterevidence",
    "rejected_unsupported_no_source",
    "deferred_conflict_unresolved",
)

#: Per-field allowlist for the audit trail.  ``ETHICS_AND_DATA_GOVERNANCE.md``
#: requires per-field allowlists and forbids logging full system prompts,
#: credentials, tool tokens, or hidden reasoning.
AUDIT_FIELD_ALLOWLIST: Tuple[str, ...] = (
    "item_id",
    "benchmark",
    "provenance",
    "run_id",
    "config_hash",
    "prompt_version",
    "model_snapshot",
    "timestamp_utc",
    "coalition",
    "final_answer_option",
    "final_answer_text",
    "abstained",
    "synthesizer_confidence",
    "calibrated_confidence",
    "agent_records",
    "agent_slot",
    "agent_role",
    "agent_present",
    "agent_answer_option",
    "agent_declared_confidence",
    "agent_recalibrated_confidence",
    "agent_abstained",
    "counterevidence_count",
    "failure_flags",
    "claims",
    "claim_index",
    "claim_evidence_status",
    "claim_has_source",
    "claim_accepted",
    "claim_rule",
    "claim_accept_probability",
    "unresolved_conflicts",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "retries",
    "cost_usd",
)

#: Substrings that must never appear in an audit-record key.  Enforced by
#: ``utils.AuditRecordWriter``; a match raises rather than silently redacting,
#: because silent redaction hides a bug in the caller.
AUDIT_FIELD_DENY_PATTERNS: Tuple[str, ...] = (
    "chain_of_thought",
    "chainofthought",
    "cot",
    "reasoning_trace",
    "rationale_text",
    "scratchpad",
    "thinking",
    "system_prompt",
    "raw_prompt",
    "api_key",
    "secret",
    "credential",
    "bearer",
    "authorization",
)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Dataset construction, splitting, and padding.

    Attributes:
        records_path: Path to a JSONL file of candidate records.  When ``None``
            the simulator is used and provenance is stamped ``"simulated"``.
        provenance: ``"real"`` or ``"simulated"``.  Derived automatically; an
            explicit value is only honoured for real records.
        simulate_n_items: Number of items the simulator produces.
        simulate_seed: Simulator RNG seed, independent of the training seed so
            that data and optimisation randomness can be varied separately.
        max_options: Hard cap on candidate answer options per item.
        max_claims: Hard cap on atomic claims per agent record.
        split_fractions: ``(train, calibration, validation, test)``.  Section 4
            of the specification requires four disjoint sets.
        split_salt: Salt for the deterministic hash split.  Changing it
            reshuffles the assignment; keep it frozen for a study.
        batch_size: Per-process batch size.
        num_workers: DataLoader worker processes.
        pin_memory: Pin host memory (ignored on non-CUDA devices).
        randomize_eval_order: Section 4 requires randomised test-item order.
        drop_incomplete_items: Drop items whose gold label is unknown from
            training (they remain available for coverage reporting).
        option_feature_dim: Dimensionality of optional precomputed answer-option
            embeddings.  ``0`` disables the projection path.
    """

    records_path: Optional[str] = None
    provenance: str = "simulated"
    simulate_n_items: int = 4096
    simulate_seed: int = 20260726
    max_options: int = 6
    max_claims: int = 12
    split_fractions: Tuple[float, float, float, float] = (0.50, 0.15, 0.15, 0.20)
    split_salt: str = "cca-synth-v1"
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    randomize_eval_order: bool = True
    drop_incomplete_items: bool = True
    option_feature_dim: int = 0

    def __post_init__(self) -> None:
        total = sum(self.split_fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split_fractions must sum to 1.0, got {total:.6f}"
            )
        if any(f < 0 for f in self.split_fractions):
            raise ValueError("split_fractions must be non-negative")
        if self.max_options < 2:
            raise ValueError("max_options must be >= 2")
        if self.provenance not in ("real", "simulated"):
            raise ValueError("provenance must be 'real' or 'simulated'")


@dataclass
class ModelConfig:
    """Architecture of the calibrated synthesiser.

    Attributes:
        d_model: Width of the candidate-token representation.
        n_heads: Attention heads (must divide ``d_model``).
        n_layers: Depth of the permutation-equivariant set encoder.
        d_ff_mult: Feed-forward expansion factor.
        dropout: Dropout probability on attention and feed-forward paths.
        n_scalar_frequencies: Fourier features per scalar input.
        use_role_embeddings: When ``False`` the synthesiser is *role blind*, as
            required by specification section 7.1 ("normalize claims without
            seeing condition names or clinical labels").  This is the primary
            knob for RQ3: gains from operations rather than labels.
        use_cost_features: Feed token/latency/cost into the model.  Off by
            default -- cost is a reporting axis (RQ4), and letting the model
            exploit it invites a verbosity shortcut.
        claim_pool: ``"attention"`` or ``"mean"`` pooling over atomic claims.
        n_status_stats: Size of the deterministic per-agent evidence-status
            simplex appended to the learned pooling (fixed at the vocabulary
            size; exposed for clarity only).
        predict_claims: Enable the per-claim adjudication head.  With this off,
            evidence gating is disabled and the synthesizer degenerates toward a
            confidence-weighted vote; that is a legitimate ablation, not a
            default.
        evidence_gate_floor: Minimum share of its vote weight an agent retains
            when every one of its claims is rejected.  Vote aggregation is
            gated by adjudicated evidence, so an agent whose claims do not hold
            up contributes proportionally less to the answer.  A floor of 0
            would let the model silence an agent entirely on the strength of a
            head that is itself only weakly supervised; 1.0 disables gating and
            restores the original vote-aggregator behaviour.
        predict_rules: Enable the discrete rule head backing the audit record.
        share_calibration_across_roles: Fit one recalibration affine shared by
            all slots.  Required for role-blind operation; a per-role affine
            re-introduces label information.
        init_std: Standard deviation of truncated-normal parameter init.
    """

    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    d_ff_mult: int = 4
    dropout: float = 0.1
    n_scalar_frequencies: int = 4
    use_role_embeddings: bool = True
    use_cost_features: bool = False
    claim_pool: str = "attention"
    n_status_stats: int = len(EVIDENCE_STATUSES)
    predict_claims: bool = True
    evidence_gate_floor: float = 0.25
    predict_rules: bool = True
    share_calibration_across_roles: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        if self.claim_pool not in ("attention", "mean"):
            raise ValueError("claim_pool must be 'attention' or 'mean'")
        if not 0.0 <= self.evidence_gate_floor <= 1.0:
            raise ValueError("evidence_gate_floor must lie in [0, 1]")
        if not self.share_calibration_across_roles and not self.use_role_embeddings:
            raise ValueError(
                "per-role calibration requires role embeddings; a role-blind "
                "synthesizer must not receive role identity through the "
                "calibration head either"
            )


@dataclass
class LossConfig:
    """Loss weights and the anti-capture regulariser.

    Attributes:
        w_answer: Weight on masked answer-selection cross entropy.
        w_abstain: Weight on the abstention BCE.
        w_calibration: Weight on the per-agent confidence-recalibration BCE.
        w_claim: Weight on the per-claim adjudication BCE.
        w_rule: Weight on the discrete rule cross entropy.
        w_dominance: Weight on the pooling-dominance penalty, which implements
            the "synthesizer capture by one verbose agent" test of section 9 as
            a *training-time* constraint rather than only a post-hoc probe.
        dominance_cap: Maximum tolerated pooling mass on a single agent before
            the penalty engages.
        label_smoothing: Label smoothing on the answer head.
    """

    w_answer: float = 1.0
    w_abstain: float = 0.30
    w_calibration: float = 0.50
    w_claim: float = 0.25
    w_rule: float = 0.10
    w_dominance: float = 0.05
    dominance_cap: float = 0.60
    label_smoothing: float = 0.02


@dataclass
class OptimConfig:
    """Optimiser and schedule.

    Attributes:
        lr: Peak learning rate.
        weight_decay: Decoupled weight decay (never applied to norms or biases).
        betas: AdamW betas.
        eps: AdamW epsilon.
        grad_clip: Global gradient-norm clip; ``0`` disables.
        warmup_ratio: Fraction of total steps spent in linear warmup.
        min_lr_ratio: Floor of the cosine schedule as a fraction of ``lr``.
        accumulation_steps: Gradient accumulation micro-steps.
    """

    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.05
    accumulation_steps: int = 1


@dataclass
class TrainConfig:
    """Training loop, coalition dropout, and checkpointing.

    Attributes:
        epochs: Number of passes over the training split.
        seed: Global seed for Python, NumPy, and torch RNGs.
        deterministic: Request deterministic kernels where available.
        amp: Enable mixed precision.  Automatically disabled on devices where
            it is not beneficial or not reliable (see ``utils.amp_settings``).
        amp_dtype: ``"bfloat16"`` or ``"float16"``.
        ema_decay: Exponential moving average of weights; ``0`` disables.
        coalition_dropout: Per-role probability of masking an agent during
            training.  This is what makes one trained synthesiser valid for all
            2**6 coalitions at evaluation time.
        p_full_coalition: Probability of forcing the complete council.
        p_empty_coalition: Probability of forcing the empty coalition, whose
            value is the direct-answer baseline (specification section 8).
        eval_every: Validate every N epochs.
        early_stop_patience: Validation rounds without improvement before
            stopping; ``0`` disables.
        select_metric: Validation metric used for checkpoint selection.
        select_mode: ``"max"`` or ``"min"``.
        output_dir: Root directory for checkpoints, logs, and reports.
        run_name: Sub-directory name; defaults to the config hash.
        log_every: Step interval for training log lines.
        save_optimizer_state: Persist optimiser/scaler state for resumption.
    """

    epochs: int = 30
    seed: int = 1337
    deterministic: bool = True
    amp: bool = True
    amp_dtype: str = "bfloat16"
    ema_decay: float = 0.999
    coalition_dropout: float = 0.35
    p_full_coalition: float = 0.25
    p_empty_coalition: float = 0.05
    eval_every: int = 1
    early_stop_patience: int = 8
    select_metric: str = "selective_accuracy"
    select_mode: str = "max"
    output_dir: str = "runs"
    run_name: Optional[str] = None
    log_every: int = 50
    save_optimizer_state: bool = True

    def __post_init__(self) -> None:
        if self.amp_dtype not in ("bfloat16", "float16"):
            raise ValueError("amp_dtype must be 'bfloat16' or 'float16'")
        if self.select_mode not in ("max", "min"):
            raise ValueError("select_mode must be 'max' or 'min'")
        if not 0.0 <= self.coalition_dropout < 1.0:
            raise ValueError("coalition_dropout must lie in [0, 1)")
        if self.p_full_coalition + self.p_empty_coalition > 1.0:
            raise ValueError(
                "p_full_coalition + p_empty_coalition must not exceed 1.0"
            )


@dataclass
class EvalConfig:
    """Evaluation, calibration, ablation, and statistics.

    Attributes:
        coalition_metric: Item-level score used as the characteristic function
            ``v(.)`` in the Shapley computation.
        abstain_credit: Item score awarded for a correct abstention decision.
            Predeclared before the run, per section 8.
        target_coverage: Coverage at which the selective-prediction threshold is
            fitted on the calibration split.
        n_bootstrap: Bootstrap resamples for confidence intervals.
        bootstrap_seed: Seed for the bootstrap resampler.
        ci_level: Two-sided confidence level.
        fdr_alpha: Benjamini-Hochberg level for the agent-by-benchmark family.
        n_calibration_bins: Bins for expected calibration error.
        calibration_binning: ``"equal_mass"`` or ``"equal_width"``.
        shapley_mode: ``"masked"`` reuses one coalition-dropout-trained model
            for all 64 coalitions; ``"retrain"`` trains a separate synthesiser
            per coalition.  ``"masked"`` is an approximation and is reported as
            such.
        run_robustness: Execute the section 9 perturbation battery.
        robustness_seed: Seed for adversarial perturbations.
        permutation_tolerance: Maximum tolerated logit deviation in the
            permutation-invariance test.
        write_audit_records: Emit per-item structured decision records.
        max_audit_records: Cap on records written, to bound artifact size.
        strict_real: Abort if data provenance is not ``"real"``.
        attribution_estimator: ``"shapley"`` runs the full 64-coalition sweep;
            ``"leave_one_out"`` evaluates only the full council and the six
            drop-one coalitions -- the estimator PREREGISTRATION.md A2.2
            registers for the scoped study.  Under ``shapley_mode="retrain"``
            this is 7 trainings instead of 64.  Leave-one-out is a different
            estimand from the Shapley value, not an approximation of it, and
            the ablation artifact says so.
        loo_moves_threshold: Minimum leave-one-out contrast for a role to count
            as moving its benchmark.  PREREGISTRATION.md A2.2 fixes this at
            0.05 for the scoped study -- five times H2's SESOI of 0.01, because
            A2.5 shows the scoped sample size cannot resolve anything smaller.
            The threshold is a registered decision rule, so it lives in the
            config and is computed into the artifact rather than applied in
            prose.
    """

    coalition_metric: str = "selective_accuracy"
    #: Repeated item-level trials per coalition.  Section 8 permits evaluating
    #: all 64 coalitions only if each receives enough repeated trials on a fixed
    #: item/seed schedule reused across coalitions.
    n_trials_per_item: int = 3
    abstain_credit: float = 0.0
    target_coverage: float = 0.90
    n_bootstrap: int = 2000
    bootstrap_seed: int = 90210
    ci_level: float = 0.95
    fdr_alpha: float = 0.05
    n_calibration_bins: int = 15
    calibration_binning: str = "equal_mass"
    shapley_mode: str = "masked"
    run_robustness: bool = True
    robustness_seed: int = 4242
    permutation_tolerance: float = 1e-4
    write_audit_records: bool = True
    max_audit_records: int = 2000
    strict_real: bool = False
    attribution_estimator: str = "shapley"
    loo_moves_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.n_trials_per_item < 1:
            raise ValueError("n_trials_per_item must be >= 1")
        if self.loo_moves_threshold < 0.0:
            raise ValueError("loo_moves_threshold must be >= 0")
        if self.coalition_metric not in COALITION_METRICS:
            raise ValueError(
                f"coalition_metric must be one of {COALITION_METRICS}; "
                f"got {self.coalition_metric!r}"
            )
        if self.shapley_mode not in ("masked", "retrain"):
            raise ValueError("shapley_mode must be 'masked' or 'retrain'")
        if self.attribution_estimator not in ("shapley", "leave_one_out"):
            raise ValueError(
                "attribution_estimator must be 'shapley' or 'leave_one_out'"
            )
        if self.calibration_binning not in ("equal_mass", "equal_width"):
            raise ValueError(
                "calibration_binning must be 'equal_mass' or 'equal_width'"
            )
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("ci_level must lie in (0, 1)")


@dataclass
class DistConfig:
    """Distributed and device placement.

    Attributes:
        device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
        backend: ``"auto"`` selects NCCL on CUDA and Gloo elsewhere.
        find_unused_parameters: DDP flag; kept ``False`` because every head is
            exercised on every step (heads disabled by config are removed from
            the module tree entirely, not merely skipped).
        timeout_minutes: Process-group rendezvous timeout.
        compile: Wrap the model in ``torch.compile``.  Off by default: the MPS
            inductor path is not uniformly available.
        allow_tf32: Permit TF32 matmuls on Ampere-and-later CUDA devices.
    """

    device: str = "auto"
    backend: str = "auto"
    find_unused_parameters: bool = False
    timeout_minutes: int = 30
    compile: bool = False
    allow_tf32: bool = True


@dataclass
class Config:
    """Root configuration tree."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    dist: DistConfig = field(default_factory=DistConfig)

    def validate(self) -> "Config":
        """Re-run every nested ``__post_init__`` validator and cross-checks.

        Returns:
            ``self``, to allow ``cfg = Config(...).validate()``.

        Raises:
            ValueError: If any nested invariant or cross-section invariant is
                violated.
        """
        for f in fields(self):
            child = getattr(self, f.name)
            if is_dataclass(child) and hasattr(child, "__post_init__"):
                child.__post_init__()  # type: ignore[attr-defined]
        if self.eval.strict_real and self.data.provenance != "real":
            raise ValueError(
                "eval.strict_real=True but data.provenance is "
                f"'{self.data.provenance}'; refusing to produce a report that "
                "could be mistaken for an empirical result"
            )
        if self.data.option_feature_dim < 0:
            raise ValueError("option_feature_dim must be non-negative")
        return self

    @property
    def run_id(self) -> str:
        """Stable run identifier: explicit name, else the config hash."""
        return self.train.run_name or f"run-{config_hash(self)[:12]}"


# ---------------------------------------------------------------------------
# (De)serialisation helpers
# ---------------------------------------------------------------------------


def config_to_dict(cfg: Any) -> Any:
    """Recursively convert a dataclass tree into JSON-serialisable primitives.

    Tuples become lists so that a round trip through JSON is stable.

    Args:
        cfg: A dataclass instance, or any nested container of primitives.

    Returns:
        A structure containing only ``dict``, ``list``, ``str``, ``int``,
        ``float``, ``bool``, and ``None``.
    """
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, Mapping):
        return {str(k): config_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [config_to_dict(v) for v in cfg]
    return cfg


@functools.lru_cache(maxsize=None)
def _resolved_hints(cls: type) -> Dict[str, Any]:
    """Resolve a dataclass's annotations into real typing objects.

    ``from __future__ import annotations`` turns every ``field.type`` into a
    string, which would defeat :func:`_coerce`.  ``typing.get_type_hints``
    evaluates them against this module's globals; the result is cached because
    the config classes are fixed at import time.

    Args:
        cls: A dataclass type defined in this module.

    Returns:
        Mapping from field name to its resolved annotation.
    """
    return typing.get_type_hints(cls)


def _coerce(value: Any, target_type: Any) -> Any:
    """Best-effort coercion of ``value`` to ``target_type``.

    Handles ``Optional[T]`` and fixed-length tuple annotations, both of which
    appear in this config tree and neither of which ``dataclasses`` coerces.

    Args:
        value: Raw value from JSON or the CLI.
        target_type: The dataclass field annotation.

    Returns:
        The coerced value, or ``value`` unchanged when no rule applies.
    """
    origin = getattr(target_type, "__origin__", None)
    args = getattr(target_type, "__args__", ())

    if origin is not None and type(None) in args:  # Optional[T]
        if value is None:
            return None
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        return _coerce(value, non_none[0]) if non_none else value

    if origin in (tuple, Tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"expected a sequence for {target_type}, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0]) for v in value)
        if args and len(args) != len(value):
            raise ValueError(
                f"expected {len(args)} elements for {target_type}, got {len(value)}"
            )
        return tuple(_coerce(v, a) for v, a in zip(value, args or [Any] * len(value)))

    if origin in (list, List):
        inner = args[0] if args else Any
        return [_coerce(v, inner) for v in value]

    if target_type is bool:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"cannot interpret {value!r} as bool")
        return bool(value)
    if target_type is int and not isinstance(value, bool):
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    return value


def config_from_dict(payload: Mapping[str, Any], base: Optional[Config] = None) -> Config:
    """Build a :class:`Config` by overlaying ``payload`` onto ``base``.

    Unknown keys raise rather than being ignored: a silently dropped
    hyper-parameter is a reproducibility hazard, and section 4 of the
    specification requires the execution manifest to be exact.

    Args:
        payload: Nested mapping, e.g. parsed from ``configs/default.json``.
        base: Starting configuration; a fresh :class:`Config` when ``None``.

    Returns:
        A validated :class:`Config`.

    Raises:
        KeyError: If ``payload`` contains an unknown section or field.
    """
    cfg = copy.deepcopy(base) if base is not None else Config()
    section_names = {f.name for f in fields(cfg)}
    for section, values in payload.items():
        if section not in section_names:
            raise KeyError(
                f"unknown config section '{section}'; expected one of "
                f"{sorted(section_names)}"
            )
        target = getattr(cfg, section)
        hints = _resolved_hints(type(target))
        if not isinstance(values, Mapping):
            raise TypeError(f"section '{section}' must be a mapping")
        for key, value in values.items():
            if key not in hints:
                raise KeyError(
                    f"unknown config field '{section}.{key}'; expected one of "
                    f"{sorted(hints)}"
                )
            setattr(target, key, _coerce(value, hints[key]))
    return cfg.validate()


def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    """Apply ``section.field=value`` override strings in order.

    Values are parsed with :func:`ast.literal_eval` where possible and fall back
    to the raw string, so ``--set train.seed=7`` and ``--set
    train.select_metric=accuracy`` both behave.

    Args:
        cfg: Configuration to modify (mutated in place and returned).
        overrides: Strings of the form ``"section.field=value"``.

    Returns:
        The validated, mutated configuration.

    Raises:
        ValueError: If an override is malformed.
        KeyError: If it names an unknown section or field.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"malformed override '{override}'; expected 'section.field=value'"
            )
        dotted, raw = override.split("=", 1)
        parts = dotted.strip().split(".")
        if len(parts) != 2:
            raise ValueError(
                f"malformed override key '{dotted}'; expected 'section.field'"
            )
        section, key = parts
        try:
            parsed: Any = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = raw
        cfg = config_from_dict({section: {key: parsed}}, base=cfg)
    return cfg.validate()


def config_hash(cfg: Config) -> str:
    """Return a stable SHA-256 hash of the configuration.

    Fields that do not affect computation (``output_dir``, ``run_name``,
    ``num_workers``, ``log_every``) are excluded so that the same experiment
    hashes identically across machines.

    Args:
        cfg: Configuration to hash.

    Returns:
        Hex digest string.
    """
    payload = config_to_dict(cfg)
    for section, keys in (
        ("train", ("output_dir", "run_name", "log_every")),
        ("data", ("num_workers", "pin_memory")),
        ("dist", ("device", "backend", "timeout_minutes")),
    ):
        for key in keys:
            payload[section].pop(key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def add_config_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the shared configuration flags to ``parser``.

    Args:
        parser: An ``argparse`` parser or sub-parser.

    Returns:
        The same parser, for chaining.
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file overlaid on the defaults.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="Override a single config field; repeatable.",
    )
    parser.add_argument(
        "--records",
        type=str,
        default=None,
        help="Path to a JSONL file of real candidate records (sets provenance=real).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override train.seed.")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override train.output_dir."
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Override train.run_name."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Override dist.device.",
    )
    parser.add_argument(
        "--strict-real",
        action="store_true",
        help="Abort unless the loaded data has provenance 'real'.",
    )
    return parser


def build_config_from_args(args: argparse.Namespace) -> Config:
    """Materialise a :class:`Config` from parsed CLI arguments.

    Precedence, lowest to highest: dataclass defaults, ``--config`` file,
    ``--set`` overrides, then the dedicated convenience flags.

    Args:
        args: Namespace produced by a parser built with
            :func:`add_config_cli_arguments`.

    Returns:
        A validated :class:`Config`.
    """
    cfg = Config()
    config_path = getattr(args, "config", None)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = config_from_dict(json.load(handle), base=cfg)
    cfg = apply_overrides(cfg, getattr(args, "overrides", []) or [])

    records = getattr(args, "records", None)
    if records:
        cfg.data.records_path = records
        cfg.data.provenance = "real"
    if getattr(args, "seed", None) is not None:
        cfg.train.seed = int(args.seed)
    if getattr(args, "output_dir", None):
        cfg.train.output_dir = args.output_dir
    if getattr(args, "run_name", None):
        cfg.train.run_name = args.run_name
    if getattr(args, "device", None):
        cfg.dist.device = args.device
    if getattr(args, "strict_real", False):
        cfg.eval.strict_real = True

    if cfg.data.records_path is None:
        cfg.data.provenance = "simulated"
    return cfg.validate()


def save_config(cfg: Config, path: str) -> None:
    """Write ``cfg`` to ``path`` as pretty-printed JSON.

    Args:
        cfg: Configuration to persist.
        path: Destination file path; parent directories are created.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config_to_dict(cfg), handle, indent=2, sort_keys=True)
        handle.write("\n")
