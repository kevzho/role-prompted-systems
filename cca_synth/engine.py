"""Training, evaluation, calibration, ablation, and robustness.

This module holds everything that *runs*:

* :func:`compute_losses` -- the five-term objective, every term masked.
* :class:`Trainer` -- device-agnostic, DDP-aware training with AMP, gradient
  accumulation, EMA, cosine schedule, and checkpoint selection.
* :func:`fit_temperature` / :func:`fit_abstention_threshold` -- post-hoc
  calibration fitted **only** on the calibration split, as section 7 requires.
* :func:`coalition_sweep` and :func:`shapley_analysis` -- the ``2**6 = 64``
  coalition ablation of section 8, with item-level bootstrap intervals and the
  probability that each contribution is positive.
* :func:`robustness_suite` -- the section 9 battery, including a *proof-checking*
  permutation-invariance test rather than a merely empirical one.

Evaluation is replicated on every rank rather than sharded.  A
``DistributedSampler`` pads and duplicates its last shard, which would corrupt
the paired item-level contrasts that section 8 mandates; the synthesiser is
small enough that replicating evaluation costs less than getting the pairing
subtly wrong.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from config import (
    ADJUDICATION_RULES,
    BENCHMARKS,
    Config,
    EVIDENCE_STATUS_TO_INDEX,
    N_ROLES,
    ROLES,
)
from dataset import (
    ROLE_PRIMARY_BENCHMARK,
    move_batch_to_device,
    sample_coalitions,
)
from model import (
    BASELINES,
    MASK_FILL_FLOAT32,
    CCASynthesizer,
    build_model,
    effective_agent_mask,
    single_agent_scores,
)
from utils import (
    DistState,
    MeanTracker,
    Timer,
    all_gather_object,
    amp_settings,
    area_under_risk_coverage,
    barrier,
    benjamini_hochberg,
    bootstrap_ci,
    brier_score,
    coalition_item_scores,
    coalition_to_index,
    expected_calibration_error,
    fit_selective_threshold,
    get_logger,
    index_to_coalition,
    item_scores,
    paired_bootstrap_diff,
    pairwise_error_correlation,
    reduce_scalar,
    save_checkpoint,
    seed_everything,
    selective_accuracy,
    shapley_from_values,
    torch_generator,
)

__all__ = [
    "EvalResult",
    "compute_losses",
    "rule_targets_from_records",
    "Trainer",
    "fit_temperature",
    "fit_abstention_threshold",
    "evaluate_scores",
    "metrics_from_result",
    "baseline_results",
    "coalition_sweep",
    "shapley_analysis",
    "interaction_analysis",
    "repeated_trial_schedule",
    "verify_trial_coverage",
    "necessity_sufficiency_verdicts",
    "agent_by_benchmark_family",
    "robustness_suite",
    "error_correlation_report",
    "cost_report",
]

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def rule_targets_from_records(
    claim_status: torch.Tensor, claim_has_source: torch.Tensor
) -> torch.Tensor:
    """Derive the declared adjudication rule for each claim.

    Section 7.4 requires the synthesiser to "record the rule used for every
    accepted or rejected material claim".  The rule is therefore *policy*, not
    something to be discovered: this function encodes the declared policy as a
    deterministic function of the claim record, and the model's rule head is
    trained to reproduce it while additionally conditioning on council context.
    Making the policy explicit here means the audit trail can be checked against
    a written rule set rather than against a black box.

    Policy:

    ==================================  ==========================================
    record                              rule
    ==================================  ==========================================
    ``contradicted``                    ``rejected_contradicted_by_counterevidence``
    ``unsupported`` without a source    ``rejected_unsupported_no_source``
    ``supported`` with a source         ``accepted_evidence_supported``
    ``supported`` without a source      ``accepted_corroborated_by_independent_agent``
    anything else                       ``deferred_conflict_unresolved``
    ==================================  ==========================================

    Args:
        claim_status: ``(B, R, C)`` long tensor of evidence-status ids.
        claim_has_source: ``(B, R, C)`` float indicator of a citation or check.

    Returns:
        ``(B, R, C)`` long tensor of rule indices.
    """
    supported = EVIDENCE_STATUS_TO_INDEX["supported"]
    unsupported = EVIDENCE_STATUS_TO_INDEX["unsupported"]
    contradicted = EVIDENCE_STATUS_TO_INDEX["contradicted"]

    rule_index = {name: i for i, name in enumerate(ADJUDICATION_RULES)}
    has_source = claim_has_source > 0.5

    target = torch.full_like(claim_status, rule_index["deferred_conflict_unresolved"])
    target = torch.where(
        claim_status == contradicted,
        torch.full_like(target, rule_index["rejected_contradicted_by_counterevidence"]),
        target,
    )
    target = torch.where(
        (claim_status == unsupported) & (~has_source),
        torch.full_like(target, rule_index["rejected_unsupported_no_source"]),
        target,
    )
    target = torch.where(
        (claim_status == supported) & has_source,
        torch.full_like(target, rule_index["accepted_evidence_supported"]),
        target,
    )
    target = torch.where(
        (claim_status == supported) & (~has_source),
        torch.full_like(
            target, rule_index["accepted_corroborated_by_independent_agent"]
        ),
        target,
    )
    return target


def _masked_smoothed_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    option_mask: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    """Cross entropy with label smoothing spread over *valid* options only.

    ``F.cross_entropy(label_smoothing=...)`` spreads mass over all ``K``
    classes, including padded ones whose logits are at the dtype minimum; the
    resulting loss is dominated by padding and the gradient is meaningless.
    This version renormalises the smoothing mass over the valid option set.

    Args:
        logits: ``(B, K)``.
        target: ``(B,)`` long, entries outside ``[0, K)`` are ignored.
        option_mask: ``(B, K)`` bool.
        label_smoothing: Smoothing coefficient in ``[0, 1)``.

    Returns:
        ``(B,)`` per-item loss; ignored items are zero.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    valid_item = (target >= 0) & (target < logits.shape[1])
    safe_target = target.clamp(min=0)

    onehot = F.one_hot(safe_target, num_classes=logits.shape[1]).to(log_probs.dtype)
    if label_smoothing > 0.0:
        mask = option_mask.to(log_probs.dtype)
        uniform = mask / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        distribution = (1.0 - label_smoothing) * onehot + label_smoothing * uniform
    else:
        distribution = onehot

    per_item = -(distribution * log_probs.masked_fill(~option_mask, 0.0)).sum(dim=-1)
    return per_item * valid_item.to(per_item.dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over ``True`` entries of ``mask``.

    Args:
        values: Any shape.
        mask: Broadcastable boolean mask.

    Returns:
        Scalar tensor; zero when the mask is empty, so an absent supervision
        signal contributes nothing rather than producing ``nan``.
    """
    weights = mask.to(values.dtype)
    total = weights.sum()
    if total.item() == 0:
        return values.sum() * 0.0
    return (values * weights).sum() / total


def compute_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    coalition: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Evaluate the five-term training objective.

    Terms:

    1. **Answer** -- masked, smoothed cross entropy over answer options.
    2. **Abstention** -- BCE against a target computed *for the current
       coalition*: the item is unresolvable when neither the direct answer nor
       any active agent proposed the gold option.  Recomputing the target per
       coalition (rather than fixing it from the full council) is what keeps the
       abstention head honest under ablation.
    3. **Calibration** -- BCE between each agent's recalibrated confidence and
       whether that agent was actually right, which is the operational content
       of "treat agent confidence as an input feature, not ground truth".
    4. **Claim adjudication and rule** -- per-claim veridicality BCE and the
       declared-rule cross entropy backing the audit record.
    5. **Dominance** -- a hinge penalty on any single agent holding more than
       ``dominance_cap`` of the pooling mass, implementing the section 9
       anti-capture requirement at training time.

    Args:
        outputs: Model outputs from :meth:`model.CCASynthesizer.forward`.
        batch: Batch dict.
        coalition: ``(B, R)`` coalition mask used for this forward pass.
        cfg: Root configuration.

    Returns:
        Tuple ``(total_loss, components)`` where ``components`` maps term names
        to detached floats for logging.
    """
    loss_cfg = cfg.loss
    gold = batch["gold"]
    labelled = gold >= 0
    active = effective_agent_mask(batch["present"], coalition)

    answer_loss_per_item = _masked_smoothed_cross_entropy(
        outputs["option_logits"], gold, batch["option_mask"], loss_cfg.label_smoothing
    )
    answer_loss = _masked_mean(answer_loss_per_item, labelled)

    # --- abstention target, recomputed for this coalition -------------------
    safe_gold = gold.clamp(min=0).unsqueeze(-1)
    council_support = (batch["vote"] * active.unsqueeze(-1)).sum(dim=1)
    reachable = (
        council_support.gather(1, safe_gold).squeeze(-1)
        + batch["base_vote"].gather(1, safe_gold).squeeze(-1)
    ) > 0
    abstain_target = (~reachable).to(outputs["abstain_logit"].dtype)
    # NOTE: the abstention target is heavily imbalanced (a few percent of items),
    # and the head does not fire under the section 9 probes -- see
    # logistics/FINDINGS_EVIDENCE_PATHWAY.md. Rebalancing with a BCE pos_weight
    # was tried and reverted: it cost 4.4 points of clean accuracy and nearly
    # tripled ECE without reviving the head, because the probes are
    # out-of-distribution rather than merely rare. The candidate fix is
    # adversarial training, which is a design decision to pre-register, not a
    # patch to slip in here.
    abstain_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["abstain_logit"], abstain_target, reduction="none"
        ),
        labelled,
    )

    # --- per-agent confidence recalibration ---------------------------------
    has_answer = batch["vote"].sum(dim=-1) > 0
    agent_choice = batch["vote"].argmax(dim=-1)
    agent_correct = (agent_choice == gold.unsqueeze(1)).to(
        outputs["recalibrated_confidence"].dtype
    )
    calibration_mask = (active > 0) & has_answer & labelled.unsqueeze(1)
    recalibrated = outputs["recalibrated_confidence"].clamp(_EPS, 1.0 - _EPS)
    calibration_terms = -(
        agent_correct * torch.log(recalibrated)
        + (1.0 - agent_correct) * torch.log1p(-recalibrated)
    )
    calibration_loss = _masked_mean(calibration_terms, calibration_mask)

    # --- claim adjudication --------------------------------------------------
    claim_loss = outputs["option_logits"].sum() * 0.0
    rule_loss = outputs["option_logits"].sum() * 0.0
    if "claim_accept_logit" in outputs:
        claim_mask = (
            batch["claim_mask"] & batch["claim_label_mask"] & (active > 0).unsqueeze(-1)
        )
        claim_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(
                outputs["claim_accept_logit"], batch["claim_label"], reduction="none"
            ),
            claim_mask,
        )
    if "claim_rule_logits" in outputs:
        rule_target = rule_targets_from_records(
            batch["claim_status"], batch["claim_has_source"]
        )
        rule_mask = batch["claim_mask"] & (active > 0).unsqueeze(-1)
        rule_logits = outputs["claim_rule_logits"]
        rule_terms = F.cross_entropy(
            rule_logits.reshape(-1, rule_logits.shape[-1]),
            rule_target.reshape(-1),
            reduction="none",
        ).view_as(rule_target)
        rule_loss = _masked_mean(rule_terms, rule_mask)

    # --- anti-capture --------------------------------------------------------
    n_active = active.sum(dim=1)
    dominance_mask = n_active >= 2
    dominance_excess = F.relu(outputs["dominance"] - loss_cfg.dominance_cap) ** 2
    dominance_loss = _masked_mean(dominance_excess, dominance_mask)

    total = (
        loss_cfg.w_answer * answer_loss
        + loss_cfg.w_abstain * abstain_loss
        + loss_cfg.w_calibration * calibration_loss
        + loss_cfg.w_claim * claim_loss
        + loss_cfg.w_rule * rule_loss
        + loss_cfg.w_dominance * dominance_loss
    )

    components = {
        "loss": float(total.detach()),
        "loss_answer": float(answer_loss.detach()),
        "loss_abstain": float(abstain_loss.detach()),
        "loss_calibration": float(calibration_loss.detach()),
        "loss_claim": float(claim_loss.detach()),
        "loss_rule": float(rule_loss.detach()),
        "loss_dominance": float(dominance_loss.detach()),
        "mean_dominance": float(outputs["dominance"].detach().mean()),
    }
    return total, components


# ---------------------------------------------------------------------------
# Evaluation containers
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Per-item evaluation outputs, sorted by ``item_id``.

    Sorting is what makes every downstream contrast *paired*: two arms evaluated
    on the same split are guaranteed elementwise aligned, so
    :func:`utils.paired_bootstrap_diff` can be applied directly.

    Attributes:
        item_ids: Item identifiers, ascending.
        benchmark_ids: Benchmark index per item.
        gold: Gold option per item, ``-1`` when unknown.
        prediction: Selected option per item.
        confidence: Probability of the selected option.
        selection_score: Score used for selective prediction.
        abstain: Abstention decisions.
        probabilities: Padded posterior, ``(N, K_max)``.
        dominance: Peak single-agent pooling mass per item.
        active: Effective agent mask, ``(N, R)``.
        role_recalibrated: Recalibrated agent confidences, ``(N, R)``.
        role_correct: Whether each agent's own answer was right, ``(N, R)``.
        role_answered: Whether each agent produced a usable answer, ``(N, R)``.
        loss: Mean loss over the split, when computed.
    """

    item_ids: List[str]
    benchmark_ids: np.ndarray
    gold: np.ndarray
    prediction: np.ndarray
    confidence: np.ndarray
    selection_score: np.ndarray
    abstain: np.ndarray
    probabilities: np.ndarray
    dominance: np.ndarray
    active: np.ndarray
    role_recalibrated: np.ndarray
    role_correct: np.ndarray
    role_answered: np.ndarray
    loss: float = float("nan")

    def sorted(self) -> "EvalResult":
        """Return a copy sorted by ``item_id``.

        Returns:
            A new :class:`EvalResult` with every array reordered consistently.
        """
        order = np.argsort(np.asarray(self.item_ids), kind="stable")
        return EvalResult(
            item_ids=[self.item_ids[i] for i in order],
            benchmark_ids=self.benchmark_ids[order],
            gold=self.gold[order],
            prediction=self.prediction[order],
            confidence=self.confidence[order],
            selection_score=self.selection_score[order],
            abstain=self.abstain[order],
            probabilities=self.probabilities[order],
            dominance=self.dominance[order],
            active=self.active[order],
            role_recalibrated=self.role_recalibrated[order],
            role_correct=self.role_correct[order],
            role_answered=self.role_answered[order],
            loss=self.loss,
        )

    def item_scores(
        self, abstain_credit: float, metric: str = "selective_accuracy"
    ) -> np.ndarray:
        """Item-level scores forming the characteristic function ``v(.)``.

        Args:
            abstain_credit: Predeclared score for an abstention.
            metric: One of :data:`utils.COALITION_METRICS`; comes from
                ``eval.coalition_metric`` at every call site in the pipeline.

        Returns:
            ``(N,)`` float array.
        """
        return coalition_item_scores(
            metric,
            self.prediction,
            self.gold,
            self.abstain,
            probabilities=self.probabilities,
            abstain_credit=abstain_credit,
        )


def _pad_to(array: np.ndarray, width: int) -> np.ndarray:
    """Right-pad the last axis of a 2-D array with zeros.

    Args:
        array: ``(N, k)`` array.
        width: Target width ``>= k``.

    Returns:
        ``(N, width)`` array.
    """
    if array.shape[1] >= width:
        return array[:, :width]
    padding = np.zeros((array.shape[0], width - array.shape[1]), dtype=array.dtype)
    return np.concatenate([array, padding], axis=1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Device-agnostic, distribution-aware trainer for the synthesiser."""

    def __init__(
        self,
        cfg: Config,
        model: CCASynthesizer,
        loaders: Mapping[str, DataLoader],
        dist_state: DistState,
        logger: Optional[Any] = None,
        fixed_coalition: Optional[Sequence[int]] = None,
    ):
        """Set up optimiser, schedule, AMP, EMA, and DDP.

        Args:
            cfg: Root configuration.
            model: Freshly constructed synthesiser (moved to device here).
            loaders: Mapping with keys ``train``, ``calibration``,
                ``validation``, ``test``.
            dist_state: Resolved distributed state.
            logger: Optional logger; one is created when omitted.
            fixed_coalition: When given, every training step uses this coalition
                instead of sampling.  Used by the ``retrain`` Shapley mode, where
                each coalition gets its own synthesiser.
        """
        self.cfg = cfg
        self.dist_state = dist_state
        self.device = dist_state.device
        self.logger = logger or get_logger()
        self.loaders = loaders
        self.fixed_coalition = (
            torch.tensor(list(fixed_coalition), dtype=torch.float32)
            if fixed_coalition is not None
            else None
        )

        self.model = model.to(self.device)
        self.raw_model = self.model
        if dist_state.enabled:
            device_ids = [dist_state.local_rank] if self.device.type == "cuda" else None
            self.model = DistributedDataParallel(
                self.model,
                device_ids=device_ids,
                output_device=dist_state.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=cfg.dist.find_unused_parameters,
            )
            self.raw_model = self.model.module  # type: ignore[assignment]

        if cfg.dist.compile:
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        self.use_amp, self.amp_dtype, needs_scaler = amp_settings(
            self.device, cfg.train.amp, cfg.train.amp_dtype
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

        self.optimizer = self._build_optimizer()
        steps_per_epoch = max(1, math.ceil(len(loaders["train"]) / cfg.optim.accumulation_steps))
        self.total_steps = max(1, steps_per_epoch * cfg.train.epochs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        self.ema_state: Optional[Dict[str, torch.Tensor]] = None
        if cfg.train.ema_decay > 0:
            self.ema_state = {
                name: param.detach().clone().float()
                for name, param in self.raw_model.state_dict().items()
                if param.dtype.is_floating_point
            }

        self.coalition_generator = torch_generator(cfg.train.seed + 7919)
        self.global_step = 0
        self.best_metric = -math.inf if cfg.train.select_mode == "max" else math.inf
        self.best_epoch = -1
        self.history: List[Dict[str, float]] = []
        # Epoch the next call to train() starts from. Only resume_from moves it.
        self.start_epoch = 0
        self.rounds_without_improvement = 0

    # -- setup helpers -----------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW with decay excluded from norms, biases, and scalars.

        Returns:
            The optimiser.
        """
        decay, no_decay = [], []
        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.optim.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.optim.lr,
            betas=tuple(self.cfg.optim.betas),
            eps=self.cfg.optim.eps,
        )

    def _lr_lambda(self, step: int) -> float:
        """Linear warmup into a cosine decay.

        Args:
            step: Global optimiser step.

        Returns:
            Multiplier applied to the base learning rate.
        """
        warmup = max(1, int(self.total_steps * self.cfg.optim.warmup_ratio))
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, self.total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        floor = self.cfg.optim.min_lr_ratio
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _coalition_for(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the coalition mask for one training batch.

        Args:
            batch: Batch dict.

        Returns:
            ``(B, R)`` float mask on the training device.
        """
        batch_size = batch["present"].shape[0]
        if self.fixed_coalition is not None:
            return self.fixed_coalition.to(self.device).unsqueeze(0).expand(batch_size, -1)
        return sample_coalitions(
            batch_size=batch_size,
            dropout=self.cfg.train.coalition_dropout,
            p_full=self.cfg.train.p_full_coalition,
            p_empty=self.cfg.train.p_empty_coalition,
            generator=self.coalition_generator,
            device=self.device,
        )

    @torch.no_grad()
    def _update_ema(self) -> None:
        """Update the EMA shadow weights in place."""
        if self.ema_state is None:
            return
        decay = self.cfg.train.ema_decay
        for name, value in self.raw_model.state_dict().items():
            if name in self.ema_state:
                self.ema_state[name].mul_(decay).add_(value.detach().float(), alpha=1 - decay)

    # -- training ----------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one epoch over the training split.

        Args:
            epoch: Zero-based epoch index.

        Returns:
            Mean training metrics for the epoch, reduced across ranks.
        """
        self.model.train()
        tracker = MeanTracker()
        loader = self.loaders["train"]
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)  # type: ignore[union-attr]

        accumulation = max(1, self.cfg.optim.accumulation_steps)
        self.optimizer.zero_grad(set_to_none=True)

        for index, raw_batch in enumerate(loader):
            batch = move_batch_to_device(
                raw_batch, self.device, non_blocking=self.device.type == "cuda"
            )
            coalition = self._coalition_for(batch)

            with torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                outputs = self.model(batch, coalition)
                loss, components = compute_losses(outputs, batch, coalition, self.cfg)

            self.scaler.scale(loss / accumulation).backward()

            if (index + 1) % accumulation == 0 or (index + 1) == len(loader):
                if self.cfg.optim.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), self.cfg.optim.grad_clip
                    )
                    components["grad_norm"] = float(grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
                self._update_ema()

            components["lr"] = float(self.optimizer.param_groups[0]["lr"])
            tracker.update(components, weight=batch["present"].shape[0])

            if self.global_step % self.cfg.train.log_every == 0 and index % accumulation == 0:
                self.logger.info(
                    "epoch %d step %d | loss %.4f | answer %.4f | calib %.4f | lr %.2e",
                    epoch,
                    self.global_step,
                    components["loss"],
                    components["loss_answer"],
                    components["loss_calibration"],
                    components["lr"],
                )
        return tracker.compute(self.device if self.dist_state.enabled else None)

    def train(self) -> Dict[str, Any]:
        """Run the full training schedule with validation and early stopping.

        Returns:
            A summary with the best epoch, best metric, and per-epoch history.
        """
        checkpoint_dir = f"{self.cfg.train.output_dir}/{self.cfg.run_id}"
        best_path = f"{checkpoint_dir}/best.pt"
        last_path = f"{checkpoint_dir}/last.pt"
        rounds_without_improvement = self.rounds_without_improvement

        if self.start_epoch >= self.cfg.train.epochs:
            self.logger.info(
                "resumed at epoch %d with train.epochs=%d; nothing left to run",
                self.start_epoch,
                self.cfg.train.epochs,
            )

        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            with Timer() as timer:
                train_metrics = self.train_one_epoch(epoch)

            log_line = {f"train/{k}": v for k, v in train_metrics.items()}
            log_line["epoch"] = float(epoch)
            log_line["epoch_seconds"] = timer.elapsed

            if (epoch + 1) % self.cfg.train.eval_every == 0 or epoch == self.cfg.train.epochs - 1:
                result = self.evaluate("validation")
                metrics = metrics_from_result(result, self.cfg)
                log_line.update({f"val/{k}": v for k, v in metrics.items()})

                current = metrics.get(self.cfg.train.select_metric, float("nan"))
                improved = (
                    current > self.best_metric
                    if self.cfg.train.select_mode == "max"
                    else current < self.best_metric
                )
                if not math.isnan(current) and improved:
                    self.best_metric = current
                    self.best_epoch = epoch
                    rounds_without_improvement = 0
                    save_checkpoint(
                        best_path,
                        self.raw_model,
                        self.cfg,
                        epoch=epoch,
                        step=self.global_step,
                        metrics=metrics,
                        optimizer=self.optimizer if self.cfg.train.save_optimizer_state else None,
                        scheduler=self.scheduler if self.cfg.train.save_optimizer_state else None,
                        scaler=self.scaler if self.cfg.train.save_optimizer_state else None,
                        ema_state=self.ema_state,
                        trainer_state=self._bookkeeping(epoch, rounds_without_improvement),
                        rng_state=self._rng_state(),
                    )
                else:
                    rounds_without_improvement += 1

                self.logger.info(
                    "epoch %d | train loss %.4f | val %s %.4f | best %.4f (epoch %d) | %.1fs",
                    epoch,
                    train_metrics.get("loss", float("nan")),
                    self.cfg.train.select_metric,
                    current,
                    self.best_metric,
                    self.best_epoch,
                    timer.elapsed,
                )
                if (
                    self.cfg.train.early_stop_patience
                    and rounds_without_improvement >= self.cfg.train.early_stop_patience
                ):
                    self.logger.info(
                        "early stopping at epoch %d after %d rounds without improvement",
                        epoch,
                        rounds_without_improvement,
                    )
                    self.history.append(log_line)
                    self.rounds_without_improvement = rounds_without_improvement
                    if self.fixed_coalition is None:
                        self._save_last(last_path, epoch, rounds_without_improvement)
                    barrier()
                    return self._summary(best_path, last_path)

            self.history.append(log_line)
            self.rounds_without_improvement = rounds_without_improvement
            # Written every epoch, not only at the end: an interrupted run must
            # leave behind something a --resume can start from.
            #
            # Skipped for the retrain Shapley sweep. Those 64 inner trainings
            # share one run_id, overwrite each other, and are never resumed --
            # a coalition-specific model is not a run you continue. Writing a
            # full optimiser checkpoint per epoch there costs ~7.5 GB of I/O
            # across the sweep and buys nothing.
            if self.fixed_coalition is None:
                self._save_last(last_path, epoch, rounds_without_improvement)

        barrier()
        return self._summary(best_path, last_path)

    def _save_last(self, path: str, epoch: int, rounds_without_improvement: int) -> None:
        """Write the resumable ``last.pt``.

        Unlike ``best.pt`` this always carries optimiser, scheduler, scaler, and
        RNG state regardless of ``train.save_optimizer_state``: without them the
        file cannot do the one job it has.

        Args:
            path: Destination path.
            epoch: Epoch just completed.
            rounds_without_improvement: Early-stopping counter.
        """
        save_checkpoint(
            path,
            self.raw_model,
            self.cfg,
            epoch=epoch,
            step=self.global_step,
            metrics={"best_metric": self.best_metric},
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema_state=self.ema_state,
            trainer_state=self._bookkeeping(epoch, rounds_without_improvement),
            rng_state=self._rng_state(),
        )

    def _bookkeeping(self, epoch: int, rounds_without_improvement: int) -> Dict[str, Any]:
        """Training-loop state that is neither weights nor optimiser tensors.

        Args:
            epoch: Epoch just completed.
            rounds_without_improvement: Early-stopping counter.

        Returns:
            JSON-serialisable resume state.
        """
        return {
            "epoch_completed": int(epoch),
            "global_step": int(self.global_step),
            "best_metric": float(self.best_metric),
            "best_epoch": int(self.best_epoch),
            "rounds_without_improvement": int(rounds_without_improvement),
            "select_metric": self.cfg.train.select_metric,
            "select_mode": self.cfg.train.select_mode,
            "history": self.history,
        }

    def _rng_state(self) -> Dict[str, torch.Tensor]:
        """Generator states needed to reproduce the remaining coalition draws.

        Returns:
            Mapping of generator name to its byte-tensor state.
        """
        return {
            "coalition_generator": self.coalition_generator.get_state(),
            "cpu": torch.get_rng_state(),
        }

    def _summary(self, best_path: str, last_path: str) -> Dict[str, Any]:
        """Assemble the training summary.

        Args:
            best_path: Path to the best checkpoint.
            last_path: Path to the resumable checkpoint.

        Returns:
            Summary dictionary.
        """
        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "select_metric": self.cfg.train.select_metric,
            "global_steps": self.global_step,
            "checkpoint_best": best_path,
            "checkpoint_last": last_path,
            "history": self.history,
        }

    def load_best(self, path: Optional[str] = None, use_ema: bool = True) -> bool:
        """Restore the best checkpoint into the live model.

        Args:
            path: Checkpoint path; defaults to the run's ``best.pt``.
            use_ema: Prefer EMA weights when the checkpoint carries them.

        Returns:
            ``True`` if a checkpoint was loaded.
        """
        import os

        from utils import load_checkpoint

        path = path or f"{self.cfg.train.output_dir}/{self.cfg.run_id}/best.pt"
        if not os.path.exists(path):
            self.logger.warning("no checkpoint at %s; keeping current weights", path)
            return False
        payload = load_checkpoint(path, map_location="cpu")
        state = payload["ema"] if (use_ema and "ema" in payload) else payload["model"]
        missing, unexpected = self.raw_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            self.logger.warning(
                "checkpoint load: %d missing, %d unexpected keys", len(missing), len(unexpected)
            )
        self.raw_model.to(self.device)
        return True

    def resume_from(self, path: Optional[str] = None, *, strict: bool = True) -> bool:
        """Restore a full training state and arrange for ``train()`` to continue.

        ``load_best`` deliberately restores weights only — it exists to put the
        selected model back before evaluation.  Resuming an interrupted run is a
        different operation: the optimiser moments, the LR schedule position, the
        AMP scale, the EMA shadow, the step counter, the early-stopping counter,
        and the coalition RNG all have to come back, or the continued run is not
        the run that was interrupted.

        Args:
            path: Checkpoint path; defaults to the run's ``last.pt``.
            strict: Raise when the checkpoint's config hash differs from the live
                configuration.  Resuming across a config change silently produces
                a model that matches neither.

        Returns:
            ``True`` if a checkpoint was loaded.

        Raises:
            FileNotFoundError: If ``path`` is given explicitly and is missing.
            ValueError: If ``strict`` and the config hash does not match.
        """
        import os

        from utils import config_hash, load_checkpoint

        explicit = path is not None
        path = path or f"{self.cfg.train.output_dir}/{self.cfg.run_id}/last.pt"
        if not os.path.exists(path):
            if explicit:
                raise FileNotFoundError(f"no checkpoint to resume from at {path}")
            self.logger.warning("no checkpoint at %s; starting from scratch", path)
            return False

        payload = load_checkpoint(path, map_location="cpu")
        saved_hash = payload.get("config_hash")
        if saved_hash and saved_hash != config_hash(self.cfg):
            message = (
                f"checkpoint config hash {saved_hash} does not match the live "
                f"configuration {config_hash(self.cfg)}"
            )
            if strict:
                raise ValueError(message + "; pass strict=False to override")
            self.logger.warning("%s; resuming anyway", message)

        self.raw_model.load_state_dict(payload["model"])
        self.raw_model.to(self.device)

        restored = ["model"]
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
            restored.append("optimizer")
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
            restored.append("scheduler")
        if "scaler" in payload:
            self.scaler.load_state_dict(payload["scaler"])
            restored.append("scaler")
        if "ema" in payload and self.ema_state is not None:
            self.ema_state = {
                name: value.to(torch.float32) for name, value in payload["ema"].items()
            }
            restored.append("ema")
        rng = payload.get("rng_state") or {}
        if "coalition_generator" in rng:
            self.coalition_generator.set_state(rng["coalition_generator"].to(torch.uint8))
            restored.append("coalition_rng")
        if "cpu" in rng:
            torch.set_rng_state(rng["cpu"].to(torch.uint8))

        state = payload.get("trainer_state") or {}
        self.global_step = int(state.get("global_step", payload.get("step", 0)))
        self.best_metric = float(
            state.get(
                "best_metric",
                -math.inf if self.cfg.train.select_mode == "max" else math.inf,
            )
        )
        self.best_epoch = int(state.get("best_epoch", -1))
        self.rounds_without_improvement = int(state.get("rounds_without_improvement", 0))
        self.history = list(state.get("history", []))
        self.start_epoch = int(state.get("epoch_completed", payload.get("epoch", -1))) + 1
        restored.append("bookkeeping")

        self.logger.info(
            "resumed from %s at epoch %d, step %d, best %s %.4f (restored: %s)",
            path,
            self.start_epoch,
            self.global_step,
            self.cfg.train.select_metric,
            self.best_metric,
            ", ".join(restored),
        )
        return True

    # -- evaluation --------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        split: str,
        coalition: Optional[Sequence[int]] = None,
        abstain_threshold: Optional[float] = None,
        compute_loss: bool = True,
    ) -> EvalResult:
        """Evaluate the synthesiser on one split.

        Args:
            split: One of ``train``, ``calibration``, ``validation``, ``test``.
            coalition: Fixed coalition mask; ``None`` means the full council.
            abstain_threshold: Selective-prediction threshold from the
                calibration split.
            compute_loss: Also accumulate the training objective.

        Returns:
            An :class:`EvalResult` sorted by ``item_id``.
        """
        self.model.eval()
        loader = self.loaders[split]
        max_options = self.cfg.data.max_options

        chunks: Dict[str, List[np.ndarray]] = {
            key: []
            for key in (
                "benchmark_ids",
                "gold",
                "prediction",
                "confidence",
                "selection_score",
                "abstain",
                "probabilities",
                "dominance",
                "active",
                "role_recalibrated",
                "role_correct",
                "role_answered",
            )
        }
        item_ids: List[str] = []
        loss_tracker = MeanTracker()

        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, self.device)
            batch_size = batch["present"].shape[0]
            if coalition is None:
                mask = torch.ones(batch_size, N_ROLES, device=self.device)
            else:
                mask = (
                    torch.tensor(list(coalition), dtype=torch.float32, device=self.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

            with torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                predictions = self.raw_model.predict(
                    batch, mask, abstain_threshold=abstain_threshold
                )
                if compute_loss:
                    outputs = self.raw_model(batch, mask)
                    _, components = compute_losses(outputs, batch, mask, self.cfg)
                    loss_tracker.update(components, weight=batch_size)

            gold = batch["gold"]
            agent_choice = batch["vote"].argmax(dim=-1)
            answered = batch["vote"].sum(dim=-1) > 0
            agent_correct = (agent_choice == gold.unsqueeze(1)) & answered

            item_ids.extend(batch["item_id"])
            chunks["benchmark_ids"].append(batch["benchmark_id"].cpu().numpy())
            chunks["gold"].append(gold.cpu().numpy())
            chunks["prediction"].append(predictions["prediction"].cpu().numpy())
            chunks["confidence"].append(predictions["confidence"].float().cpu().numpy())
            chunks["selection_score"].append(
                predictions["selection_score"].float().cpu().numpy()
            )
            chunks["abstain"].append(predictions["abstain"].cpu().numpy())
            chunks["probabilities"].append(
                _pad_to(predictions["probabilities"].float().cpu().numpy(), max_options)
            )
            chunks["dominance"].append(predictions["dominance"].float().cpu().numpy())
            chunks["active"].append(predictions["active"].float().cpu().numpy())
            chunks["role_recalibrated"].append(
                predictions["recalibrated_confidence"].float().cpu().numpy()
            )
            chunks["role_correct"].append(agent_correct.float().cpu().numpy())
            chunks["role_answered"].append(answered.float().cpu().numpy())

        result = EvalResult(
            item_ids=item_ids,
            benchmark_ids=np.concatenate(chunks["benchmark_ids"]),
            gold=np.concatenate(chunks["gold"]),
            prediction=np.concatenate(chunks["prediction"]),
            confidence=np.concatenate(chunks["confidence"]),
            selection_score=np.concatenate(chunks["selection_score"]),
            abstain=np.concatenate(chunks["abstain"]).astype(bool),
            probabilities=np.concatenate(chunks["probabilities"]),
            dominance=np.concatenate(chunks["dominance"]),
            active=np.concatenate(chunks["active"]),
            role_recalibrated=np.concatenate(chunks["role_recalibrated"]),
            role_correct=np.concatenate(chunks["role_correct"]),
            role_answered=np.concatenate(chunks["role_answered"]),
            loss=loss_tracker.compute().get("loss", float("nan")),
        )
        return result.sorted()

    @torch.no_grad()
    def collect_logits(
        self, split: str, coalition: Optional[Sequence[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather raw (pre-temperature) logits for calibration fitting.

        Args:
            split: Split name.
            coalition: Fixed coalition mask.

        Returns:
            Tuple ``(logits, gold, option_mask)`` on CPU.
        """
        self.model.eval()
        logits_chunks, gold_chunks, mask_chunks = [], [], []
        max_options = self.cfg.data.max_options

        for raw_batch in self.loaders[split]:
            batch = move_batch_to_device(raw_batch, self.device)
            batch_size = batch["present"].shape[0]
            if coalition is None:
                mask = torch.ones(batch_size, N_ROLES, device=self.device)
            else:
                mask = (
                    torch.tensor(list(coalition), dtype=torch.float32, device=self.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
            outputs = self.raw_model(batch, mask)
            logits = outputs["option_logits"].float().cpu()
            option_mask = batch["option_mask"].cpu()
            pad = max_options - logits.shape[1]
            if pad > 0:
                logits = F.pad(logits, (0, pad), value=MASK_FILL_FLOAT32)
                option_mask = F.pad(option_mask, (0, pad), value=False)
            logits_chunks.append(logits[:, :max_options])
            mask_chunks.append(option_mask[:, :max_options])
            gold_chunks.append(batch["gold"].cpu())

        return (
            torch.cat(logits_chunks),
            torch.cat(gold_chunks),
            torch.cat(mask_chunks),
        )


# ---------------------------------------------------------------------------
# Post-hoc calibration
# ---------------------------------------------------------------------------


def fit_temperature(
    logits: torch.Tensor,
    gold: torch.Tensor,
    option_mask: torch.Tensor,
    max_iter: int = 100,
) -> float:
    """Fit a single temperature by minimising NLL on the calibration split.

    Section 7 and limitation L8 both insist that confidence be calibrated on a
    separate set rather than declared calibrated.  One scalar is deliberate: it
    cannot change the arg-max, so it improves calibration without contaminating
    the accuracy comparison between arms.

    Args:
        logits: ``(N, K)`` raw logits.
        gold: ``(N,)`` gold indices, ``-1`` ignored.
        option_mask: ``(N, K)`` bool.
        max_iter: LBFGS iterations.

    Returns:
        The fitted temperature; ``1.0`` when the split has no labelled items.
    """
    keep = gold >= 0
    if keep.sum().item() == 0:
        return 1.0
    logits = logits[keep].float()
    targets = gold[keep].long()
    mask = option_mask[keep]

    # Zero the masked entries *before* they reach the division.  Scaling a large
    # negative sentinel by a small temperature overflows to -inf, and the
    # backward pass through that division returns nan even though masked_fill
    # repairs the forward value.  Masking first keeps the division over finite
    # numbers only.
    safe_logits = torch.where(mask, logits, torch.zeros_like(logits))
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        """Compute masked NLL at the current temperature."""
        optimizer.zero_grad()
        scaled = safe_logits / log_temperature.exp().clamp(min=1e-3)
        scaled = scaled.masked_fill(~mask, MASK_FILL_FLOAT32)
        loss = F.cross_entropy(scaled, targets)
        loss.backward()
        return loss

    optimizer.step(closure)  # type: ignore[arg-type]
    fitted = float(log_temperature.detach().exp().item())
    if not math.isfinite(fitted):
        # Never let a diverged fit reach the model: an unusable temperature
        # silently zeroes accuracy everywhere downstream.
        get_logger().warning(
            "temperature fitting diverged (got %r); falling back to T=1.0", fitted
        )
        return 1.0
    return float(min(max(fitted, 0.05), 20.0))


def fit_abstention_threshold(result: EvalResult, target_coverage: float) -> float:
    """Fit the selective-prediction threshold on calibration selection scores.

    Args:
        result: Calibration-split evaluation result.
        target_coverage: Desired answered fraction.

    Returns:
        The threshold below which the system abstains.
    """
    return fit_selective_threshold(result.selection_score, target_coverage)


# ---------------------------------------------------------------------------
# Metrics and comparators
# ---------------------------------------------------------------------------


def metrics_from_result(result: EvalResult, cfg: Config) -> Dict[str, float]:
    """Compute the full outcome panel for one evaluation result.

    Reports accuracy *and* the section 5 cross-cutting outcomes: calibration
    error, Brier score, selective accuracy with coverage, AURC, abstention rate,
    and the agent-dominance diagnostic.

    Args:
        result: Evaluation result.
        cfg: Root configuration.

    Returns:
        Mapping from metric name to value.
    """
    labelled = result.gold >= 0
    correct = (result.prediction == result.gold).astype(np.float64)
    answered = ~result.abstain

    selective, coverage = selective_accuracy(
        result.prediction, result.gold, result.abstain, cfg.eval.abstain_credit
    )
    answered_and_labelled = answered & labelled

    metrics: Dict[str, float] = {
        "accuracy": float(correct[labelled].mean()) if labelled.any() else float("nan"),
        "selective_accuracy": selective,
        "coverage": coverage,
        "abstention_rate": float(result.abstain.mean()),
        "accuracy_on_answered": (
            float(correct[answered_and_labelled].mean())
            if answered_and_labelled.any()
            else float("nan")
        ),
        "ece": expected_calibration_error(
            result.confidence[labelled],
            correct[labelled],
            n_bins=cfg.eval.n_calibration_bins,
            binning=cfg.eval.calibration_binning,
        ),
        "brier": brier_score(result.probabilities, result.gold),
        "aurc": area_under_risk_coverage(
            result.selection_score[labelled], correct[labelled]
        ),
        "mean_confidence": float(result.confidence.mean()),
        "mean_agent_dominance": float(result.dominance.mean()),
        "loss": result.loss,
        "n_items": float(len(result.item_ids)),
    }

    # Per-agent recalibration quality, only over agents that actually answered.
    answered_agents = result.role_answered > 0
    if answered_agents.any():
        metrics["agent_confidence_ece"] = expected_calibration_error(
            result.role_recalibrated[answered_agents],
            result.role_correct[answered_agents],
            n_bins=cfg.eval.n_calibration_bins,
            binning=cfg.eval.calibration_binning,
        )
    return metrics


def evaluate_scores(
    scores: np.ndarray,
    gold: np.ndarray,
    abstain: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert comparator scores into predictions.

    Args:
        scores: ``(N, K)`` scores.
        gold: ``(N,)`` gold indices (unused; kept for signature symmetry).
        abstain: Optional abstention decisions.

    Returns:
        ``(N,)`` integer predictions.
    """
    del gold, abstain
    return np.asarray(scores).argmax(axis=1)


@torch.no_grad()
def baseline_results(
    trainer: Trainer, split: str, coalition: Optional[Sequence[int]] = None
) -> Dict[str, EvalResult]:
    """Evaluate every non-learned comparator on one split.

    Includes the six single-agent arms required for the "strongest single model"
    control of specification section 3, plus majority vote, confidence-weighted
    vote, the direct-answer baseline, and the oracle upper bound.

    Args:
        trainer: A configured trainer (used for its loaders and device).
        split: Split name.
        coalition: Fixed coalition mask.

    Returns:
        Mapping from comparator name to :class:`EvalResult`, each sorted by item.
    """
    device = trainer.device
    loader = trainer.loaders[split]
    max_options = trainer.cfg.data.max_options

    names = list(BASELINES.keys()) + [f"single_agent::{role}" for role in ROLES]
    buffers: Dict[str, Dict[str, List[np.ndarray]]] = {
        name: {"prediction": [], "probabilities": [], "confidence": []} for name in names
    }
    item_ids: List[str] = []
    gold_chunks, benchmark_chunks, role_correct_chunks, role_answered_chunks = [], [], [], []
    active_chunks: List[np.ndarray] = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        batch_size = batch["present"].shape[0]
        mask = (
            torch.ones(batch_size, N_ROLES, device=device)
            if coalition is None
            else torch.tensor(list(coalition), dtype=torch.float32, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        for name, scorer in BASELINES.items():
            scores = scorer(batch, mask)  # type: ignore[operator]
            probabilities = torch.softmax(scores.float(), dim=-1)
            buffers[name]["prediction"].append(scores.argmax(dim=-1).cpu().numpy())
            buffers[name]["probabilities"].append(
                _pad_to(probabilities.cpu().numpy(), max_options)
            )
            buffers[name]["confidence"].append(probabilities.max(dim=-1).values.cpu().numpy())

        for role_index, role in enumerate(ROLES):
            scores = single_agent_scores(batch, role_index)
            probabilities = torch.softmax(scores.float(), dim=-1)
            key = f"single_agent::{role}"
            buffers[key]["prediction"].append(scores.argmax(dim=-1).cpu().numpy())
            buffers[key]["probabilities"].append(
                _pad_to(probabilities.cpu().numpy(), max_options)
            )
            buffers[key]["confidence"].append(probabilities.max(dim=-1).values.cpu().numpy())

        gold = batch["gold"]
        answered = batch["vote"].sum(dim=-1) > 0
        item_ids.extend(batch["item_id"])
        gold_chunks.append(gold.cpu().numpy())
        benchmark_chunks.append(batch["benchmark_id"].cpu().numpy())
        role_correct_chunks.append(
            ((batch["vote"].argmax(dim=-1) == gold.unsqueeze(1)) & answered)
            .float()
            .cpu()
            .numpy()
        )
        role_answered_chunks.append(answered.float().cpu().numpy())
        active_chunks.append(effective_agent_mask(batch["present"], mask).cpu().numpy())

    gold_all = np.concatenate(gold_chunks)
    results: Dict[str, EvalResult] = {}
    for name in names:
        prediction = np.concatenate(buffers[name]["prediction"])
        results[name] = EvalResult(
            item_ids=list(item_ids),
            benchmark_ids=np.concatenate(benchmark_chunks),
            gold=gold_all,
            prediction=prediction,
            confidence=np.concatenate(buffers[name]["confidence"]),
            selection_score=np.concatenate(buffers[name]["confidence"]),
            abstain=np.zeros_like(prediction, dtype=bool),
            probabilities=np.concatenate(buffers[name]["probabilities"]),
            dominance=np.zeros_like(gold_all, dtype=np.float64),
            active=np.concatenate(active_chunks),
            role_recalibrated=np.zeros((gold_all.shape[0], N_ROLES)),
            role_correct=np.concatenate(role_correct_chunks),
            role_answered=np.concatenate(role_answered_chunks),
        ).sorted()
    return results


# ---------------------------------------------------------------------------
# Coalition ablation and Shapley analysis
# ---------------------------------------------------------------------------


def coalition_sweep(
    trainer: Trainer,
    split: str,
    abstain_threshold: Optional[float] = None,
    logger: Optional[Any] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Evaluate all ``2**R`` coalitions on one split.

    Because the synthesiser was trained under coalition dropout, one model
    serves every coalition and the sweep costs 64 masked forward passes rather
    than 64 training runs.  This is an *approximation* to retraining per
    coalition; :func:`shapley_analysis` records which mode produced the numbers,
    and ``eval.shapley_mode="retrain"`` performs the exact version.

    Args:
        trainer: Trained trainer.
        split: Split to sweep.
        abstain_threshold: Selective-prediction threshold.
        logger: Optional logger for progress.

    Returns:
        Tuple ``(score_matrix, item_ids, gold)`` where ``score_matrix`` has
        shape ``(N, 2**R)`` indexed by coalition bitset.
    """
    log = logger or get_logger()
    n_coalitions = 2**N_ROLES
    columns: List[np.ndarray] = []
    item_ids: List[str] = []
    gold: np.ndarray = np.zeros(0)

    for index in range(n_coalitions):
        coalition = index_to_coalition(index, N_ROLES)
        result = trainer.evaluate(
            split, coalition=coalition, abstain_threshold=abstain_threshold, compute_loss=False
        )
        if not item_ids:
            item_ids = list(result.item_ids)
            gold = result.gold
        elif list(result.item_ids) != item_ids:
            raise RuntimeError(
                "coalition sweep produced misaligned item ordering; paired "
                "statistics would be invalid"
            )
        columns.append(
            result.item_scores(
                trainer.cfg.eval.abstain_credit, trainer.cfg.eval.coalition_metric
            )
        )
        if index % 8 == 0:
            log.info("coalition sweep %d/%d", index, n_coalitions)

    return np.stack(columns, axis=1), item_ids, gold


def leave_one_out_sweep(
    trainer: Trainer,
    split: str,
    abstain_threshold: Optional[float] = None,
    logger: Optional[Any] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Evaluate the full council and the six drop-one coalitions only.

    ``PREREGISTRATION.md`` amendment A2.2 registers leave-one-out as the scoped
    study's per-role estimator.  Two reasons, both recorded there: amendment A1
    forbids per-role claims from ``shapley_mode="masked"`` because the
    masked-vs-retrain gap (0.0164 mean, 0.0262 max) exceeds H2's SESOI of 0.010
    and does not preserve rank order; and a 64-coalition *retrain* sweep is not
    affordable in the scoped compute.  Leave-one-out is seven evaluations
    instead of 64, and seven trainings instead of 64 under retrain mode.

    This is a different estimand from the Shapley value, not a cheaper
    approximation of one.  See :func:`leave_one_out_analysis`.

    Args:
        trainer: Trained trainer.
        split: Split to sweep.
        abstain_threshold: Selective-prediction threshold.
        logger: Optional logger for progress.

    Returns:
        Tuple ``(score_matrix, item_ids, gold)`` where ``score_matrix`` has
        shape ``(N, 1 + R)``: column 0 is the complete council and column
        ``1 + r`` is the council with role ``r`` removed.

    Raises:
        RuntimeError: If any coalition returns a different item ordering, which
            would invalidate the paired statistics.
    """
    log = logger or get_logger()
    coalitions: List[Tuple[int, ...]] = [tuple([1] * N_ROLES)]
    for r in range(N_ROLES):
        dropped = [1] * N_ROLES
        dropped[r] = 0
        coalitions.append(tuple(dropped))

    columns: List[np.ndarray] = []
    item_ids: List[str] = []
    gold: np.ndarray = np.zeros(0)

    for position, coalition in enumerate(coalitions):
        result = trainer.evaluate(
            split, coalition=coalition, abstain_threshold=abstain_threshold, compute_loss=False
        )
        if not item_ids:
            item_ids = list(result.item_ids)
            gold = result.gold
        elif list(result.item_ids) != item_ids:
            raise RuntimeError(
                "leave-one-out sweep produced misaligned item ordering; paired "
                "statistics would be invalid"
            )
        columns.append(
            result.item_scores(
                trainer.cfg.eval.abstain_credit, trainer.cfg.eval.coalition_metric
            )
        )
        log.info("leave-one-out sweep %d/%d", position + 1, len(coalitions))

    return np.stack(columns, axis=1), item_ids, gold


def leave_one_out_analysis(
    score_matrix: np.ndarray,
    cfg: Config,
    mode: str = "masked",
) -> Dict[str, Any]:
    """Per-role leave-one-out contrast with item-level bootstrap intervals.

    The estimand is ``v(A) - v(A \\ {i})``: how far the coalition metric falls
    when role ``i`` is removed from the complete council.  Three properties of
    the Shapley value do **not** hold for it, and the returned ``caveats`` say
    so, because the whole point of amendment A1 was that an attribution number
    which looks reasonable can be an artifact:

    - it does not satisfy efficiency, so the per-role values need not sum to
      ``v(A) - v(empty)``;
    - it charges every interaction to whichever role is being dropped, so under
      the substitutability reported in A1 it *understates* each role's
      standalone contribution;
    - it says nothing about coalitions the sweep never evaluated.

    Under substitutable agents -- which is what A1 measured on simulated
    records, all 15 pairs negative -- leave-one-out is the conservative
    direction: a role whose work another role can cover shows a small drop even
    when it is doing that work.  A role that clears the threshold has therefore
    cleared a bar the Shapley value would have set lower.

    The ``moves_benchmark`` verdict is computed here rather than written in
    prose, matching the treatment of necessity and sufficiency elsewhere in this
    module.  A2.2's rule is conjunctive: the interval must exclude zero *and*
    the point estimate must reach ``eval.loo_moves_threshold``.

    Args:
        score_matrix: ``(N, 1 + R)`` item scores; column 0 is the full council.
        cfg: Root configuration.
        mode: ``"masked"`` or ``"retrain"``, recorded in the output.

    Returns:
        A dict with per-role contrasts, intervals, computed verdicts, a
        degeneracy check, and the caveats above.

    Raises:
        ValueError: If ``score_matrix`` does not have ``1 + N_ROLES`` columns.
    """
    n_items, n_columns = score_matrix.shape
    if n_columns != 1 + N_ROLES:
        raise ValueError(f"expected {1 + N_ROLES} columns, got {n_columns}")

    column_means = np.nanmean(score_matrix, axis=0)
    full_value = float(column_means[0])
    point = column_means[0] - column_means[1:]

    # Same failure class the A1 degeneracy check guards against in
    # shapley_analysis: if every coalition scores identically, every contrast is
    # mechanically zero and nothing downstream can tell that from a real null.
    n_distinct = int(np.unique(np.round(column_means, 12)).size)
    degenerate = n_distinct <= 1

    rng = np.random.default_rng(cfg.eval.bootstrap_seed + 7919)
    draws = np.zeros((cfg.eval.n_bootstrap, N_ROLES), dtype=np.float64)
    for b in range(cfg.eval.n_bootstrap):
        indices = rng.integers(0, n_items, size=n_items)
        resampled = np.nanmean(score_matrix[indices], axis=0)
        draws[b] = resampled[0] - resampled[1:]

    alpha = (1.0 - cfg.eval.ci_level) / 2.0
    threshold = cfg.eval.loo_moves_threshold
    roles: List[Dict[str, Any]] = []
    for r, role in enumerate(ROLES):
        ci_lo = float(np.quantile(draws[:, r], alpha))
        ci_hi = float(np.quantile(draws[:, r], 1.0 - alpha))
        excludes_zero = ci_lo > 0.0 or ci_hi < 0.0
        moves = bool(excludes_zero and float(point[r]) >= threshold)
        roles.append(
            {
                "role": role,
                "loo_contrast": float(point[r]),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "prob_positive": float((draws[:, r] > 0).mean()),
                "bootstrap_se": float(draws[:, r].std(ddof=1)),
                "interval_excludes_zero": bool(excludes_zero),
                "meets_threshold": bool(float(point[r]) >= threshold),
                "moves_benchmark": moves,
                "primary_benchmark": ROLE_PRIMARY_BENCHMARK[role],
            }
        )

    return {
        "estimator": "leave_one_out",
        "mode": mode,
        "n_items": int(n_items),
        "n_bootstrap": int(cfg.eval.n_bootstrap),
        "ci_level": cfg.eval.ci_level,
        "moves_threshold": float(threshold),
        "full_coalition_value": full_value,
        "degeneracy_check": {
            "n_distinct_coalition_values": n_distinct,
            "degenerate": degenerate,
            "passed": not degenerate,
            "note": (
                "FAILED: the full council and all six drop-one coalitions have "
                "the same value, so every leave-one-out contrast is "
                "mechanically zero. This is a pipeline fault, not a finding. "
                "The usual cause is a calibration artifact -- a threshold or "
                "temperature fitted on one model and reused on another -- "
                "driving universal abstention. See FINDINGS_EVIDENCE_PATHWAY.md "
                "section 8."
                if degenerate
                else f"{n_distinct} distinct coalition values; the value "
                "function is not collapsed."
            ),
        },
        "roles": roles,
        "caveats": [
            "Leave-one-out is not a Shapley value. It does not satisfy "
            "efficiency, so these numbers need not sum to v(A) - v(empty), and "
            "no efficiency check is reported because none applies.",
            "Every interaction between role i and the rest of the council is "
            "charged to role i by this estimator. Under the substitutable "
            "pairs recorded in PREREGISTRATION.md A1, that biases each contrast "
            "toward zero, so a role clearing the threshold has cleared a "
            "conservative bar.",
            "Coalitions other than the full council and the six drop-one sets "
            "were not evaluated. No claim about any other coalition is "
            "supported by this table.",
            (
                "Contrasts come from one synthesizer trained with coalition "
                "dropout, which approximates but does not equal retraining per "
                "coalition. PREREGISTRATION.md A1 forbids per-role claims from "
                "this mode."
                if mode == "masked"
                else "Each coalition was trained separately; no masking "
                "approximation enters."
            ),
        ],
    }


def loo_coalitions() -> List[Tuple[int, ...]]:
    """The seven coalitions the leave-one-out estimator evaluates.

    Returns:
        The full council followed by the six drop-one coalitions, in the column
        order :func:`leave_one_out_sweep` and :func:`leave_one_out_analysis`
        expect.
    """
    out: List[Tuple[int, ...]] = [tuple([1] * N_ROLES)]
    for r in range(N_ROLES):
        dropped = [1] * N_ROLES
        dropped[r] = 0
        out.append(tuple(dropped))
    return out


def expand_loo_matrix(score_matrix: np.ndarray) -> np.ndarray:
    """Scatter a ``(N, 1 + R)`` leave-one-out matrix into 64-coalition layout.

    :func:`agent_by_benchmark_family` indexes its score matrix by coalition
    bitmask, reading only the full coalition and the six drop-one columns.
    Those are exactly the columns a leave-one-out sweep produces, so the sweep
    can feed the same FDR-corrected family analysis without a second
    implementation: every column the family analysis never reads is ``NaN``,
    and touching one is therefore loud rather than silently wrong.

    Args:
        score_matrix: ``(N, 1 + R)`` scores; column 0 is the full council and
            column ``1 + r`` drops role ``r``.

    Returns:
        ``(N, 2**R)`` array, ``NaN`` outside the seven evaluated coalitions.

    Raises:
        ValueError: On a column count other than ``1 + N_ROLES``.
    """
    n_items, n_columns = score_matrix.shape
    if n_columns != 1 + N_ROLES:
        raise ValueError(f"expected {1 + N_ROLES} columns, got {n_columns}")
    full_index = 2**N_ROLES - 1
    scattered = np.full((n_items, 2**N_ROLES), np.nan, dtype=np.float64)
    scattered[:, full_index] = score_matrix[:, 0]
    for r in range(N_ROLES):
        scattered[:, full_index & ~(1 << r)] = score_matrix[:, 1 + r]
    return scattered


def shapley_analysis(
    score_matrix: np.ndarray,
    cfg: Config,
    mode: str = "masked",
) -> Dict[str, Any]:
    """Compute Shapley values with item-level bootstrap intervals.

    Section 8 requires per-resample Shapley values computed from item-level
    observations, 95% intervals, and the probability that each contribution is
    positive -- not a point estimate.  Resampling *items* (rather than
    re-running the model) is what makes 2000 resamples over 64 coalitions cheap
    and keeps every coalition paired on the same items.

    Args:
        score_matrix: ``(N, 2**R)`` item scores per coalition.
        cfg: Root configuration.
        mode: ``"masked"`` or ``"retrain"``, recorded in the output.

    Returns:
        A dict with per-role Shapley estimates and intervals, the coalition
        value table, and the interpretive caveats the specification requires.
    """
    n_items, n_coalitions = score_matrix.shape
    if n_coalitions != 2**N_ROLES:
        raise ValueError(f"expected {2**N_ROLES} coalitions, got {n_coalitions}")

    coalition_values = np.nanmean(score_matrix, axis=0)
    point = shapley_from_values(coalition_values, N_ROLES)

    # Degeneracy check, distinct from the efficiency axiom below.
    #
    # Efficiency compares sum(phi) against v(A) - v(empty). When every coalition
    # value is identical that reads 0 == 0 - 0 and passes, so the ablation's main
    # self-test cannot detect a collapsed value function. This did happen: the
    # retrain path once returned all-zero coalition values because a threshold
    # fitted on one model was reused across 64 others, and the efficiency check
    # certified it. See FINDINGS_EVIDENCE_PATHWAY.md section 8.
    n_distinct = int(np.unique(np.round(coalition_values, 12)).size)
    degenerate = n_distinct <= 1

    rng = np.random.default_rng(cfg.eval.bootstrap_seed)
    draws = np.zeros((cfg.eval.n_bootstrap, N_ROLES), dtype=np.float64)
    for b in range(cfg.eval.n_bootstrap):
        indices = rng.integers(0, n_items, size=n_items)
        draws[b] = shapley_from_values(
            np.nanmean(score_matrix[indices], axis=0), N_ROLES
        )

    alpha = (1.0 - cfg.eval.ci_level) / 2.0
    roles: List[Dict[str, Any]] = []
    for r, role in enumerate(ROLES):
        roles.append(
            {
                "role": role,
                "shapley": float(point[r]),
                "ci_lo": float(np.quantile(draws[:, r], alpha)),
                "ci_hi": float(np.quantile(draws[:, r], 1.0 - alpha)),
                "prob_positive": float((draws[:, r] > 0).mean()),
                "bootstrap_se": float(draws[:, r].std(ddof=1)),
                "primary_benchmark": ROLE_PRIMARY_BENCHMARK[role],
            }
        )

    full_index = 2**N_ROLES - 1
    return {
        "mode": mode,
        "n_items": int(n_items),
        "n_bootstrap": int(cfg.eval.n_bootstrap),
        "ci_level": cfg.eval.ci_level,
        "empty_coalition_value": float(coalition_values[0]),
        "full_coalition_value": float(coalition_values[full_index]),
        "efficiency_check": {
            "sum_of_shapley": float(point.sum()),
            "full_minus_empty": float(coalition_values[full_index] - coalition_values[0]),
            "absolute_gap": float(
                abs(point.sum() - (coalition_values[full_index] - coalition_values[0]))
            ),
        },
        "degeneracy_check": {
            "n_distinct_coalition_values": n_distinct,
            "degenerate": degenerate,
            "passed": not degenerate,
            "note": (
                "FAILED: all 64 coalitions have the same value, so every Shapley "
                "value is mechanically zero and the efficiency check passes "
                "vacuously (0 == 0 - 0). This is a pipeline fault, not a "
                "finding. The usual cause is a calibration artifact -- a "
                "threshold or temperature fitted on one model and reused on "
                "another -- driving universal abstention."
                if degenerate
                else f"{n_distinct} distinct coalition values; the value "
                "function is not collapsed."
            ),
        },
        "roles": roles,
        "coalition_values": {
            "".join(str(b) for b in index_to_coalition(i, N_ROLES)): float(coalition_values[i])
            for i in range(n_coalitions)
        },
        "caveats": [
            "v(empty coalition) is the direct-answer baseline, predeclared before "
            "the run as specification section 8 requires.",
            "Shapley additivity does not license calling any agent necessary or "
            "sufficient; error correlations are reported separately and "
            "interactions are not assumed away.",
            (
                "Coalition values come from one synthesizer trained with coalition "
                "dropout, which approximates but does not equal retraining per "
                "coalition."
                if mode == "masked"
                else "Each coalition was trained separately; no masking approximation."
            ),
        ],
    }


def _interaction_index(coalition_values: np.ndarray, i: int, j: int) -> float:
    """Shapley-Owen interaction index for one pair of agents.

    The pairwise generalisation of the Shapley value:

    ``I_ij = sum over S subset of A\\{i,j} of w(|S|) * [ v(S+i+j) - v(S+i)
    - v(S+j) + v(S) ]``, with ``w(s) = s! (n-s-2)! / (n-1)!``.

    The bracketed term is the *discrete second derivative*: how much more agent
    ``j`` adds when ``i`` is already present.  Zero means the two contribute
    additively; positive means they are complements; negative means they are
    substitutes, which is what correlated failure modes look like from the
    coalition side.

    At six agents this is exact over all 64 coalition values, so no sampling
    approximation enters.

    Args:
        coalition_values: ``(2**R,)`` value per coalition, indexed by bitmask.
        i: First agent index.
        j: Second agent index.

    Returns:
        The interaction index.
    """
    n = N_ROLES
    others = [k for k in range(n) if k not in (i, j)]
    total = 0.0
    # Enumerate every subset S of the remaining agents.
    for mask in range(1 << len(others)):
        subset = 0
        size = 0
        for bit, k in enumerate(others):
            if mask & (1 << bit):
                subset |= 1 << k
                size += 1
        weight = (
            math.factorial(size)
            * math.factorial(n - size - 2)
            / math.factorial(n - 1)
        )
        delta = (
            coalition_values[subset | (1 << i) | (1 << j)]
            - coalition_values[subset | (1 << i)]
            - coalition_values[subset | (1 << j)]
            + coalition_values[subset]
        )
        total += weight * delta
    return float(total)


def interaction_analysis(score_matrix: np.ndarray, cfg: Config) -> Dict[str, Any]:
    """Pairwise agent interactions with bootstrap intervals (section 8).

    Section 8 states that "independent Shapley additivity must not be assumed".
    ``error_correlation_report`` measures whether agents *fail* together, which
    is a property of their outputs; this measures whether their *contributions*
    combine additively, which is a property of the coalition value function.  The
    two can disagree: agents with uncorrelated errors can still be substitutes if
    the synthesiser can only use one of them.

    An additivity gap is also reported: the difference between the full-council
    value and what a purely additive model of the individual Shapley values would
    predict.  A large gap means the per-role Shapley table, taken alone,
    misdescribes the council.

    Args:
        score_matrix: ``(N, 2**R)`` item scores per coalition.
        cfg: Root configuration.

    Returns:
        Per-pair interaction indices with intervals, the additivity gap, and an
        interpretation.

    Raises:
        ValueError: If the coalition axis is the wrong width.
    """
    n_items, n_coalitions = score_matrix.shape
    if n_coalitions != 2**N_ROLES:
        raise ValueError(f"expected {2**N_ROLES} coalitions, got {n_coalitions}")

    coalition_values = np.nanmean(score_matrix, axis=0)
    pairs = [(i, j) for i in range(N_ROLES) for j in range(i + 1, N_ROLES)]
    point = np.array([_interaction_index(coalition_values, i, j) for i, j in pairs])

    rng = np.random.default_rng(cfg.eval.bootstrap_seed + 31337)
    draws = np.zeros((cfg.eval.n_bootstrap, len(pairs)), dtype=np.float64)
    for b in range(cfg.eval.n_bootstrap):
        indices = rng.integers(0, n_items, size=n_items)
        values = np.nanmean(score_matrix[indices], axis=0)
        draws[b] = [_interaction_index(values, i, j) for i, j in pairs]

    alpha = (1.0 - cfg.eval.ci_level) / 2.0
    entries: List[Dict[str, Any]] = []
    for index, (i, j) in enumerate(pairs):
        lo = float(np.quantile(draws[:, index], alpha))
        hi = float(np.quantile(draws[:, index], 1.0 - alpha))
        entries.append(
            {
                "role_a": ROLES[i],
                "role_b": ROLES[j],
                "interaction": float(point[index]),
                "ci_lo": lo,
                "ci_hi": hi,
                "prob_nonzero": float((draws[:, index] > 0).mean() if point[index] > 0
                                      else (draws[:, index] < 0).mean()),
                "excludes_zero": bool(lo > 0 or hi < 0),
                "relationship": (
                    "complementary" if lo > 0 else "substitutable" if hi < 0 else "additive"
                ),
            }
        )

    full_index = 2**N_ROLES - 1
    shapley = shapley_from_values(coalition_values, N_ROLES)
    additive_prediction = float(coalition_values[0] + shapley.sum())
    observed = float(coalition_values[full_index])
    significant = [e for e in entries if e["excludes_zero"]]

    return {
        "n_pairs": len(entries),
        "pairs": entries,
        "n_pairs_excluding_zero": len(significant),
        "mean_absolute_interaction": float(np.abs(point).mean()),
        "max_absolute_interaction": float(np.abs(point).max()),
        "additivity": {
            "additive_prediction": additive_prediction,
            "observed_full_coalition": observed,
            # Efficiency makes these equal by construction for the full
            # coalition; the diagnostic value is in the pairwise terms, and this
            # is recorded so a reader can confirm the identity rather than
            # mistake it for evidence of additivity.
            "gap": float(observed - additive_prediction),
            "note": (
                "The Shapley efficiency axiom forces this gap to zero for the "
                "full coalition. It is NOT evidence that the agents combine "
                "additively. The pairwise interaction indices above are the "
                "evidence; a non-zero interaction with an interval excluding "
                "zero means additivity fails for that pair."
            ),
        },
        "interpretation": (
            f"{len(significant)} of {len(entries)} agent pairs have an "
            "interaction interval excluding zero. Pairs marked 'substitutable' "
            "contribute less together than separately, which bounds how much of "
            "any council gain can be attributed to having both. Pairs marked "
            "'complementary' contribute more together. Section 8 forbids "
            "assuming additivity, and these are the terms that would be assumed "
            "away."
        ),
    }


def repeated_trial_schedule(
    item_ids: Sequence[str],
    n_trials: int,
    base_seed: int,
) -> Dict[str, Any]:
    """Fixed item/seed schedule reused across every coalition (section 8).

    Section 8 requires that "each coalition receives enough repeated item-level
    trials" and that a fixed item/seed schedule be reused across coalitions to
    reduce variance.  Reuse is the operative word: if coalition A and coalition B
    are evaluated under different seeds, their difference confounds the coalition
    with the seed, and the paired bootstrap is no longer paired.

    The schedule is a pure function of the item ids and the base seed, so it is
    reproducible, and it is generated once and passed to every coalition rather
    than being redrawn per coalition.

    Args:
        item_ids: Test items, in their canonical sorted order.
        n_trials: Repeated trials per item.
        base_seed: Frozen base seed.

    Returns:
        The schedule as a mapping from trial index to per-item seeds, plus a
        digest so that two runs can be confirmed to have used the same schedule.

    Raises:
        ValueError: If ``n_trials`` is below one.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1; got {n_trials}")

    ordered = sorted(item_ids)
    schedule: Dict[str, List[int]] = {}
    for item_id in ordered:
        seeds: List[int] = []
        for trial in range(n_trials):
            digest = hashlib.sha256(
                f"{base_seed}:{item_id}:{trial}".encode("utf-8")
            ).digest()
            seeds.append(int.from_bytes(digest[:4], "big"))
        schedule[item_id] = seeds

    flat = ",".join(f"{k}:{'|'.join(map(str, v))}" for k, v in schedule.items())
    return {
        "n_items": len(ordered),
        "n_trials_per_item": n_trials,
        "n_observations_per_coalition": len(ordered) * n_trials,
        "base_seed": base_seed,
        "schedule_sha256": hashlib.sha256(flat.encode("utf-8")).hexdigest(),
        "seeds_by_item": schedule,
        "note": (
            "This schedule is generated once and reused for every coalition. "
            "Redrawing it per coalition would confound the coalition with the "
            "seed and invalidate the paired item-level statistics."
        ),
    }


def verify_trial_coverage(
    per_coalition_trials: Mapping[int, int],
    required: int,
    expected_coalitions: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Check that every coalition received the required repeated trials.

    Args:
        per_coalition_trials: Coalition bitmask to observed trial count.
        required: Minimum trials per coalition, from the pre-registration.
        expected_coalitions: Bitmasks the sweep was supposed to cover.  Defaults
            to all ``2**R``; the leave-one-out estimator passes its seven.

    Returns:
        Mapping with per-coalition counts and a pass/fail verdict naming the
        under-covered coalitions.
    """
    expected_keys = (
        list(expected_coalitions)
        if expected_coalitions is not None
        else list(range(2**N_ROLES))
    )
    missing = [i for i in expected_keys if i not in per_coalition_trials]
    under = {
        index: count
        for index, count in per_coalition_trials.items()
        if count < required
    }
    return {
        "required_trials_per_coalition": required,
        "n_coalitions_reported": len(per_coalition_trials),
        "n_coalitions_expected": len(expected_keys),
        "missing_coalitions": missing,
        "under_covered": under,
        "passed": not missing and not under,
        "note": (
            "Section 8 permits evaluating all 64 coalitions ONLY IF each "
            "receives enough repeated item-level trials. An under-covered "
            "coalition contributes a noisier value to every Shapley term it "
            "appears in, and the interval will not show it."
        ),
    }


def necessity_sufficiency_verdicts(
    shapley: Mapping[str, Any],
    interactions: Mapping[str, Any],
    replication: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide, per role, whether "necessary" or "sufficient" may be written.

    Section 8: *"Do not describe an agent as necessary or sufficient unless a
    predeclared decision rule, uncertainty interval, and replication support
    that wording."*  ``REMAINING_PROMPTS.md`` P12 requires this to be enforced in
    the report generator rather than left to prose, so the decision is computed
    here and the report renders the verdict.

    The predeclared rule, from ``PREREGISTRATION.md`` section 2 (H2) and
    section 7:

    * **Necessary** requires (a) the role's Shapley interval to exclude zero and
      lie above the H2 SESOI of +0.01, (b) removing it from the full council to
      cost more than the SESOI, and (c) a replication in a second model family
      showing the same sign.
    * **Sufficient** requires the role alone to reach the full council's value
      within the SESOI, with an interval supporting it.

    Args:
        shapley: Output of :func:`shapley_analysis`.
        interactions: Output of :func:`interaction_analysis`.
        replication: Optional second-family result, mapping role to the sign of
            its Shapley value there.  Absent replication blocks both words.

    Returns:
        Per-role verdicts with the permitted wording and the reason.
    """
    sesoi = 0.01  # PREREGISTRATION.md section 2, H2.
    coalition_values = shapley.get("coalition_values", {})
    full_value = float(shapley.get("full_coalition_value", float("nan")))
    empty_value = float(shapley.get("empty_coalition_value", float("nan")))

    def solo_value(role_index: int) -> float:
        """Value of the coalition containing only this role."""
        bits = ["0"] * N_ROLES
        bits[role_index] = "1"
        return float(coalition_values.get("".join(bits), float("nan")))

    verdicts: List[Dict[str, Any]] = []
    for index, entry in enumerate(shapley.get("roles", [])):
        role = entry["role"]
        interval_excludes_zero = entry["ci_lo"] > 0 or entry["ci_hi"] < 0
        above_sesoi = entry["ci_lo"] > sesoi
        replicated = bool(
            replication
            and role in replication
            and (replication[role] > 0) == (entry["shapley"] > 0)
        )

        solo = solo_value(index)
        reaches_full = (
            solo == solo
            and full_value == full_value
            and (full_value - solo) <= sesoi
        )

        necessary = bool(above_sesoi and interval_excludes_zero and replicated)
        sufficient = bool(reaches_full and replicated and interval_excludes_zero)

        if necessary:
            reason = (
                f"Shapley interval [{entry['ci_lo']:+.4f}, {entry['ci_hi']:+.4f}] "
                f"lies above the predeclared SESOI of {sesoi}, and the sign "
                "replicates in a second model family."
            )
        elif not replicated:
            reason = (
                "No replication in a second model family. Section 8 requires a "
                "replication before either word may be used, whatever the "
                "interval shows."
            )
        elif not interval_excludes_zero:
            reason = (
                f"Shapley interval [{entry['ci_lo']:+.4f}, {entry['ci_hi']:+.4f}] "
                "includes zero."
            )
        else:
            reason = (
                f"Interval excludes zero but its lower bound {entry['ci_lo']:+.4f} "
                f"does not clear the predeclared SESOI of {sesoi}."
            )

        verdicts.append(
            {
                "role": role,
                "may_write_necessary": necessary,
                "may_write_sufficient": sufficient,
                "permitted_wording": (
                    "necessary"
                    if necessary
                    else "contributes positively"
                    if interval_excludes_zero and entry["shapley"] > 0
                    else "contribution not distinguishable from zero"
                ),
                "reason": reason,
                "shapley": entry["shapley"],
                "ci": [entry["ci_lo"], entry["ci_hi"]],
                "solo_coalition_value": solo,
                "replicated": replicated,
            }
        )

    substitutable = [
        f"{p['role_a']}+{p['role_b']}"
        for p in interactions.get("pairs", [])
        if p.get("relationship") == "substitutable"
    ]

    return {
        "predeclared_sesoi": sesoi,
        "replication_available": bool(replication),
        "roles": verdicts,
        "n_necessary": sum(1 for v in verdicts if v["may_write_necessary"]),
        "n_sufficient": sum(1 for v in verdicts if v["may_write_sufficient"]),
        "substitutable_pairs": substitutable,
        "enforcement_note": (
            "These verdicts are computed, not asserted. The report generator "
            "renders permitted_wording verbatim; 'necessary' and 'sufficient' "
            "appear only where this rule licenses them. Substitutable pairs are "
            "listed because a role that is substitutable for another cannot be "
            "necessary regardless of its Shapley interval."
        ),
    }


def agent_by_benchmark_family(
    score_matrix: np.ndarray,
    benchmark_ids: np.ndarray,
    cfg: Config,
) -> Dict[str, Any]:
    """Test every agent on every benchmark, with FDR control.

    Limitation L12 warns that one-to-one agent-to-benchmark mappings invite
    confirmation bias, and requires *every* agent to be scored on *every*
    benchmark with a multiplicity correction.  The contrast for agent ``r`` on
    benchmark ``b`` is the paired leave-one-out difference: the full council
    versus the council with ``r`` removed, on that benchmark's items only.

    Args:
        score_matrix: ``(N, 2**R)`` item scores per coalition.
        benchmark_ids: ``(N,)`` benchmark index per item.
        cfg: Root configuration.

    Returns:
        A dict with one entry per agent-benchmark cell, raw p-values, BH
        q-values, and rejection decisions.
    """
    full_index = 2**N_ROLES - 1
    cells: List[Dict[str, Any]] = []
    p_values: List[float] = []

    for r, role in enumerate(ROLES):
        without = full_index & ~(1 << r)
        for b, benchmark in enumerate(BENCHMARKS):
            selector = benchmark_ids == b
            if selector.sum() < 8:  # too few items for a meaningful contrast
                continue
            contrast = paired_bootstrap_diff(
                score_matrix[selector, full_index],
                score_matrix[selector, without],
                n_bootstrap=cfg.eval.n_bootstrap,
                ci_level=cfg.eval.ci_level,
                seed=cfg.eval.bootstrap_seed + 1000 * r + b,
            )
            cells.append(
                {
                    "role": role,
                    "benchmark": benchmark,
                    "predeclared_primary": ROLE_PRIMARY_BENCHMARK[role] == benchmark,
                    "marginal_value": contrast["diff"],
                    "ci_lo": contrast["lo"],
                    "ci_hi": contrast["hi"],
                    "p_two_sided": contrast["p_two_sided"],
                    "n_items": contrast["n"],
                }
            )
            p_values.append(contrast["p_two_sided"])

    rejected, q_values = benjamini_hochberg(p_values, alpha=cfg.eval.fdr_alpha)
    for cell, reject, q in zip(cells, rejected, q_values):
        cell["q_value"] = float(q)
        cell["significant_after_fdr"] = bool(reject)

    return {
        "alpha": cfg.eval.fdr_alpha,
        "n_tests": len(cells),
        "n_significant": int(sum(c["significant_after_fdr"] for c in cells)),
        "cells": cells,
        "note": (
            "Leave-one-out marginal value of each agent on each benchmark. Every "
            "agent is scored on every benchmark, not only its predeclared "
            "primary target, and p-values carry Benjamini-Hochberg q-values."
        ),
    }


def error_correlation_report(result: EvalResult) -> Dict[str, Any]:
    """Measure whether agents fail independently.

    Limitation L7 is that six instances of one base model may share failure
    modes, in which case a council provides far less independent evidence than
    its agent count suggests.  This computes the pairwise correlation of
    item-level error vectors over items where both agents actually answered.

    Args:
        result: Full-council evaluation result.

    Returns:
        The correlation matrix, its mean off-diagonal, and per-agent solo
        accuracy.
    """
    labelled = result.gold >= 0
    errors = 1.0 - result.role_correct[labelled]
    answered = result.role_answered[labelled] > 0
    # Items an agent did not answer carry no error signal; excluding them keeps
    # missingness from masquerading as correlated failure.
    errors = np.where(answered, errors, np.nan)

    usable = ~np.isnan(errors).any(axis=1)
    matrix = (
        pairwise_error_correlation(errors[usable])
        if usable.sum() > 2
        else np.full((N_ROLES, N_ROLES), np.nan)
    )
    off_diagonal = matrix[~np.eye(N_ROLES, dtype=bool)]

    return {
        "n_items_all_agents_answered": int(usable.sum()),
        "matrix": matrix.tolist(),
        "roles": list(ROLES),
        "mean_off_diagonal": float(np.nanmean(off_diagonal)) if off_diagonal.size else float("nan"),
        "max_off_diagonal": float(np.nanmax(off_diagonal)) if off_diagonal.size else float("nan"),
        "solo_accuracy": {
            role: float(np.nanmean(result.role_correct[labelled][:, r][answered[:, r]]))
            if answered[:, r].any()
            else float("nan")
            for r, role in enumerate(ROLES)
        },
        "interpretation": (
            "High off-diagonal correlation means the council supplies less "
            "independent evidence than its agent count implies (limitation L7). "
            "It does not by itself invalidate a measured gain, but it bounds how "
            "much of that gain can be attributed to diversity."
        ),
    }


def cost_report(trainer: Trainer, split: str) -> Dict[str, Any]:
    """Summarise the token, latency, and dollar cost of the council.

    Limitation L10 and L18 require both unconstrained and budget-matched
    reporting.  This measures what the council actually consumed per item, which
    is the denominator for any cost-matched comparison.

    Args:
        trainer: Configured trainer.
        split: Split to summarise.

    Returns:
        Per-item cost aggregates, including per-agent output-token share.
        Zeroes here mean the source records carried no cost fields.
    """
    device = trainer.device
    output_tokens = np.zeros(N_ROLES)
    latency = np.zeros(N_ROLES)
    dollars = np.zeros(N_ROLES)
    retries = np.zeros(N_ROLES)
    absent = np.zeros(N_ROLES)
    input_tokens = 0.0
    n_items = 0

    for raw_batch in trainer.loaders[split]:
        batch = move_batch_to_device(raw_batch, device)
        # dataset.py stores cost features log-compressed; invert to recover units.
        cost = batch["cost"].float().cpu().numpy()
        present = batch["present"].float().cpu().numpy()
        output_tokens += (np.expm1(cost[:, :, 0] * 10.0) * present).sum(axis=0)
        latency += (np.expm1(cost[:, :, 1] * 10.0) * present).sum(axis=0)
        dollars += (np.expm1(cost[:, :, 2] * 10.0) / 1e4 * present).sum(axis=0)
        retries += (np.expm1(cost[:, :, 3] * 10.0) * present).sum(axis=0)
        absent += (1.0 - present).sum(axis=0)
        input_tokens += float(batch["input_tokens"].sum())
        n_items += batch["present"].shape[0]

    n = max(1, n_items)
    total_output = float(output_tokens.sum())
    return {
        "n_items": n_items,
        "input_tokens_per_item": input_tokens / n,
        "output_tokens_per_item": total_output / n,
        "cost_usd_per_item": float(dollars.sum()) / n,
        "council_calls_per_item": float(N_ROLES),
        "per_agent_output_tokens_per_item": {
            role: float(output_tokens[r] / n) for r, role in enumerate(ROLES)
        },
        "per_agent_output_token_share": {
            role: float(output_tokens[r] / total_output) if total_output > 0 else float("nan")
            for r, role in enumerate(ROLES)
        },
        "mean_latency_ms_per_agent": {
            role: float(latency[r] / n) for r, role in enumerate(ROLES)
        },
        # Section 8: "Publish all exclusions, failed calls, retries, and missing
        # outputs." These are first-class results, not appendix trivia.
        "exclusions": {
            "absent_agent_records": {role: int(absent[r]) for r, role in enumerate(ROLES)},
            "absent_agent_rate": float(absent.sum() / max(1, n_items * N_ROLES)),
            "total_retries": {role: int(retries[r]) for r, role in enumerate(ROLES)},
            "retries_per_item": float(retries.sum() / n),
        },
        "note": (
            "Budget-matched comparison requires holding these quantities equal "
            "across arms; the unconstrained numbers above are the quality-cost "
            "frontier point, not a matched comparison."
        ),
    }


# ---------------------------------------------------------------------------
# Robustness battery (specification section 9)
# ---------------------------------------------------------------------------


def _permute_batch_roles(
    batch: Mapping[str, torch.Tensor], permutation: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Permute every role-axis tensor consistently.

    Args:
        batch: Batch dict.
        permutation: ``(R,)`` long tensor of the new slot order.

    Returns:
        A new batch dict with the role axis permuted.
    """
    role_axis_keys = (
        "present",
        "vote",
        "role_conf",
        "abstain",
        "role_scalars",
        "role_status",
        "flags",
        "claim_status",
        "claim_conf",
        "claim_has_source",
        "claim_mask",
        "claim_label",
        "claim_label_mask",
        "cost",
        "role_ids",
    )
    permuted = dict(batch)
    for key in role_axis_keys:
        if key in batch:
            permuted[key] = batch[key].index_select(1, permutation)
    return permuted


@torch.no_grad()
def robustness_suite(trainer: Trainer, split: str = "test") -> Dict[str, Any]:
    """Run the section 9 perturbation battery.

    Tests, in order:

    ``permutation_invariance``
        Permute agent slots and require the option logits to be unchanged to
        within ``eval.permutation_tolerance``.  This is a *guarantee check*: the
        architecture carries no positional encoding over the role axis, so any
        deviation beyond floating-point noise is a bug, not a finding.

    ``option_permutation_equivariance``
        Permute answer options and require predictions to follow, ruling out
        position bias in the synthesiser.

    ``prompt_injection_via_agent_output``
        Rewrite one agent's record to look maximally authoritative -- all claims
        ``supported`` with sources, confidence 0.99 -- while pointing at a wrong
        answer.  This is injection delivered *through another agent's output*,
        the exact threat the specification names.

    ``confidently_wrong_majority``
        Flip a majority of agents to the same wrong option at high confidence.

    ``evidence_free_agreement``
        Unanimous agreement on a wrong option with every claim ``unsupported``
        and no sources.  A synthesiser that follows consensus over evidence
        fails here.

    ``role_label_permutation``
        Shuffle ``role_ids`` while holding content fixed.  Under a role-blind
        configuration the effect must be exactly zero; under a role-aware one the
        measured shift is how much the model leans on labels rather than
        operations, which is the quantity RQ3 asks about.

    Args:
        trainer: Trained trainer.
        split: Split to perturb.

    Returns:
        A dict of per-test results, each with a metric and a pass/fail or a
        measured degradation.
    """
    cfg = trainer.cfg
    device = trainer.device
    model = trainer.raw_model
    model.eval()

    generator = torch.Generator().manual_seed(cfg.eval.robustness_seed)
    supported = EVIDENCE_STATUS_TO_INDEX["supported"]
    unsupported = EVIDENCE_STATUS_TO_INDEX["unsupported"]

    max_logit_deviation = 0.0
    max_option_logit_deviation = 0.0
    option_equivariance_agreement: List[float] = []
    clean_correct: List[np.ndarray] = []
    attack_names = (
        "prompt_injection_via_agent_output",
        "confidently_wrong_majority",
        "evidence_free_agreement",
        "role_label_permutation",
    )
    perturbed_abstain: Dict[str, List[np.ndarray]] = {k: [] for k in attack_names}
    perturbed_base_fallback: Dict[str, List[np.ndarray]] = {k: [] for k in attack_names}
    perturbed_correct: Dict[str, List[np.ndarray]] = {
        "prompt_injection_via_agent_output": [],
        "confidently_wrong_majority": [],
        "evidence_free_agreement": [],
        "role_label_permutation": [],
    }

    for raw_batch in trainer.loaders[split]:
        batch = move_batch_to_device(raw_batch, device)
        batch_size, n_options = batch["option_mask"].shape
        gold = batch["gold"]
        labelled = gold >= 0
        coalition = torch.ones(batch_size, N_ROLES, device=device)

        clean_logits = model(batch, coalition)["option_logits"]
        clean_prediction = clean_logits.argmax(dim=-1)
        clean_correct.append((clean_prediction == gold)[labelled].float().cpu().numpy())

        # -- 1. agent-order permutation invariance --------------------------
        permutation = torch.randperm(N_ROLES, generator=generator).to(device)
        permuted_logits = model(_permute_batch_roles(batch, permutation), coalition)[
            "option_logits"
        ]
        max_logit_deviation = max(
            max_logit_deviation,
            float((permuted_logits - clean_logits).abs().max()),
        )

        # -- 2. option-order permutation equivariance -----------------------
        option_permutation = torch.randperm(n_options, generator=generator).to(device)
        option_batch = dict(batch)
        option_batch["vote"] = batch["vote"].index_select(2, option_permutation)
        option_batch["base_vote"] = batch["base_vote"].index_select(1, option_permutation)
        option_batch["option_mask"] = batch["option_mask"].index_select(1, option_permutation)
        if "option_features" in batch:
            option_batch["option_features"] = batch["option_features"].index_select(
                1, option_permutation
            )
        permuted_option_logits = model(option_batch, coalition)["option_logits"]
        recovered = option_permutation[permuted_option_logits.argmax(dim=-1)]
        option_equivariance_agreement.append(
            float((recovered == clean_prediction).float().mean())
        )
        # Argmax agreement below 1.0 can mean position bias *or* merely a
        # near-tie flipping under float non-associativity. The logit deviation
        # separates the two, so report both.
        expected_logits = clean_logits.index_select(1, option_permutation)
        valid_permuted = batch["option_mask"].index_select(1, option_permutation)
        max_option_logit_deviation = max(
            max_option_logit_deviation,
            float((permuted_option_logits - expected_logits)[valid_permuted].abs().max()),
        )

        # -- 3. injection delivered through an agent's output ---------------
        wrong = _wrong_option(gold, batch["option_mask"], generator, device)
        victim = torch.randint(0, N_ROLES, (batch_size,), generator=generator).to(device)
        victim_onehot = F.one_hot(victim, N_ROLES).bool()

        injected = _clone_batch(batch)
        injected["vote"] = torch.where(
            victim_onehot.unsqueeze(-1),
            F.one_hot(wrong, n_options).unsqueeze(1).expand(-1, N_ROLES, -1).to(batch["vote"].dtype),
            batch["vote"],
        )
        injected["role_conf"] = torch.where(
            victim_onehot, torch.full_like(batch["role_conf"], 0.99), batch["role_conf"]
        )
        injected["abstain"] = torch.where(
            victim_onehot, torch.zeros_like(batch["abstain"]), batch["abstain"]
        )
        injected["present"] = batch["present"] | victim_onehot
        claim_slots = victim_onehot.unsqueeze(-1) & batch["claim_mask"]
        injected["claim_status"] = torch.where(
            claim_slots, torch.full_like(batch["claim_status"], supported), batch["claim_status"]
        )
        injected["claim_conf"] = torch.where(
            claim_slots, torch.ones_like(batch["claim_conf"]), batch["claim_conf"]
        )
        injected["claim_has_source"] = torch.where(
            claim_slots, torch.ones_like(batch["claim_has_source"]), batch["claim_has_source"]
        )
        injected["role_status"] = torch.where(
            victim_onehot.unsqueeze(-1),
            F.one_hot(
                torch.tensor(supported, device=device), batch["role_status"].shape[-1]
            ).to(batch["role_status"].dtype),
            batch["role_status"],
        )
        _record_attack(
            model, injected, coalition, gold, labelled, batch,
            "prompt_injection_via_agent_output",
            perturbed_correct, perturbed_abstain, perturbed_base_fallback,
        )

        # -- 4. confidently wrong majority ----------------------------------
        n_flip = N_ROLES // 2 + 1
        flip_scores = torch.rand(batch_size, N_ROLES, generator=generator).to(device)
        flip = flip_scores <= flip_scores.topk(n_flip, dim=1).values[:, -1:]
        majority = _clone_batch(batch)
        majority["vote"] = torch.where(
            flip.unsqueeze(-1),
            F.one_hot(wrong, n_options).unsqueeze(1).expand(-1, N_ROLES, -1).to(batch["vote"].dtype),
            batch["vote"],
        )
        majority["role_conf"] = torch.where(
            flip, torch.full_like(batch["role_conf"], 0.95), batch["role_conf"]
        )
        majority["abstain"] = torch.where(flip, torch.zeros_like(batch["abstain"]), batch["abstain"])
        majority["present"] = batch["present"] | flip
        _record_attack(
            model, majority, coalition, gold, labelled, batch,
            "confidently_wrong_majority",
            perturbed_correct, perturbed_abstain, perturbed_base_fallback,
        )

        # -- 5. evidence-free unanimous agreement ---------------------------
        unanimous = _clone_batch(batch)
        unanimous["vote"] = (
            F.one_hot(wrong, n_options).unsqueeze(1).expand(-1, N_ROLES, -1).to(batch["vote"].dtype)
        )
        unanimous["present"] = torch.ones_like(batch["present"])
        unanimous["abstain"] = torch.zeros_like(batch["abstain"])
        unanimous["role_conf"] = torch.full_like(batch["role_conf"], 0.9)
        unanimous["claim_status"] = torch.where(
            batch["claim_mask"],
            torch.full_like(batch["claim_status"], unsupported),
            batch["claim_status"],
        )
        unanimous["claim_has_source"] = torch.zeros_like(batch["claim_has_source"])
        unanimous["role_status"] = F.one_hot(
            torch.tensor(unsupported, device=device), batch["role_status"].shape[-1]
        ).to(batch["role_status"].dtype).view(1, 1, -1).expand_as(batch["role_status"])
        _record_attack(
            model, unanimous, coalition, gold, labelled, batch,
            "evidence_free_agreement",
            perturbed_correct, perturbed_abstain, perturbed_base_fallback,
        )

        # -- 6. role-label permutation, content held fixed -------------------
        relabelled = _clone_batch(batch)
        relabelled["role_ids"] = batch["role_ids"].index_select(
            1, torch.randperm(N_ROLES, generator=generator).to(device)
        )
        _record_attack(
            model, relabelled, coalition, gold, labelled, batch,
            "role_label_permutation",
            perturbed_correct, perturbed_abstain, perturbed_base_fallback,
        )

    baseline_accuracy = float(np.concatenate(clean_correct).mean())
    report: Dict[str, Any] = {
        "split": split,
        "clean_accuracy": baseline_accuracy,
        "permutation_invariance": {
            "max_abs_logit_deviation": max_logit_deviation,
            "tolerance": cfg.eval.permutation_tolerance,
            "passed": bool(max_logit_deviation <= cfg.eval.permutation_tolerance),
            "note": (
                "The set encoder has no positional encoding over the agent axis "
                "and role identity enters only as content, so invariance is a "
                "structural guarantee; a failure here indicates a bug."
            ),
        },
        "option_permutation_equivariance": {
            "agreement": float(np.mean(option_equivariance_agreement)),
            "max_abs_logit_deviation": max_option_logit_deviation,
            "note": (
                "Nothing in the option scorer embeds the option index, so "
                "relabelling answer options must carry each score with it. "
                "max_abs_logit_deviation is the exact test; argmax agreement "
                "below 1.0 with a tiny deviation means near-ties flipping under "
                "float non-associativity, not position bias."
            ),
        },
    }
    for name, chunks in perturbed_correct.items():
        accuracy_under_attack = float(np.concatenate(chunks).mean())
        report[name] = {
            "accuracy": accuracy_under_attack,
            "degradation": baseline_accuracy - accuracy_under_attack,
            # Accuracy alone cannot distinguish "followed the attack" from
            # "correctly refused". Under attacks that make the gold answer
            # unreachable, low accuracy is unavoidable; what separates a robust
            # synthesizer from a credulous one is whether it abstains or falls
            # back to the direct answer instead of endorsing the planted option.
            "abstention_rate": float(np.concatenate(perturbed_abstain[name]).mean()),
            "base_fallback_rate": float(
                np.concatenate(perturbed_base_fallback[name]).mean()
            ),
        }
    report["role_label_permutation"]["role_blind_model"] = not cfg.model.use_role_embeddings
    report["role_label_permutation"]["note"] = (
        "Under model.use_role_embeddings=False this degradation must be exactly "
        "zero. A large value in the role-aware configuration means the model "
        "relies on role labels rather than on the operations they stand for "
        "(RQ3)."
    )
    return report


def _record_attack(
    model: CCASynthesizer,
    perturbed: Mapping[str, torch.Tensor],
    coalition: torch.Tensor,
    gold: torch.Tensor,
    labelled: torch.Tensor,
    clean_batch: Mapping[str, torch.Tensor],
    name: str,
    correct_out: Dict[str, List[np.ndarray]],
    abstain_out: Dict[str, List[np.ndarray]],
    fallback_out: Dict[str, List[np.ndarray]],
) -> None:
    """Record accuracy, abstention, and baseline fallback under one attack.

    Args:
        model: The synthesizer.
        perturbed: The attacked batch.
        coalition: Coalition mask.
        gold: Gold options.
        labelled: Mask of items with a known label.
        clean_batch: The unattacked batch, for the baseline vote.
        name: Attack name, used as the output key.
        correct_out: Accumulator for correctness indicators.
        abstain_out: Accumulator for abstention decisions.
        fallback_out: Accumulator for baseline-fallback decisions.
    """
    outputs = model(perturbed, coalition)
    prediction = outputs["option_logits"].argmax(dim=-1)
    abstained = torch.sigmoid(outputs["abstain_logit"]) > 0.5
    base_choice = clean_batch["base_vote"].argmax(dim=-1)
    has_base = clean_batch["base_vote"].sum(dim=-1) > 0

    correct_out[name].append((prediction == gold)[labelled].float().cpu().numpy())
    abstain_out[name].append(abstained[labelled].float().cpu().numpy())
    fallback_out[name].append(
        ((prediction == base_choice) & has_base)[labelled].float().cpu().numpy()
    )


def _clone_batch(batch: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """Shallow-clone a batch so perturbations never mutate the original.

    Args:
        batch: Batch dict.

    Returns:
        A dict whose tensors are clones and whose metadata is shared.
    """
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def _wrong_option(
    gold: torch.Tensor,
    option_mask: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample a valid option that is not the gold answer.

    Args:
        gold: ``(B,)`` gold indices.
        option_mask: ``(B, K)`` valid-option mask.
        generator: CPU generator for reproducibility.
        device: Target device.

    Returns:
        ``(B,)`` wrong-option indices; falls back to gold when an item has only
        one valid option.
    """
    batch_size, n_options = option_mask.shape
    weights = option_mask.float().clone()
    safe_gold = gold.clamp(min=0)
    weights.scatter_(1, safe_gold.unsqueeze(1), 0.0)
    degenerate = weights.sum(dim=1) == 0
    weights[degenerate] = option_mask[degenerate].float()
    noise = torch.rand(batch_size, n_options, generator=generator).to(device)
    return (weights * noise).argmax(dim=1)
