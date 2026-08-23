"""The calibrated synthesiser and its non-learned comparators.

Architecture, in one paragraph.  Each agent's atomic claims are embedded from
their evidence status, confidence, and source presence, then pooled by masked
attention into one claim summary.  That summary is concatenated with the agent's
bounded scalar features (and, optionally, a role embedding) to form a *candidate
token*.  The six candidate tokens plus one always-present ``[BASE]`` token --
which carries the direct-answer baseline and therefore gives the empty coalition
a defined value -- are processed by a permutation-equivariant transformer with
**no positional encoding** and an additive coalition mask.  Masked attention
pooling yields a council summary, from which four heads read out: a
recalibrated per-agent confidence, a score for each answer option, an abstention
probability, and a per-claim accept/reject decision with a discrete rule label.

Three properties are structural rather than incidental, and each answers a
specific requirement in ``/logistics``:

**Permutation invariance.**  No module indexes the role axis by position; role
identity enters only through a content embedding of ``role_ids``, and every
reduction over that axis is a masked sum or attention.  The output is therefore
invariant to any permutation of agent slots, which turns the "order and
role-label permutations" robustness test of specification section 9 into a
theorem plus a numerical check rather than an empirical hope.

**Option-index equivariance.**  Answer options enter only through the vote
matrix and pooled candidate representations; nothing embeds ``k``.  Scores are
therefore equivariant to relabelling options, which removes position bias from
the synthesiser by construction (an outcome the AdvConsistency row of section 5
lists as a secondary measure).

**Role blindness.**  Setting ``model.use_role_embeddings=False`` deletes the
only path by which role identity reaches the network, implementing section
7.1's "normalize claims without seeing condition names or clinical labels" and
providing the ablation that RQ3 requires.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ADJUDICATION_RULES,
    Config,
    EVIDENCE_PAD_INDEX,
    EVIDENCE_STATUSES,
    FAILURE_FLAGS,
    ModelConfig,
    N_ROLES,
)
from dataset import N_COST_FEATURES, N_ROLE_SCALARS

__all__ = [
    "FourierScalarEncoder",
    "ClaimEncoder",
    "MaskedAttentionPool",
    "SetEncoderLayer",
    "SetEncoder",
    "ConfidenceRecalibrator",
    "OptionScorer",
    "CCASynthesizer",
    "build_model",
    "effective_agent_mask",
    "majority_vote_scores",
    "confidence_weighted_vote_scores",
    "base_answer_scores",
    "single_agent_scores",
    "oracle_upper_bound_scores",
    "BASELINES",
]


#: Smallest temperature post-hoc calibration may fit.  The masking sentinel is
#: sized against this so that ``sentinel / MIN_TEMPERATURE`` never overflows.
#: Kept here rather than in engine.py because the sentinel depends on it.
MIN_TEMPERATURE: float = 0.05


def _mask_fill_value(dtype: torch.dtype) -> float:
    """Return a safe additive mask value for ``dtype``.

    Three constraints, and the third is easy to get wrong.  The value must be
    negative enough to zero a softmax; it must be finite, because ``-inf``
    interacts badly with autocast and produces ``nan`` gradients through masked
    entries; and it must be small enough in magnitude to survive later
    *arithmetic*.  An earlier version used ``finfo.min / 4`` (about -8.5e37 in
    float32), which satisfies the first two and fails the third: post-hoc
    temperature scaling divides logits by ``T``, and once ``T`` drops below 1
    that sentinel overflows to ``-inf``, sending the temperature-fitting
    gradient to ``nan`` and poisoning every downstream prediction.

    The bound is therefore derived from the temperature floor rather than
    guessed: the sentinel must satisfy ``|s| / MIN_TEMPERATURE < finfo.max`` with
    margin.  This matters most in float16, whose maximum is only 65504 — a
    sentinel chosen for float32 overflows there, which is a CUDA-only failure
    invisible on MPS or CPU.

    Args:
        dtype: Tensor dtype the mask will be added to.

    Returns:
        A large negative finite value safe to divide by ``MIN_TEMPERATURE``.
    """
    headroom = torch.finfo(dtype).max * MIN_TEMPERATURE / 4.0
    return -min(1e9, headroom)


#: Sentinel used when padding logits outside the model, kept in sync with
#: :func:`_mask_fill_value` for float32.
MASK_FILL_FLOAT32: float = _mask_fill_value(torch.float32)


def _truncated_normal_(tensor: torch.Tensor, std: float) -> torch.Tensor:
    """In-place truncated-normal initialisation at +/- 2 standard deviations.

    Args:
        tensor: Parameter to initialise.
        std: Standard deviation.

    Returns:
        The same tensor, initialised.
    """
    with torch.no_grad():
        tensor.normal_(0.0, std)
        tensor.clamp_(-2.0 * std, 2.0 * std)
    return tensor


class FourierScalarEncoder(nn.Module):
    """Lift bounded scalars into a smoother feature space.

    Each scalar ``x`` becomes ``[x, sin(2^l pi x), cos(2^l pi x)]`` for
    ``l = 0..L-1``.  A single linear layer applied to raw scalars struggles to
    represent thresholds like "confidence above 0.9"; the sinusoidal lift makes
    such decision boundaries reachable at negligible parameter cost, which
    matters here because per-agent inputs are only a handful of numbers.
    """

    def __init__(self, n_frequencies: int = 4):
        """Initialise the encoder.

        Args:
            n_frequencies: Number of octaves ``L``.
        """
        super().__init__()
        self.n_frequencies = int(n_frequencies)
        frequencies = 2.0 ** torch.arange(self.n_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies * math.pi, persistent=False)

    @property
    def expansion(self) -> int:
        """Output features produced per input scalar."""
        return 1 + 2 * self.n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a tensor of scalars.

        Args:
            x: Tensor of shape ``(..., S)`` with entries nominally in ``[0, 1]``.

        Returns:
            Tensor of shape ``(..., S * expansion)``.
        """
        if self.n_frequencies == 0:
            return x
        scaled = x.unsqueeze(-1) * self.frequencies.to(x.dtype)
        features = torch.cat([x.unsqueeze(-1), scaled.sin(), scaled.cos()], dim=-1)
        return features.flatten(start_dim=-2)


class ClaimEncoder(nn.Module):
    """Embed atomic claims and pool them into one vector per agent.

    A learned null-claim token is prepended to every agent's claim sequence, so
    an agent that produced no claims still yields a well-defined summary and the
    attention softmax never sees an all-masked row.
    """

    def __init__(self, cfg: ModelConfig):
        """Build the encoder.

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        self.scalar_encoder = FourierScalarEncoder(cfg.n_scalar_frequencies)

        # +1 for the padding class, which is embedded but always masked out.
        self.status_embedding = nn.Embedding(len(EVIDENCE_STATUSES) + 1, d_model)
        self.scalar_projection = nn.Linear(2 * self.scalar_encoder.expansion, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.null_claim = nn.Parameter(torch.zeros(d_model))

        self.n_heads = cfg.n_heads
        self.head_dim = d_model // cfg.n_heads
        self.query = nn.Parameter(torch.zeros(cfg.n_heads, self.head_dim))
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def encode_claims(
        self,
        claim_status: torch.Tensor,
        claim_conf: torch.Tensor,
        claim_has_source: torch.Tensor,
    ) -> torch.Tensor:
        """Embed each claim independently.

        Args:
            claim_status: Long tensor ``(B, R, C)`` of evidence-status ids.
            claim_conf: Float tensor ``(B, R, C)`` of per-claim confidences.
            claim_has_source: Float tensor ``(B, R, C)`` of source indicators.

        Returns:
            Float tensor ``(B, R, C, D)``.
        """
        status = self.status_embedding(claim_status.clamp(max=EVIDENCE_PAD_INDEX))
        scalars = torch.stack([claim_conf, claim_has_source], dim=-1)
        projected = self.scalar_projection(self.scalar_encoder(scalars))
        return self.norm(status + projected)

    def forward(
        self,
        claim_status: torch.Tensor,
        claim_conf: torch.Tensor,
        claim_has_source: torch.Tensor,
        claim_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and pool claims.

        Args:
            claim_status: ``(B, R, C)`` long.
            claim_conf: ``(B, R, C)`` float.
            claim_has_source: ``(B, R, C)`` float.
            claim_mask: ``(B, R, C)`` bool, ``True`` for real claims.

        Returns:
            Tuple ``(pooled, per_claim, attention)`` with shapes
            ``(B, R, D)``, ``(B, R, C, D)``, and ``(B, R, H, C+1)``.
        """
        batch, roles, n_claims = claim_status.shape
        per_claim = self.encode_claims(claim_status, claim_conf, claim_has_source)

        null = self.null_claim.to(per_claim.dtype).view(1, 1, 1, -1).expand(
            batch, roles, 1, -1
        )
        tokens = torch.cat([null, per_claim], dim=2)
        mask = torch.cat(
            [torch.ones(batch, roles, 1, dtype=torch.bool, device=claim_mask.device), claim_mask],
            dim=2,
        )

        if self.cfg.claim_pool == "mean":
            weights = mask.to(tokens.dtype)
            pooled = (tokens * weights.unsqueeze(-1)).sum(dim=2) / weights.sum(
                dim=2, keepdim=True
            ).clamp(min=1.0)
            attention = weights.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
            return self.output_projection(pooled), per_claim, attention

        keys = self.key_projection(tokens).view(
            batch, roles, n_claims + 1, self.n_heads, self.head_dim
        )
        values = self.value_projection(tokens).view(
            batch, roles, n_claims + 1, self.n_heads, self.head_dim
        )
        query = self.query.to(tokens.dtype)
        scores = torch.einsum("hd,brchd->brhc", query, keys) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~mask.unsqueeze(2), _mask_fill_value(scores.dtype)
        )
        attention = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("brhc,brchd->brhd", self.dropout(attention), values)
        pooled = pooled.reshape(batch, roles, self.cfg.d_model)
        return self.output_projection(pooled), per_claim, attention


class MaskedAttentionPool(nn.Module):
    """Multi-head attention pooling with a learned query and a key mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        """Build the pooler.

        Args:
            d_model: Token width.
            n_heads: Attention heads.
            dropout: Dropout on attention weights.
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool a masked token set.

        Args:
            tokens: ``(B, T, D)``.
            mask: ``(B, T)`` bool, ``True`` for valid tokens.

        Returns:
            Tuple ``(pooled, attention)`` with shapes ``(B, D)`` and
            ``(B, H, T)``.
        """
        batch, n_tokens, _ = tokens.shape
        keys = self.key_projection(tokens).view(batch, n_tokens, self.n_heads, self.head_dim)
        values = self.value_projection(tokens).view(
            batch, n_tokens, self.n_heads, self.head_dim
        )
        scores = torch.einsum("hd,bthd->bht", self.query.to(tokens.dtype), keys)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask.unsqueeze(1), _mask_fill_value(scores.dtype))
        attention = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bht,bthd->bhd", self.dropout(attention), values)
        return self.output_projection(pooled.reshape(batch, -1)), attention


class SetEncoderLayer(nn.Module):
    """One pre-LayerNorm transformer block over an unordered token set.

    Deliberately carries no positional information: permutation equivariance is
    a required property of this model, not an inconvenience to work around.
    """

    def __init__(self, cfg: ModelConfig):
        """Build the layer.

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = d_model // cfg.n_heads
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attention_out = nn.Linear(d_model, d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, cfg.d_ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff_mult * d_model, d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masked self-attention and a feed-forward block.

        Args:
            tokens: ``(B, T, D)``.
            mask: ``(B, T)`` bool key mask.

        Returns:
            Updated tokens, ``(B, T, D)``.
        """
        batch, n_tokens, d_model = tokens.shape
        normed = self.norm_attention(tokens)
        qkv = self.qkv(normed).view(batch, n_tokens, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~mask.view(batch, 1, 1, n_tokens), _mask_fill_value(scores.dtype)
        )
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, n_tokens, d_model)
        tokens = tokens + self.dropout(self.attention_out(attended))
        tokens = tokens + self.dropout(self.feedforward(self.norm_feedforward(tokens)))
        # Masked-out tokens carry no information downstream; zeroing them keeps
        # any accidental unmasked reduction from silently reading stale state.
        return tokens * mask.unsqueeze(-1).to(tokens.dtype)


class SetEncoder(nn.Module):
    """Stack of :class:`SetEncoderLayer` with a final LayerNorm."""

    def __init__(self, cfg: ModelConfig):
        """Build the encoder.

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        self.layers = nn.ModuleList([SetEncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the token set.

        Args:
            tokens: ``(B, T, D)``.
            mask: ``(B, T)`` bool key mask.

        Returns:
            Encoded tokens, ``(B, T, D)``.
        """
        for layer in self.layers:
            tokens = layer(tokens, mask)
        return self.norm(tokens)


class ConfidenceRecalibrator(nn.Module):
    """Map declared agent confidence to a calibrated probability.

    Implements section 7.3 -- "treat agent confidence as an input feature, not
    ground truth" -- as an explicit, inspectable transform rather than an
    implicit hope.  The affine acts in logit space (Platt scaling) and is
    corrected by a content term read from the contextualised candidate token, so
    an agent that is overconfident *specifically when its claims are
    unsupported* can be corrected conditionally rather than uniformly.
    """

    def __init__(self, cfg: ModelConfig):
        """Build the recalibrator.

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        self.shared = cfg.share_calibration_across_roles
        n_slots = 1 if self.shared else N_ROLES
        self.log_scale = nn.Parameter(torch.zeros(n_slots))
        self.bias = nn.Parameter(torch.zeros(n_slots))
        self.content = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(
        self, declared: torch.Tensor, tokens: torch.Tensor, role_ids: torch.Tensor
    ) -> torch.Tensor:
        """Recalibrate declared confidences.

        Args:
            declared: ``(B, R)`` declared confidences in ``[0, 1]``.
            tokens: ``(B, R, D)`` contextualised candidate tokens.
            role_ids: ``(B, R)`` role identifiers.

        Returns:
            ``(B, R)`` recalibrated probabilities.
        """
        clamped = declared.clamp(1e-4, 1.0 - 1e-4)
        logits = torch.log(clamped) - torch.log1p(-clamped)
        if self.shared:
            scale = self.log_scale[0].exp()
            bias = self.bias[0]
        else:
            scale = self.log_scale[role_ids].exp()
            bias = self.bias[role_ids]
        adjusted = scale * logits + bias + self.content(tokens).squeeze(-1)
        return torch.sigmoid(adjusted)


class OptionScorer(nn.Module):
    """Score each answer option from the agents that endorsed it.

    Nothing in this module depends on the integer ``k``: an option is
    represented purely by the (evidence-weighted) mean of the contextualised
    candidate tokens that voted for it, the evidence profile of those agents,
    their recalibrated confidence mass, and whether the direct-answer baseline
    chose it.  Options with no support fall back to a learned vector, which is
    what lets the empty coalition still select the baseline answer.

    **Evidence gating.**  Votes are aggregated with weight
    ``floor + (1 - floor) * credibility_r``, where ``credibility_r`` is the mean
    adjudicated accept probability over agent ``r``'s claims.  An agent whose
    claims are all rejected therefore contributes ``floor`` of the weight of an
    agent whose claims hold up.  This exists because the first version of this
    model scored the claim head as a detached side task: with no gradient path
    from evidence to the answer, the synthesizer learned to be a pure vote
    aggregator, and unanimous agreement on a wrong option with every claim
    marked ``unsupported`` still drove accuracy to 0.0099.  Gating is structural
    rather than learned so the model cannot simply route around it; the raw
    unweighted support count is retained as a separate channel so vote counting
    remains available where it is genuinely informative.
    """

    def __init__(self, cfg: ModelConfig, option_feature_dim: int = 0):
        """Build the scorer.

        Args:
            cfg: Model configuration.
            option_feature_dim: Width of optional precomputed option embeddings.
        """
        super().__init__()
        d_model = cfg.d_model
        self.evidence_gate_floor = float(cfg.evidence_gate_floor)
        self.scalar_encoder = FourierScalarEncoder(cfg.n_scalar_frequencies)
        # raw support fraction, evidence-weighted support, mean supporter
        # credibility, confidence mass, baseline vote, unsupported flag, plus
        # the four-way evidence-status simplex of the endorsing agents.
        self.n_option_scalars = 6 + len(EVIDENCE_STATUSES)
        self.value_projection = nn.Linear(d_model, d_model)
        self.null_option = nn.Parameter(torch.zeros(d_model))
        self.scalar_projection = nn.Linear(
            self.n_option_scalars * self.scalar_encoder.expansion, d_model
        )
        self.option_feature_projection = (
            nn.Linear(option_feature_dim, d_model) if option_feature_dim > 0 else None
        )
        self.score = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        summary: torch.Tensor,
        vote: torch.Tensor,
        active: torch.Tensor,
        recalibrated: torch.Tensor,
        role_status: torch.Tensor,
        base_vote: torch.Tensor,
        option_mask: torch.Tensor,
        evidence_credibility: torch.Tensor,
        option_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute one logit per answer option.

        Args:
            tokens: ``(B, R, D)`` contextualised candidate tokens.
            summary: ``(B, D)`` council summary.
            vote: ``(B, R, K)`` one-hot votes.
            active: ``(B, R)`` float mask of present-and-in-coalition agents.
            recalibrated: ``(B, R)`` recalibrated confidences.
            role_status: ``(B, R, 4)`` per-agent evidence-status simplex.
            base_vote: ``(B, K)`` direct-answer baseline one-hot.
            option_mask: ``(B, K)`` bool mask of valid options.
            evidence_credibility: ``(B, R)`` mean adjudicated accept probability
                per agent, in ``[0, 1]``.
            option_features: Optional ``(B, K, d)`` precomputed embeddings.

        Returns:
            ``(B, K)`` logits, with invalid options driven to the dtype minimum.
        """
        floor = self.evidence_gate_floor
        gate = floor + (1.0 - floor) * evidence_credibility.clamp(0.0, 1.0)

        raw_vote = vote * active.unsqueeze(-1)
        effective_vote = raw_vote * gate.unsqueeze(-1)
        support = raw_vote.sum(dim=1)
        weighted_support = effective_vote.sum(dim=1)
        denominator = weighted_support.clamp(min=1e-4).unsqueeze(-1)

        values = self.value_projection(tokens)
        option_embedding = torch.einsum("brk,brd->bkd", effective_vote, values) / denominator
        has_support = (support > 0).unsqueeze(-1)
        option_embedding = torch.where(
            has_support, option_embedding, self.null_option.to(values.dtype)
        )
        if self.option_feature_projection is not None and option_features is not None:
            option_embedding = option_embedding + self.option_feature_projection(option_features)

        status = torch.einsum("brk,brs->bks", effective_vote, role_status) / denominator
        confidence_mass = torch.einsum("brk,br->bk", raw_vote, recalibrated)
        # Mean credibility of the agents backing this option, which is the
        # signal that lets the scorer discount a well-supported-looking option
        # endorsed only by agents whose claims were rejected.
        supporter_credibility = torch.einsum("brk,br->bk", raw_vote, gate) / support.clamp(
            min=1.0
        )
        n_active = active.sum(dim=1, keepdim=True).clamp(min=1.0)

        scalars = torch.cat(
            [
                (support / n_active).unsqueeze(-1),
                (weighted_support / n_active).unsqueeze(-1),
                supporter_credibility.unsqueeze(-1),
                (confidence_mass / n_active).unsqueeze(-1),
                base_vote.unsqueeze(-1),
                (~has_support).to(values.dtype),
                status,
            ],
            dim=-1,
        )
        scalar_embedding = self.scalar_projection(self.scalar_encoder(scalars))

        expanded = summary.unsqueeze(1).expand_as(option_embedding)
        features = torch.cat(
            [option_embedding, expanded, option_embedding * expanded, scalar_embedding],
            dim=-1,
        )
        logits = self.score(features).squeeze(-1)
        return logits.masked_fill(~option_mask, _mask_fill_value(logits.dtype))


class CCASynthesizer(nn.Module):
    """Coalition-masked, permutation-invariant calibrated synthesiser."""

    def __init__(self, cfg: Config):
        """Build the model.

        Args:
            cfg: Root configuration.
        """
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model
        d_model = model_cfg.d_model

        self.claim_encoder = ClaimEncoder(model_cfg)
        self.scalar_encoder = FourierScalarEncoder(model_cfg.n_scalar_frequencies)

        n_scalars = N_ROLE_SCALARS + (N_COST_FEATURES if model_cfg.use_cost_features else 0)
        self.candidate_scalar_projection = nn.Linear(
            n_scalars * self.scalar_encoder.expansion, d_model
        )
        self.flag_projection = nn.Linear(len(FAILURE_FLAGS), d_model)
        self.claim_projection = nn.Linear(d_model, d_model)
        self.status_projection = nn.Linear(len(EVIDENCE_STATUSES), d_model)
        self.candidate_norm = nn.LayerNorm(d_model)

        self.role_embedding = (
            nn.Embedding(N_ROLES, d_model) if model_cfg.use_role_embeddings else None
        )
        # The [BASE] token is always unmasked.  It carries the direct-answer
        # baseline, so v(empty coalition) is architecturally the base model's
        # answer -- the predeclared null policy of specification section 8 --
        # and every attention softmax has at least one valid key.
        self.base_token = nn.Parameter(torch.zeros(d_model))
        # Two features: whether the baseline produced an answer at all, and the
        # option-set size (which sets the chance level the council must beat).
        self.base_projection = nn.Linear(2, d_model)

        self.encoder = SetEncoder(model_cfg)
        self.pool = MaskedAttentionPool(d_model, model_cfg.n_heads, model_cfg.dropout)
        self.recalibrator = ConfidenceRecalibrator(model_cfg)
        self.option_scorer = OptionScorer(model_cfg, cfg.data.option_feature_dim)

        self.abstention_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.claim_head = (
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(model_cfg.dropout),
                nn.Linear(d_model, 1),
            )
            if model_cfg.predict_claims
            else None
        )
        # Neutral credibility prior for an agent that produced no claims at all.
        # Initialised at logit(0.5) = 0 so it starts as "no opinion".
        self.no_claim_credibility = nn.Parameter(torch.zeros(1))
        self.rule_head = (
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, len(ADJUDICATION_RULES)),
            )
            if model_cfg.predict_rules
            else None
        )

        # Post-hoc temperature, fitted on the calibration split after training.
        self.register_buffer("logit_temperature", torch.ones(1))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise linear and embedding weights.

        Args:
            module: Submodule visited by :meth:`torch.nn.Module.apply`.
        """
        std = self.cfg.model.init_std
        if isinstance(module, nn.Linear):
            _truncated_normal_(module.weight, std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            _truncated_normal_(module.weight, std)
        elif isinstance(module, (ClaimEncoder, MaskedAttentionPool)):
            _truncated_normal_(module.query, std)

    def build_candidate_tokens(
        self, batch: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble one token per agent slot.

        Args:
            batch: Batch dict from :func:`dataset.collate_records`.

        Returns:
            Tuple ``(tokens, per_claim, claim_attention)`` with shapes
            ``(B, R, D)``, ``(B, R, C, D)``, ``(B, R, H, C+1)``.
        """
        claim_summary, per_claim, claim_attention = self.claim_encoder(
            batch["claim_status"],
            batch["claim_conf"],
            batch["claim_has_source"],
            batch["claim_mask"],
        )

        scalars = batch["role_scalars"]
        if self.cfg.model.use_cost_features:
            scalars = torch.cat([scalars, batch["cost"]], dim=-1)

        tokens = (
            self.claim_projection(claim_summary)
            + self.candidate_scalar_projection(self.scalar_encoder(scalars))
            + self.flag_projection(batch["flags"])
            + self.status_projection(batch["role_status"])
        )
        if self.role_embedding is not None:
            tokens = tokens + self.role_embedding(batch["role_ids"])
        return self.candidate_norm(tokens), per_claim, claim_attention

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        coalition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the synthesiser.

        Args:
            batch: Batch dict from :func:`dataset.collate_records`.
            coalition: ``(B, R)`` float mask; ``None`` means the full council.

        Returns:
            Dict with keys ``option_logits`` ``(B, K)``, ``abstain_logit``
            ``(B,)``, ``recalibrated_confidence`` ``(B, R)``, ``pool_attention``
            ``(B, H, R+1)``, ``dominance`` ``(B,)``, ``summary`` ``(B, D)``,
            ``active`` ``(B, R)``, and -- when the corresponding heads are
            enabled -- ``claim_accept_logit`` ``(B, R, C)`` and
            ``claim_rule_logits`` ``(B, R, C, n_rules)``.
        """
        present = batch["present"]
        batch_size = present.shape[0]
        if coalition is None:
            coalition = torch.ones_like(present, dtype=torch.float32)
        active = effective_agent_mask(present, coalition)

        candidate_tokens, per_claim, claim_attention = self.build_candidate_tokens(batch)

        base_features = torch.stack(
            [batch["base_vote"].sum(dim=-1), batch["n_options"].to(candidate_tokens.dtype) / 8.0],
            dim=-1,
        )
        base = self.base_token.to(candidate_tokens.dtype).unsqueeze(0) + self.base_projection(
            base_features.to(candidate_tokens.dtype)
        )
        tokens = torch.cat([base.unsqueeze(1), candidate_tokens], dim=1)
        token_mask = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=torch.bool, device=present.device),
                active > 0,
            ],
            dim=1,
        )

        encoded = self.encoder(tokens, token_mask)
        summary, pool_attention = self.pool(encoded, token_mask)
        role_tokens = encoded[:, 1:]

        recalibrated = self.recalibrator(
            batch["role_conf"], role_tokens, batch["role_ids"]
        ) * active

        # Claim adjudication runs *before* the answer head, not after it.  In
        # the first version this block sat at the end of forward(), so no
        # gradient path connected adjudicated evidence to the answer decision
        # and the model learned to ignore claims entirely.  Its output now gates
        # vote aggregation, which means the answer loss trains the claim head.
        claim_accept_logit: Optional[torch.Tensor] = None
        if self.claim_head is not None or self.rule_head is not None:
            context = summary.unsqueeze(1).unsqueeze(1).expand(
                -1, per_claim.shape[1], per_claim.shape[2], -1
            )
            claim_joint = torch.cat([per_claim, context], dim=-1)
            if self.claim_head is not None:
                claim_accept_logit = self.claim_head(claim_joint).squeeze(-1)

        evidence_credibility = self.evidence_credibility(
            claim_accept_logit, batch["claim_mask"]
        )

        option_logits = self.option_scorer(
            tokens=role_tokens,
            summary=summary,
            vote=batch["vote"],
            active=active,
            recalibrated=recalibrated,
            role_status=batch["role_status"],
            base_vote=batch["base_vote"],
            option_mask=batch["option_mask"],
            evidence_credibility=evidence_credibility,
            option_features=batch.get("option_features"),
        )

        # Dominance: the largest share of pooling mass any single *agent* holds,
        # renormalised over agents only.  This is the quantity the section 9
        # "synthesizer capture by one verbose agent" test probes; regularising it
        # during training is cheaper than discovering capture afterwards.
        agent_attention = pool_attention[:, :, 1:] * (active > 0).unsqueeze(1).to(
            pool_attention.dtype
        )
        agent_mass = agent_attention.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        dominance = (agent_attention / agent_mass).max(dim=-1).values.mean(dim=1)

        outputs: Dict[str, torch.Tensor] = {
            "option_logits": option_logits,
            "abstain_logit": self.abstention_head(summary).squeeze(-1),
            "recalibrated_confidence": recalibrated,
            "pool_attention": pool_attention,
            "claim_attention": claim_attention,
            "dominance": dominance,
            "summary": summary,
            "active": active,
            "evidence_credibility": evidence_credibility,
        }
        if claim_accept_logit is not None:
            outputs["claim_accept_logit"] = claim_accept_logit
        if self.rule_head is not None:
            outputs["claim_rule_logits"] = self.rule_head(claim_joint)
        return outputs

    def evidence_credibility(
        self, claim_accept_logit: Optional[torch.Tensor], claim_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean adjudicated accept probability per agent.

        Agents that produced no claims cannot be scored on their evidence, so
        they receive a learned neutral prior rather than a zero that would
        silently penalise a terse-but-correct agent.

        Args:
            claim_accept_logit: ``(B, R, C)`` accept logits, or ``None`` when the
                claim head is disabled.
            claim_mask: ``(B, R, C)`` bool mask of real claims.

        Returns:
            ``(B, R)`` credibility in ``[0, 1]``; all ones when the head is off,
            which disables gating and restores vote-aggregator behaviour.
        """
        if claim_accept_logit is None:
            return torch.ones(
                claim_mask.shape[:2], device=claim_mask.device, dtype=torch.float32
            )
        weights = claim_mask.to(claim_accept_logit.dtype)
        counts = weights.sum(dim=-1)
        accepted = (torch.sigmoid(claim_accept_logit) * weights).sum(dim=-1)
        mean_accepted = accepted / counts.clamp(min=1.0)
        neutral = torch.sigmoid(self.no_claim_credibility).to(mean_accepted.dtype)
        return torch.where(counts > 0, mean_accepted, neutral.expand_as(mean_accepted))

    @torch.no_grad()
    def predict(
        self,
        batch: Mapping[str, torch.Tensor],
        coalition: Optional[torch.Tensor] = None,
        abstain_threshold: Optional[float] = None,
        apply_temperature: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Produce deployable predictions with calibration applied.

        Args:
            batch: Batch dict.
            coalition: ``(B, R)`` coalition mask.
            abstain_threshold: Selection threshold fitted on the calibration
                split; ``None`` disables threshold-based abstention and leaves
                only the learned abstention head.
            apply_temperature: Divide logits by the fitted temperature.

        Returns:
            Dict with ``prediction`` ``(B,)``, ``probabilities`` ``(B, K)``,
            ``confidence`` ``(B,)``, ``abstain`` ``(B,)`` bool, and
            ``selection_score`` ``(B,)``.
        """
        outputs = self.forward(batch, coalition)
        logits = outputs["option_logits"]
        if apply_temperature:
            logits = logits / self.logit_temperature.clamp(min=1e-3).to(logits.dtype)
        probabilities = torch.softmax(logits, dim=-1)
        confidence, prediction = probabilities.max(dim=-1)

        abstain_probability = torch.sigmoid(outputs["abstain_logit"])
        selection_score = confidence * (1.0 - abstain_probability)
        abstain = abstain_probability > 0.5
        if abstain_threshold is not None:
            abstain = abstain | (selection_score < abstain_threshold)

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "confidence": confidence,
            "abstain_probability": abstain_probability,
            "abstain": abstain,
            "selection_score": selection_score,
            "recalibrated_confidence": outputs["recalibrated_confidence"],
            "dominance": outputs["dominance"],
            "active": outputs["active"],
            "claim_accept_logit": outputs.get("claim_accept_logit"),
            "claim_rule_logits": outputs.get("claim_rule_logits"),
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters.

        Args:
            trainable_only: Count only parameters requiring gradients.

        Returns:
            Number of parameters.
        """
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )


def build_model(cfg: Config) -> CCASynthesizer:
    """Instantiate the synthesiser from a configuration.

    Args:
        cfg: Root configuration.

    Returns:
        An initialised :class:`CCASynthesizer`.
    """
    return CCASynthesizer(cfg)


# ---------------------------------------------------------------------------
# Masking helper and non-learned comparators
# ---------------------------------------------------------------------------


def effective_agent_mask(present: torch.Tensor, coalition: torch.Tensor) -> torch.Tensor:
    """Combine record availability with coalition membership.

    An agent contributes only when it is both in the coalition *and* actually
    produced a usable record; failed and refused calls are excluded everywhere
    rather than being silently treated as abstentions.

    Args:
        present: ``(B, R)`` bool.
        coalition: ``(B, R)`` float in ``{0, 1}``.

    Returns:
        ``(B, R)`` float mask.
    """
    return present.to(coalition.dtype) * coalition


def _scores_from_weights(weights: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Turn non-negative per-option weights into maskable scores.

    A small deterministic tilt keeps ties resolved by option index rather than
    by an implementation detail of ``argmax``.

    Args:
        weights: ``(B, K)`` non-negative weights.
        option_mask: ``(B, K)`` bool.

    Returns:
        ``(B, K)`` scores with invalid options at the dtype minimum.
    """
    tilt = torch.linspace(0, 1, weights.shape[1], device=weights.device, dtype=weights.dtype)
    scores = weights - 1e-6 * tilt
    return scores.masked_fill(~option_mask, _mask_fill_value(scores.dtype))


def majority_vote_scores(
    batch: Mapping[str, torch.Tensor], coalition: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Simple majority vote over active agents, backing off to the baseline.

    Args:
        batch: Batch dict.
        coalition: ``(B, R)`` coalition mask; ``None`` means full council.

    Returns:
        ``(B, K)`` scores.
    """
    present = batch["present"]
    if coalition is None:
        coalition = torch.ones_like(present, dtype=torch.float32)
    active = effective_agent_mask(present, coalition)
    counts = (batch["vote"] * active.unsqueeze(-1)).sum(dim=1)
    # A council with no votes falls back to the direct answer, which is exactly
    # the empty-coalition null policy.
    fallback = counts.sum(dim=-1, keepdim=True) == 0
    counts = torch.where(fallback, batch["base_vote"], counts)
    return _scores_from_weights(counts, batch["option_mask"])


def confidence_weighted_vote_scores(
    batch: Mapping[str, torch.Tensor], coalition: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Vote weighted by *declared* agent confidence.

    This is the comparator that section 7.3 warns about: it takes self-reported
    confidence at face value.  Reporting it alongside the learned synthesiser
    shows what recalibration is worth.

    Args:
        batch: Batch dict.
        coalition: ``(B, R)`` coalition mask.

    Returns:
        ``(B, K)`` scores.
    """
    present = batch["present"]
    if coalition is None:
        coalition = torch.ones_like(present, dtype=torch.float32)
    active = effective_agent_mask(present, coalition) * (1.0 - batch["abstain"])
    weights = (batch["vote"] * (active * batch["role_conf"]).unsqueeze(-1)).sum(dim=1)
    fallback = weights.sum(dim=-1, keepdim=True) == 0
    weights = torch.where(fallback, batch["base_vote"], weights)
    return _scores_from_weights(weights, batch["option_mask"])


def base_answer_scores(
    batch: Mapping[str, torch.Tensor], coalition: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Direct-answer baseline; the predeclared value of the empty coalition.

    Args:
        batch: Batch dict.
        coalition: Ignored; present for a uniform comparator signature.

    Returns:
        ``(B, K)`` scores.
    """
    del coalition
    return _scores_from_weights(batch["base_vote"], batch["option_mask"])


def single_agent_scores(
    batch: Mapping[str, torch.Tensor], role_index: int
) -> torch.Tensor:
    """Use one agent's answer alone, backing off to the baseline.

    Args:
        batch: Batch dict.
        role_index: Slot index of the agent.

    Returns:
        ``(B, K)`` scores.
    """
    vote = batch["vote"][:, role_index]
    usable = (
        batch["present"][:, role_index].to(vote.dtype)
        * (1.0 - batch["abstain"][:, role_index])
    ).unsqueeze(-1)
    weights = vote * usable
    fallback = weights.sum(dim=-1, keepdim=True) == 0
    weights = torch.where(fallback, batch["base_vote"], weights)
    return _scores_from_weights(weights, batch["option_mask"])


def oracle_upper_bound_scores(
    batch: Mapping[str, torch.Tensor], coalition: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Pick the gold answer whenever any active agent proposed it.

    This is an *upper bound on routing*, not a deployable system: it consumes
    the label.  Specification section 7 requires it to be reported and requires
    it never to be presented as achievable performance; the name and this
    docstring are the enforcement mechanism.

    Args:
        batch: Batch dict.
        coalition: ``(B, R)`` coalition mask.

    Returns:
        ``(B, K)`` scores.
    """
    present = batch["present"]
    if coalition is None:
        coalition = torch.ones_like(present, dtype=torch.float32)
    active = effective_agent_mask(present, coalition)
    votes = batch["vote"] * active.unsqueeze(-1)
    gold = batch["gold"]
    valid = gold >= 0
    safe_gold = gold.clamp(min=0)
    proposed = votes.sum(dim=1).gather(1, safe_gold.unsqueeze(-1)).squeeze(-1) > 0

    weights = torch.where(
        (valid & proposed).unsqueeze(-1),
        F.one_hot(safe_gold, num_classes=votes.shape[-1]).to(votes.dtype),
        majority_vote_scores(batch, coalition).clamp(min=0.0),
    )
    return _scores_from_weights(weights, batch["option_mask"])


#: Registry of non-learned comparators, used by the evaluation report.
BASELINES: Dict[str, object] = {
    "direct_answer_base": base_answer_scores,
    "majority_vote": majority_vote_scores,
    "confidence_weighted_vote": confidence_weighted_vote_scores,
    "oracle_upper_bound_not_deployable": oracle_upper_bound_scores,
}
