"""Structured candidate records: schema, splits, tensorisation, and simulation.

The synthesiser never sees free text.  Its entire input is the structured
candidate record defined in ``RESEARCH_SPECIFICATION.md`` section 7 -- an
answer, a list of atomic claims with evidence status, counter-evidence,
failure flags, a declared confidence, and an abstention bit -- which is exactly
what makes the audit trail of ``ETHICS_AND_DATA_GOVERNANCE.md`` possible without
storing chain-of-thought.

Two data paths are supported and are never confused with one another:

``records_path`` set
    Real logged records are read from JSONL and stamped ``provenance="real"``.

``records_path`` unset
    :func:`simulate_records` produces a labelled synthetic corpus stamped
    ``provenance="simulated"``.  It exists so that the pipeline, the Shapley
    machinery, and the statistics are executable and testable *today*; the PDF
    reports no experiments and this module fabricates none.  Every artifact
    derived from it carries the provenance stamp, and ``--strict-real`` refuses
    to run at all.

Tensor layout produced by :func:`collate_records`, for a batch of ``B`` items
with ``R = 6`` role slots, ``K`` padded answer options, and ``C`` padded claims:

===========================  ===================  ==========================
key                          shape                meaning
===========================  ===================  ==========================
``present``                  ``(B, R)`` bool      agent produced a usable record
``vote``                     ``(B, R, K)`` float  one-hot answer selection
``base_vote``                ``(B, K)`` float     direct-answer baseline vote
``option_mask``              ``(B, K)`` bool      valid answer options
``role_conf``                ``(B, R)`` float     declared confidence
``role_scalars``             ``(B, R, S)`` float  bounded per-agent features
``role_status``              ``(B, R, 4)`` float  evidence-status simplex
``flags``                    ``(B, R, F)`` float  multi-hot failure flags
``claim_status``             ``(B, R, C)`` long   evidence-status ids (+pad)
``claim_conf``               ``(B, R, C)`` float  per-claim confidence
``claim_has_source``         ``(B, R, C)`` float  citation/named-check present
``claim_mask``               ``(B, R, C)`` bool   valid claim slots
``claim_label``              ``(B, R, C)`` float  veridicality supervision
``claim_label_mask``         ``(B, R, C)`` bool   where supervision exists
``cost``                     ``(B, R, 4)`` float  tokens, latency, dollars, retries
``gold``                     ``(B,)`` long        gold option, ``-1`` unknown
===========================  ===================  ==========================
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from config import (
    BENCHMARK_TO_INDEX,
    BENCHMARKS,
    Config,
    EVIDENCE_PAD_INDEX,
    EVIDENCE_STATUS_TO_INDEX,
    EVIDENCE_STATUSES,
    FAILURE_FLAGS,
    FLAG_TO_INDEX,
    N_ROLES,
    ROLE_TO_INDEX,
    ROLES,
)
from utils import DistState, seed_worker, torch_generator

__all__ = [
    "N_ROLE_SCALARS",
    "N_COST_FEATURES",
    "SPLIT_NAMES",
    "ClaimRecord",
    "CandidateRecord",
    "ItemRecord",
    "parse_item",
    "hash_split",
    "simulate_records",
    "CCARecordDataset",
    "collate_records",
    "build_dataloaders",
    "sample_coalitions",
    "full_coalition",
    "empty_coalition",
    "move_batch_to_device",
    "dataset_summary",
    "exclusion_ledger",
]


#: Per-agent bounded scalar features, in fixed order:
#: declared confidence, abstain bit, log-scaled counter-evidence count,
#: claim-count ratio, flag-count ratio, has-answer bit, mean claim confidence,
#: fraction of claims carrying a source or named check.
N_ROLE_SCALARS: int = 8

#: Cost-side features: output tokens, latency, dollars, retries (log-compressed).
#: Retries are carried here because specification section 8 requires publishing
#: retries alongside exclusions and failed calls; an earlier version parsed the
#: field into CandidateRecord and then dropped it before it reached a tensor.
N_COST_FEATURES: int = 4

SPLIT_NAMES: Tuple[str, ...] = ("train", "calibration", "validation", "test")

#: Role-to-primary-benchmark predictions from specification section 5.  Used
#: only by the simulator (to give the synthetic corpus realistic structure) and
#: by reporting (to label the pre-registered agent-by-benchmark family).  It is
#: never fed to the model.
ROLE_PRIMARY_BENCHMARK: Dict[str, str] = {
    "counter_evidence_checker": "truthfulqa",
    "consistency_checker": "math_aime",
    "local_claim_checker": "advconsistency",
    "explore_then_prune": "creative_divergence",
    "rare_hypothesis_checker": "temporal_qa",
    "confidence_calibrator": "sycophancy",
}


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------


@dataclass
class ClaimRecord:
    """One atomic factual claim inside a candidate record.

    Attributes:
        evidence_status: One of :data:`config.EVIDENCE_STATUSES`.
        confidence: Agent-declared confidence in the claim, ``[0, 1]``.
        has_source: Whether a citation or named verification check accompanies
            the claim.  The *content* of the source is deliberately not modelled:
            the synthesiser must not become a text encoder.
        veridical: Optional supervision for the claim-adjudication head;
            ``None`` when unlabelled.
    """

    evidence_status: str
    confidence: float
    has_source: bool
    veridical: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.evidence_status not in EVIDENCE_STATUS_TO_INDEX:
            raise ValueError(
                f"unknown evidence_status '{self.evidence_status}'; expected one "
                f"of {list(EVIDENCE_STATUSES)}"
            )
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


@dataclass
class CandidateRecord:
    """One agent's structured output for one item.

    Attributes:
        role: Functional role label; must be in :data:`config.ROLES`.
        present: ``False`` when the call failed, refused, or was truncated.
            Absent agents are masked out of every computation and counted in the
            exclusions report required by specification section 8.
        answer_option: Index into the item's option list, or ``-1``.
        abstain: The agent declined to answer.
        confidence: Declared confidence, treated as an input feature and never
            as ground truth (section 7.3).
        counterevidence_count: Number of counter-evidence items supplied.
        failure_flags: Subset of :data:`config.FAILURE_FLAGS`.
        claims: Atomic claims.
        output_tokens: Generated tokens, for the cost report.
        latency_ms: Wall-clock latency.
        cost_usd: Provider cost.
        retries: Retry count.
    """

    role: str
    present: bool = True
    answer_option: int = -1
    abstain: bool = False
    confidence: float = 0.5
    counterevidence_count: int = 0
    failure_flags: Tuple[str, ...] = ()
    claims: Tuple[ClaimRecord, ...] = ()
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    retries: int = 0

    def __post_init__(self) -> None:
        if self.role not in ROLE_TO_INDEX:
            raise ValueError(
                f"unknown role '{self.role}'; expected one of {list(ROLES)}. "
                "Clinical identifiers are not accepted as role labels."
            )
        unknown = set(self.failure_flags) - set(FAILURE_FLAGS)
        if unknown:
            raise ValueError(
                f"unknown failure flags {sorted(unknown)}; the flag vocabulary is "
                "closed by design (see config.FAILURE_FLAGS)"
            )
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))


@dataclass
class ItemRecord:
    """A complete council transcript for one benchmark item.

    Attributes:
        item_id: Stable identifier; also drives the deterministic split.
        benchmark: One of :data:`config.BENCHMARKS`.
        n_options: Number of normalised candidate answers.
        gold_option: Index of the correct option, or ``-1`` when unknown.
        base_answer_option: Direct-answer baseline choice defining ``v(empty)``.
        candidates: One record per role slot, ordered by :data:`config.ROLES`.
        option_features: Optional precomputed answer embeddings, ``(K, d)``.
        input_tokens: Shared prompt cost for the item.
    """

    item_id: str
    benchmark: str
    n_options: int
    gold_option: int
    base_answer_option: int
    candidates: Tuple[CandidateRecord, ...]
    option_features: Optional[np.ndarray] = None
    input_tokens: int = 0

    def __post_init__(self) -> None:
        if self.benchmark not in BENCHMARK_TO_INDEX:
            raise ValueError(
                f"unknown benchmark '{self.benchmark}'; expected one of "
                f"{list(BENCHMARKS)}"
            )
        if self.n_options < 1:
            raise ValueError("n_options must be >= 1")
        if self.gold_option >= self.n_options:
            raise ValueError(
                f"gold_option {self.gold_option} outside option range "
                f"[0, {self.n_options})"
            )
        if len(self.candidates) != N_ROLES:
            raise ValueError(
                f"expected exactly {N_ROLES} candidate slots, got "
                f"{len(self.candidates)}"
            )
        seen = [c.role for c in self.candidates]
        if sorted(seen) != sorted(ROLES):
            raise ValueError(f"candidate roles must be a permutation of {list(ROLES)}")


def parse_item(payload: Mapping[str, Any], max_options: int, max_claims: int) -> ItemRecord:
    """Parse one JSON object into a validated :class:`ItemRecord`.

    Missing role slots are materialised as ``present=False`` candidates rather
    than causing a failure, because a partial council is a normal operating
    condition (provider refusal, truncation, timeout) that the specification
    requires to be reported rather than silently dropped.

    Args:
        payload: One decoded JSONL line.
        max_options: Cap on answer options; extra options are truncated and the
            gold index is invalidated if it falls outside the cap.
        max_claims: Cap on claims retained per agent.

    Returns:
        The parsed record.

    Raises:
        KeyError: If a required top-level field is absent.
        ValueError: If a field violates the schema.
    """
    item_id = str(payload["item_id"])
    benchmark = str(payload["benchmark"])
    n_options = int(payload["n_options"])
    if n_options > max_options:
        n_options = max_options
    gold = int(payload.get("gold_option", -1))
    if gold >= n_options:
        gold = -1
    base = int(payload.get("base_answer_option", -1))
    if base >= n_options:
        base = -1

    by_role: Dict[str, CandidateRecord] = {}
    for raw in payload.get("candidates", []):
        role = str(raw["role"])
        claims = tuple(
            ClaimRecord(
                evidence_status=str(c["evidence_status"]),
                confidence=float(c.get("confidence", 0.5)),
                has_source=bool(c.get("source_or_check")) or bool(c.get("has_source", False)),
                veridical=(None if c.get("veridical") is None else bool(c["veridical"])),
            )
            for c in list(raw.get("atomic_claims", []))[:max_claims]
        )
        answer = int(raw.get("answer_option", -1))
        if answer >= n_options:
            answer = -1
        by_role[role] = CandidateRecord(
            role=role,
            present=bool(raw.get("present", True)),
            answer_option=answer,
            abstain=bool(raw.get("abstain", False)),
            confidence=float(raw.get("confidence", 0.5)),
            counterevidence_count=len(raw.get("counterevidence", []))
            if "counterevidence" in raw
            else int(raw.get("counterevidence_count", 0)),
            failure_flags=tuple(raw.get("failure_flags", ())),
            claims=claims,
            output_tokens=int(raw.get("output_tokens", 0)),
            latency_ms=float(raw.get("latency_ms", 0.0)),
            cost_usd=float(raw.get("cost_usd", 0.0)),
            retries=int(raw.get("retries", 0)),
        )

    candidates = tuple(
        by_role.get(role, CandidateRecord(role=role, present=False, answer_option=-1))
        for role in ROLES
    )

    features = payload.get("option_features")
    option_features = (
        np.asarray(features, dtype=np.float32)[:n_options] if features is not None else None
    )

    return ItemRecord(
        item_id=item_id,
        benchmark=benchmark,
        n_options=n_options,
        gold_option=gold,
        base_answer_option=base,
        candidates=candidates,
        option_features=option_features,
        input_tokens=int(payload.get("input_tokens", 0)),
    )


# ---------------------------------------------------------------------------
# Deterministic splitting
# ---------------------------------------------------------------------------


def hash_split(item_id: str, salt: str, fractions: Sequence[float]) -> str:
    """Assign an item to a split by hashing its identifier.

    Hash-based assignment is stable under corpus growth: adding items never
    moves an existing item between splits, which a shuffled index-based split
    would do.  Section 4 requires four disjoint sets and section 15 of the
    limitations register requires hashed release splits.

    Args:
        item_id: Stable item identifier.
        salt: Study-level salt; keep frozen for the duration of a study.
        fractions: ``(train, calibration, validation, test)`` summing to 1.

    Returns:
        One of :data:`SPLIT_NAMES`.
    """
    digest = hashlib.sha256(f"{salt}::{item_id}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)
    cumulative = 0.0
    for name, fraction in zip(SPLIT_NAMES, fractions):
        cumulative += fraction
        if position < cumulative:
            return name
    return SPLIT_NAMES[-1]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

# Per-role simulator parameters.  These are *modelling assumptions of the
# simulator*, not measured quantities, and are stated explicitly so that nobody
# mistakes the resulting numbers for evidence.
#
# Calibrated so the corpus sits in a defensible regime rather than a convenient
# or an adversarial one.  At these values a solo agent scores 0.49-0.61 against
# a chance level near 0.25, mean pairwise error correlation is about 0.10,
# majority vote only barely exceeds the best single agent, and a near-unanimous
# council is right about 72% of the time.  Earlier, stronger common-mode
# settings made unanimity 96% *wrong*, which inverts the normal relationship
# between consensus and correctness and would have trained the synthesizer on an
# artifact.
_SIM_ROLE_SKILL: Dict[str, float] = {
    "counter_evidence_checker": 1.90,
    "consistency_checker": 1.95,
    "local_claim_checker": 1.80,
    "explore_then_prune": 1.70,
    "rare_hypothesis_checker": 1.60,
    "confidence_calibrator": 1.85,
}

# Susceptibility to the canonical-but-wrong answer.  The rare-hypothesis checker
# is deliberately least susceptible; that is the operation it implements.
_SIM_ROLE_LURE_SUSCEPTIBILITY: Dict[str, float] = {
    "counter_evidence_checker": 0.7,
    "consistency_checker": 0.9,
    "local_claim_checker": 0.8,
    "explore_then_prune": 0.6,
    "rare_hypothesis_checker": 0.2,
    "confidence_calibrator": 0.8,
}

# Confidence distortion: (multiplicative logit gain, additive logit shift).
# Gains above 1 with a positive shift produce the overconfidence that the
# recalibration head exists to correct.
_SIM_ROLE_CONF_DISTORTION: Dict[str, Tuple[float, float]] = {
    "counter_evidence_checker": (1.4, 0.7),
    "consistency_checker": (1.6, 0.9),
    "local_claim_checker": (1.3, 0.6),
    "explore_then_prune": (1.5, 1.1),
    "rare_hypothesis_checker": (1.2, 0.4),
    "confidence_calibrator": (1.1, -0.6),
}

# Mean generated tokens.  The explore-then-prune role runs two passes and is
# therefore the natural stress case for the "synthesizer capture by one verbose
# agent" test in specification section 9.
_SIM_ROLE_TOKENS: Dict[str, float] = {
    "counter_evidence_checker": 420.0,
    "consistency_checker": 380.0,
    "local_claim_checker": 300.0,
    "explore_then_prune": 900.0,
    "rare_hypothesis_checker": 260.0,
    "confidence_calibrator": 240.0,
}

_SIM_ROLE_ABSTAIN: Dict[str, float] = {
    "counter_evidence_checker": 0.04,
    "consistency_checker": 0.03,
    "local_claim_checker": 0.05,
    "explore_then_prune": 0.02,
    "rare_hypothesis_checker": 0.02,
    "confidence_calibrator": 0.11,
}


def _sigmoid(x: float) -> float:
    """Numerically stable scalar logistic function.

    Args:
        x: Real input.

    Returns:
        ``1 / (1 + exp(-x))``.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float, eps: float = 1e-4) -> float:
    """Inverse of :func:`_sigmoid` with clamping.

    Args:
        p: Probability.
        eps: Clamp margin keeping the result finite.

    Returns:
        ``log(p / (1 - p))``.
    """
    p = float(np.clip(p, eps, 1.0 - eps))
    return math.log(p / (1.0 - p))


def simulate_records(
    n_items: int,
    seed: int,
    max_options: int = 6,
    max_claims: int = 12,
) -> List[Dict[str, Any]]:
    """Generate a labelled synthetic corpus of candidate records.

    The generative process is stated in full so that it can be criticised:

    1. Each item draws a benchmark, an option count, a gold option, and a
       distinct *lure* option standing for the canonical-but-wrong answer.
    2. A latent difficulty ``d ~ Beta(2, 3)`` and a latent shared-prior strength
       ``u ~ Beta(2, 2)`` are drawn.  ``u`` is what induces **correlated errors
       across agents** -- the concern raised as limitation L7 -- because every
       agent's error probability is pushed by the same ``u`` toward the same
       lure, scaled by its own susceptibility.
    3. With probability ``0.45 u`` the item enters *common mode*: the shared
       prior is wrong and every agent inherits it, its correctness probability
       falling by ``0.65 s_r``.  Latent difficulty alone produces only weak error
       correlation (about 0.03), which would understate L7 badly; common-mode
       failure is what makes the simulated council fail the way a real one built
       on a single base model does.
    4. Agent ``r`` answers correctly with probability
       ``sigmoid(skill_r + bonus_r(benchmark) - 2.2 d - 1.8 u s_r)``; otherwise it
       takes the lure with probability 0.65 (0.90 in common mode) and a uniform
       other option otherwise.
    5. Declared confidence is a *distorted* transform of the correctness
       probability **before** the common-mode adjustment: an agent cannot see
       that the shared prior is wrong, so common-mode items are exactly where it
       is confidently wrong.  This gives the recalibration head a real signal and
       makes pre-recalibration ECE genuinely non-zero.
    6. Claims, counter-evidence, flags, tokens, latency, and cost are drawn from
       role-conditioned distributions.

    This produces data with the *structure* the specification anticipates. It
    produces no evidence about CCA, and every downstream artifact is stamped
    ``provenance="simulated"``.

    Args:
        n_items: Number of items to generate.
        seed: RNG seed; the corpus is a deterministic function of it.
        max_options: Maximum answer options per item.
        max_claims: Maximum claims per agent record.

    Returns:
        A list of JSON-serialisable item payloads accepted by :func:`parse_item`.
    """
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    for index in range(n_items):
        benchmark = str(rng.choice(BENCHMARKS))
        n_options = int(rng.integers(3, max_options + 1))
        gold = int(rng.integers(0, n_options))
        lure_choices = [k for k in range(n_options) if k != gold]
        lure = int(rng.choice(lure_choices))

        difficulty = float(rng.beta(2.0, 3.0))
        shared_prior = float(rng.beta(2.0, 2.0))
        # Common-mode failure.  On some items the base model's prior is simply
        # wrong, and six instances of that base model all inherit it.  This is
        # the concrete form of limitation L7: without it the simulated agents
        # fail almost independently, which would make any council look better
        # than a real one.  Susceptibility is role-dependent, so the
        # rare-hypothesis checker retains a genuine differential advantage here.
        common_mode = bool(rng.random() < 0.45 * shared_prior)

        candidates: List[Dict[str, Any]] = []
        for role in ROLES:
            skill = _SIM_ROLE_SKILL[role]
            bonus = 0.85 if ROLE_PRIMARY_BENCHMARK[role] == benchmark else 0.0
            if role == "counter_evidence_checker" and benchmark == "halueval":
                bonus = 0.55  # secondary target: hallucination detection
            susceptibility = _SIM_ROLE_LURE_SUSCEPTIBILITY[role]

            p_correct = _sigmoid(
                skill + bonus - 2.2 * difficulty - 1.8 * shared_prior * susceptibility
            )
            # The agent cannot see that the shared prior is wrong, so its
            # declared confidence is anchored to the pre-common-mode estimate.
            # That is what produces confidently-wrong agreement -- the failure
            # the section 9 battery probes -- rather than agents that
            # conveniently announce their own unreliability.
            p_declared_basis = p_correct
            lure_probability = 0.65
            if common_mode:
                p_correct *= 1.0 - 0.65 * susceptibility
                lure_probability = 0.90

            correct = bool(rng.random() < p_correct)
            if correct:
                answer = gold
            elif rng.random() < lure_probability:
                answer = lure
            else:
                answer = int(rng.choice([k for k in range(n_options) if k != gold]))

            present = bool(rng.random() > 0.03)
            abstain = bool(rng.random() < _SIM_ROLE_ABSTAIN[role])

            gain, shift = _SIM_ROLE_CONF_DISTORTION[role]
            declared = _sigmoid(
                gain * _logit(p_declared_basis) + shift + float(rng.normal(0, 0.35))
            )
            declared = float(np.clip(declared, 0.02, 0.99))

            n_claims = int(min(max_claims, 1 + rng.poisson(3.0)))
            claims: List[Dict[str, Any]] = []
            for _ in range(n_claims):
                veridical = bool(rng.random() < (0.82 if correct else 0.45))
                if veridical:
                    status_probs = [0.62, 0.13, 0.05, 0.20]
                else:
                    status_probs = [0.30, 0.28, 0.24, 0.18]
                if role == "local_claim_checker":
                    # Local-claim checking surfaces more explicit contradictions.
                    status_probs = list(np.array(status_probs) * np.array([0.9, 1.1, 1.4, 0.9]))
                total = float(sum(status_probs))
                status = str(
                    rng.choice(EVIDENCE_STATUSES, p=[p / total for p in status_probs])
                )
                claim_conf = float(np.clip(rng.beta(5, 2) if veridical else rng.beta(3, 3), 0, 1))
                has_source = bool(rng.random() < (0.7 if status == "supported" else 0.35))
                claims.append(
                    {
                        "claim": f"claim-{len(claims)}",
                        "evidence_status": status,
                        "source_or_check": "named_check" if has_source else "",
                        "confidence": claim_conf,
                        "veridical": veridical,
                    }
                )

            counter_lambda = 3.2 if role == "counter_evidence_checker" else 1.0
            counterevidence = int(rng.poisson(counter_lambda))

            flags: List[str] = []
            if role == "consistency_checker" and not correct and rng.random() < 0.45:
                flags.append("contradiction")
            if benchmark == "math_aime" and not correct and rng.random() < 0.30:
                flags.append("calculation_error")
            if any(c["evidence_status"] == "unsupported" for c in claims) and rng.random() < 0.5:
                flags.append("unsupported_claim")
            if not present:
                flags.append("refusal" if rng.random() < 0.5 else "truncated_output")

            tokens = int(max(16, rng.normal(_SIM_ROLE_TOKENS[role], 90)))
            candidates.append(
                {
                    "role": role,
                    "present": present,
                    "answer_option": -1 if (abstain or not present) else answer,
                    "abstain": abstain,
                    "confidence": declared,
                    "counterevidence_count": counterevidence,
                    "failure_flags": flags,
                    "atomic_claims": claims if present else [],
                    "output_tokens": tokens if present else 0,
                    "latency_ms": float(max(50.0, rng.normal(tokens * 3.1, 220.0))),
                    "cost_usd": tokens * 1.2e-5,
                    "retries": int(rng.random() < 0.05),
                }
            )

        # Direct-answer baseline: a weaker, maximally lure-susceptible agent.
        p_base = _sigmoid(0.70 - 2.2 * difficulty - 2.1 * shared_prior)
        if rng.random() < p_base:
            base_answer = gold
        elif rng.random() < 0.75:
            base_answer = lure
        else:
            base_answer = int(rng.choice([k for k in range(n_options) if k != gold]))

        records.append(
            {
                "item_id": f"sim-{seed}-{index:07d}",
                "benchmark": benchmark,
                "n_options": n_options,
                "gold_option": gold,
                "base_answer_option": base_answer,
                "candidates": candidates,
                "input_tokens": int(max(64, rng.normal(700, 180))),
                "provenance": "simulated",
            }
        )
    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CCARecordDataset(Dataset):
    """Torch dataset over parsed :class:`ItemRecord` objects.

    Each ``__getitem__`` returns per-item tensors with the item's *own* option
    and claim counts; :func:`collate_records` pads them to the batch maximum.
    Padding at collate time rather than at construction keeps memory
    proportional to actual content, which matters because claim counts are
    long-tailed.
    """

    def __init__(
        self,
        items: Sequence[ItemRecord],
        max_claims: int,
        option_feature_dim: int = 0,
        provenance: str = "simulated",
    ):
        """Build the dataset.

        Args:
            items: Parsed item records.
            max_claims: Cap on claims retained per agent.
            option_feature_dim: Width of optional option embeddings; ``0``
                disables them.
            provenance: ``"real"`` or ``"simulated"``, propagated to artifacts.
        """
        self.items: List[ItemRecord] = list(items)
        self.max_claims = int(max_claims)
        self.option_feature_dim = int(option_feature_dim)
        self.provenance = provenance

    def __len__(self) -> int:
        """Number of items."""
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Tensorise one item.

        Args:
            index: Item index.

        Returns:
            A dict of unpadded tensors plus the string ``item_id``.
        """
        item = self.items[index]
        n_options = item.n_options
        n_claims = max(
            1, min(self.max_claims, max((len(c.claims) for c in item.candidates), default=1))
        )

        present = torch.zeros(N_ROLES, dtype=torch.bool)
        vote = torch.zeros(N_ROLES, n_options, dtype=torch.float32)
        role_conf = torch.zeros(N_ROLES, dtype=torch.float32)
        role_scalars = torch.zeros(N_ROLES, N_ROLE_SCALARS, dtype=torch.float32)
        role_status = torch.zeros(N_ROLES, len(EVIDENCE_STATUSES), dtype=torch.float32)
        flags = torch.zeros(N_ROLES, len(FAILURE_FLAGS), dtype=torch.float32)
        claim_status = torch.full(
            (N_ROLES, n_claims), EVIDENCE_PAD_INDEX, dtype=torch.long
        )
        claim_conf = torch.zeros(N_ROLES, n_claims, dtype=torch.float32)
        claim_has_source = torch.zeros(N_ROLES, n_claims, dtype=torch.float32)
        claim_mask = torch.zeros(N_ROLES, n_claims, dtype=torch.bool)
        claim_label = torch.zeros(N_ROLES, n_claims, dtype=torch.float32)
        claim_label_mask = torch.zeros(N_ROLES, n_claims, dtype=torch.bool)
        cost = torch.zeros(N_ROLES, N_COST_FEATURES, dtype=torch.float32)
        abstain = torch.zeros(N_ROLES, dtype=torch.float32)

        for slot, candidate in enumerate(item.candidates):
            if not candidate.present:
                continue
            present[slot] = True
            abstain[slot] = float(candidate.abstain)
            role_conf[slot] = candidate.confidence

            has_answer = 0.0
            if not candidate.abstain and 0 <= candidate.answer_option < n_options:
                vote[slot, candidate.answer_option] = 1.0
                has_answer = 1.0

            kept = candidate.claims[: self.max_claims][:n_claims]
            status_counts = np.zeros(len(EVIDENCE_STATUSES), dtype=np.float32)
            conf_sum = 0.0
            source_sum = 0.0
            for j, claim in enumerate(kept):
                status_index = EVIDENCE_STATUS_TO_INDEX[claim.evidence_status]
                claim_status[slot, j] = status_index
                claim_conf[slot, j] = claim.confidence
                claim_has_source[slot, j] = float(claim.has_source)
                claim_mask[slot, j] = True
                status_counts[status_index] += 1.0
                conf_sum += claim.confidence
                source_sum += float(claim.has_source)
                if claim.veridical is not None:
                    claim_label[slot, j] = float(claim.veridical)
                    claim_label_mask[slot, j] = True

            n_kept = max(1, len(kept))
            role_status[slot] = torch.from_numpy(status_counts / n_kept)

            for flag in candidate.failure_flags:
                flags[slot, FLAG_TO_INDEX[flag]] = 1.0

            role_scalars[slot] = torch.tensor(
                [
                    candidate.confidence,
                    float(candidate.abstain),
                    math.log1p(candidate.counterevidence_count) / 3.0,
                    len(kept) / float(self.max_claims),
                    len(candidate.failure_flags) / float(len(FAILURE_FLAGS)),
                    has_answer,
                    conf_sum / n_kept,
                    source_sum / n_kept,
                ],
                dtype=torch.float32,
            )
            cost[slot] = torch.tensor(
                [
                    math.log1p(candidate.output_tokens) / 10.0,
                    math.log1p(candidate.latency_ms) / 10.0,
                    math.log1p(candidate.cost_usd * 1e4) / 10.0,
                    math.log1p(candidate.retries) / 10.0,
                ],
                dtype=torch.float32,
            )

        base_vote = torch.zeros(n_options, dtype=torch.float32)
        if 0 <= item.base_answer_option < n_options:
            base_vote[item.base_answer_option] = 1.0

        sample: Dict[str, Any] = {
            "item_id": item.item_id,
            "benchmark_id": torch.tensor(BENCHMARK_TO_INDEX[item.benchmark], dtype=torch.long),
            "n_options": torch.tensor(n_options, dtype=torch.long),
            "gold": torch.tensor(item.gold_option, dtype=torch.long),
            "present": present,
            "vote": vote,
            "base_vote": base_vote,
            "role_conf": role_conf,
            "abstain": abstain,
            "role_scalars": role_scalars,
            "role_status": role_status,
            "flags": flags,
            "claim_status": claim_status,
            "claim_conf": claim_conf,
            "claim_has_source": claim_has_source,
            "claim_mask": claim_mask,
            "claim_label": claim_label,
            "claim_label_mask": claim_label_mask,
            "cost": cost,
            "input_tokens": torch.tensor(float(item.input_tokens), dtype=torch.float32),
        }
        if self.option_feature_dim > 0:
            if item.option_features is not None:
                features = torch.from_numpy(item.option_features.astype(np.float32))
                if features.shape[1] != self.option_feature_dim:
                    raise ValueError(
                        f"option_features width {features.shape[1]} does not match "
                        f"configured option_feature_dim {self.option_feature_dim}"
                    )
            else:
                features = torch.zeros(n_options, self.option_feature_dim)
            sample["option_features"] = features
        return sample


def collate_records(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pad a list of per-item samples into a batch.

    Answer options and claims are padded to the batch maximum and accompanied by
    boolean masks; every consumer downstream is mask-aware, and no padded slot
    ever contributes to a loss, a metric, or an attention distribution.

    Args:
        batch: Samples from :meth:`CCARecordDataset.__getitem__`.

    Returns:
        A batch dict; see the module docstring for shapes.
    """
    batch_size = len(batch)
    max_options = int(max(int(s["n_options"]) for s in batch))
    max_claims = int(max(s["claim_status"].shape[1] for s in batch))

    option_mask = torch.zeros(batch_size, max_options, dtype=torch.bool)
    vote = torch.zeros(batch_size, N_ROLES, max_options, dtype=torch.float32)
    base_vote = torch.zeros(batch_size, max_options, dtype=torch.float32)
    claim_status = torch.full(
        (batch_size, N_ROLES, max_claims), EVIDENCE_PAD_INDEX, dtype=torch.long
    )
    claim_conf = torch.zeros(batch_size, N_ROLES, max_claims, dtype=torch.float32)
    claim_has_source = torch.zeros(batch_size, N_ROLES, max_claims, dtype=torch.float32)
    claim_mask = torch.zeros(batch_size, N_ROLES, max_claims, dtype=torch.bool)
    claim_label = torch.zeros(batch_size, N_ROLES, max_claims, dtype=torch.float32)
    claim_label_mask = torch.zeros(batch_size, N_ROLES, max_claims, dtype=torch.bool)

    has_option_features = "option_features" in batch[0]
    option_features = None
    if has_option_features:
        feature_dim = batch[0]["option_features"].shape[1]
        option_features = torch.zeros(batch_size, max_options, feature_dim)

    for i, sample in enumerate(batch):
        k = int(sample["n_options"])
        c = int(sample["claim_status"].shape[1])
        option_mask[i, :k] = True
        vote[i, :, :k] = sample["vote"]
        base_vote[i, :k] = sample["base_vote"]
        claim_status[i, :, :c] = sample["claim_status"]
        claim_conf[i, :, :c] = sample["claim_conf"]
        claim_has_source[i, :, :c] = sample["claim_has_source"]
        claim_mask[i, :, :c] = sample["claim_mask"]
        claim_label[i, :, :c] = sample["claim_label"]
        claim_label_mask[i, :, :c] = sample["claim_label_mask"]
        if option_features is not None:
            option_features[i, :k] = sample["option_features"]

    out: Dict[str, Any] = {
        "item_id": [s["item_id"] for s in batch],
        "benchmark_id": torch.stack([s["benchmark_id"] for s in batch]),
        "n_options": torch.stack([s["n_options"] for s in batch]),
        "gold": torch.stack([s["gold"] for s in batch]),
        "present": torch.stack([s["present"] for s in batch]),
        # Role identity travels as *content*, never as an axis position.  This is
        # what makes the permutation-invariance guarantee in model.py hold: to
        # permute agents you permute this tensor alongside every other role-axis
        # tensor, and the model output must not move.
        "role_ids": torch.arange(N_ROLES, dtype=torch.long).unsqueeze(0).expand(
            batch_size, N_ROLES
        ).contiguous(),
        "vote": vote,
        "base_vote": base_vote,
        "option_mask": option_mask,
        "role_conf": torch.stack([s["role_conf"] for s in batch]),
        "abstain": torch.stack([s["abstain"] for s in batch]),
        "role_scalars": torch.stack([s["role_scalars"] for s in batch]),
        "role_status": torch.stack([s["role_status"] for s in batch]),
        "flags": torch.stack([s["flags"] for s in batch]),
        "claim_status": claim_status,
        "claim_conf": claim_conf,
        "claim_has_source": claim_has_source,
        "claim_mask": claim_mask,
        "claim_label": claim_label,
        "claim_label_mask": claim_label_mask,
        "cost": torch.stack([s["cost"] for s in batch]),
        "input_tokens": torch.stack([s["input_tokens"] for s in batch]),
    }
    if option_features is not None:
        out["option_features"] = option_features
    return out


def move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device, non_blocking: bool = False
) -> Dict[str, Any]:
    """Move every tensor in a batch to ``device``, leaving metadata untouched.

    Args:
        batch: Batch dict from :func:`collate_records`.
        device: Destination device.
        non_blocking: Use asynchronous copies (CUDA pinned memory only).

    Returns:
        A new dict with tensors relocated.
    """
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


# ---------------------------------------------------------------------------
# Coalition sampling
# ---------------------------------------------------------------------------


def sample_coalitions(
    batch_size: int,
    dropout: float,
    p_full: float,
    p_empty: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Draw per-item coalition masks for training.

    Training under random coalitions is what makes a *single* synthesiser valid
    for all ``2**6`` coalitions at evaluation time, turning specification
    section 8's 64-coalition sweep into 64 masked forward passes instead of 64
    training runs.  The full and empty coalitions are over-sampled because they
    anchor the two ends of the Shapley computation: the empty coalition defines
    ``v(empty)``, which section 8 requires to be predeclared as the direct-answer
    baseline.

    Args:
        batch_size: Number of items in the batch.
        dropout: Per-role masking probability in the mixed regime.
        p_full: Probability of forcing the complete council.
        p_empty: Probability of forcing the empty coalition.
        generator: Seeded generator, kept on CPU for cross-device determinism.
        device: Device the returned mask should live on.

    Returns:
        Float mask of shape ``(batch_size, N_ROLES)`` with entries in ``{0, 1}``.
    """
    regime = torch.rand(batch_size, generator=generator)
    bernoulli = (
        torch.rand(batch_size, N_ROLES, generator=generator) >= dropout
    ).float()
    full = torch.ones(batch_size, N_ROLES)
    empty = torch.zeros(batch_size, N_ROLES)

    is_full = (regime < p_full).unsqueeze(1)
    is_empty = ((regime >= p_full) & (regime < p_full + p_empty)).unsqueeze(1)
    coalition = torch.where(is_full, full, torch.where(is_empty, empty, bernoulli))
    return coalition.to(device)


def full_coalition(batch_size: int, device: torch.device) -> torch.Tensor:
    """Return the all-ones coalition mask.

    Args:
        batch_size: Batch size.
        device: Target device.

    Returns:
        Float tensor ``(batch_size, N_ROLES)`` of ones.
    """
    return torch.ones(batch_size, N_ROLES, device=device)


def empty_coalition(batch_size: int, device: torch.device) -> torch.Tensor:
    """Return the all-zeros coalition mask defining ``v(empty)``.

    Args:
        batch_size: Batch size.
        device: Target device.

    Returns:
        Float tensor ``(batch_size, N_ROLES)`` of zeros.
    """
    return torch.zeros(batch_size, N_ROLES, device=device)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: Source path.

    Returns:
        Decoded records.

    Raises:
        FileNotFoundError: If the path does not exist.
        json.JSONDecodeError: On a malformed line, with the line number
            attached to the message.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"records file not found: {path}")
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
                    f"{error.msg} (in {path} at line {line_number})", error.doc, error.pos
                ) from error
    return out


def build_datasets(cfg: Config) -> Tuple[Dict[str, CCARecordDataset], Dict[str, Any]]:
    """Load or simulate records and split them four ways.

    Args:
        cfg: Root configuration.

    Returns:
        Tuple ``(datasets, summary)`` where ``datasets`` is keyed by
        :data:`SPLIT_NAMES` and ``summary`` describes the corpus, including the
        exclusions that specification section 8 requires to be published.
    """
    if cfg.data.records_path:
        raw = _load_jsonl(cfg.data.records_path)
        # Provenance comes from the records, not from the fact that a file
        # exists. Inferring "real" from "loaded from disk" would let stub-provider
        # output past --strict-real, which is precisely what that flag exists to
        # stop. A corpus is real only if every record says so; anything else --
        # a "simulated" stamp, a missing stamp, or a mixture -- is not real.
        stamps = {str(record.get("provenance", "unstamped")) for record in raw}
        provenance = "real" if stamps == {"real"} else "simulated"
        if provenance != "real":
            _mixed = ", ".join(sorted(stamps))
            warnings.warn(
                f"{cfg.data.records_path} carries provenance stamps {{{_mixed}}}; "
                "treating the corpus as simulated. Only a corpus in which every "
                "record is stamped 'real' may be reported as an empirical result.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        raw = simulate_records(
            n_items=cfg.data.simulate_n_items,
            seed=cfg.data.simulate_seed,
            max_options=cfg.data.max_options,
            max_claims=cfg.data.max_claims,
        )
        provenance = "simulated"
    cfg.data.provenance = provenance

    parsed: List[ItemRecord] = []
    parse_failures = 0
    for payload in raw:
        try:
            parsed.append(parse_item(payload, cfg.data.max_options, cfg.data.max_claims))
        except (KeyError, ValueError):
            parse_failures += 1
    if not parsed:
        raise RuntimeError(
            f"no parsable items (attempted {len(raw)}, failed {parse_failures})"
        )

    buckets: Dict[str, List[ItemRecord]] = {name: [] for name in SPLIT_NAMES}
    for item in parsed:
        buckets[hash_split(item.item_id, cfg.data.split_salt, cfg.data.split_fractions)].append(item)

    unlabelled_dropped = 0
    if cfg.data.drop_incomplete_items:
        before = len(buckets["train"])
        buckets["train"] = [i for i in buckets["train"] if i.gold_option >= 0]
        unlabelled_dropped = before - len(buckets["train"])

    datasets = {
        name: CCARecordDataset(
            items=items,
            max_claims=cfg.data.max_claims,
            option_feature_dim=cfg.data.option_feature_dim,
            provenance=provenance,
        )
        for name, items in buckets.items()
    }

    total_absent = sum(
        sum(1 for c in item.candidates if not c.present) for item in parsed
    )
    summary = {
        "provenance": provenance,
        "records_path": cfg.data.records_path,
        "n_raw": len(raw),
        "n_parsed": len(parsed),
        "parse_failures": parse_failures,
        "unlabelled_train_items_dropped": unlabelled_dropped,
        "absent_agent_records": total_absent,
        "absent_agent_rate": total_absent / max(1, len(parsed) * N_ROLES),
        "split_sizes": {name: len(items) for name, items in buckets.items()},
        "split_salt": cfg.data.split_salt,
        "split_fractions": list(cfg.data.split_fractions),
        "exclusion_ledger": exclusion_ledger(parsed, n_raw=len(raw), parse_failures=parse_failures),
    }
    return datasets, summary


def exclusion_ledger(
    items: Sequence[ItemRecord], *, n_raw: int, parse_failures: int
) -> Dict[str, Any]:
    """Account for every call that did not yield a usable candidate record.

    Specification section 8 requires publishing "all exclusions, failed calls,
    retries, and missing outputs".  Counting absent records is not enough on its
    own: a refusal, a truncation, and a schema violation are different failures
    with different implications, and an aggregate hides which one dominates.
    This walks the parsed corpus once and reports the breakdown by role and by
    declared failure flag.

    Retries are counted from the parsed records rather than recovered from the
    log-compressed cost tensor, so the number is exact.

    Args:
        items: Parsed item records.
        n_raw: Lines read from the source corpus.
        parse_failures: Lines that could not be parsed at all.

    Returns:
        A JSON-serialisable ledger.
    """
    absent_by_role: Dict[str, int] = {role: 0 for role in ROLES}
    absent_by_flag: Dict[str, int] = {flag: 0 for flag in FAILURE_FLAGS}
    absent_unexplained = 0
    retries_by_role: Dict[str, int] = {role: 0 for role in ROLES}
    abstain_by_role: Dict[str, int] = {role: 0 for role in ROLES}
    flags_present_by_flag: Dict[str, int] = {flag: 0 for flag in FAILURE_FLAGS}
    retried_calls = 0

    for item in items:
        for candidate in item.candidates:
            retries_by_role[candidate.role] += int(candidate.retries)
            retried_calls += int(candidate.retries > 0)
            if candidate.present:
                abstain_by_role[candidate.role] += int(candidate.abstain)
                for flag in candidate.failure_flags:
                    flags_present_by_flag[flag] += 1
                continue
            absent_by_role[candidate.role] += 1
            if candidate.failure_flags:
                for flag in candidate.failure_flags:
                    absent_by_flag[flag] += 1
            else:
                # A missing record with no declared reason is itself a finding:
                # the orchestrator is expected to attach a failure_flag.
                absent_unexplained += 1

    n_items = max(1, len(items))
    total_calls = n_items * N_ROLES
    total_absent = sum(absent_by_role.values())
    total_retries = sum(retries_by_role.values())
    return {
        "n_items": len(items),
        "n_expected_calls": total_calls,
        "source_lines": n_raw,
        "unparsable_lines": parse_failures,
        "missing_outputs": {
            "total": total_absent,
            "rate": total_absent / total_calls,
            "by_role": absent_by_role,
            "by_failure_flag": absent_by_flag,
            "unexplained": absent_unexplained,
        },
        "retries": {
            "total": total_retries,
            "per_item": total_retries / n_items,
            "calls_with_at_least_one_retry": retried_calls,
            "by_role": retries_by_role,
        },
        "abstentions_by_role": abstain_by_role,
        "flags_on_present_records": flags_present_by_flag,
        "note": (
            "Published per specification section 8. 'unexplained' counts absent "
            "records carrying no failure_flag; a non-zero value means the "
            "orchestrator dropped a call without recording why."
        ),
    }


def build_dataloaders(
    cfg: Config, dist_state: DistState
) -> Tuple[Dict[str, DataLoader], Dict[str, Any]]:
    """Construct dataloaders for all four splits.

    Training uses a :class:`~torch.utils.data.DistributedSampler` when a process
    group is active.  Evaluation splits are *not* distributed-sampled: each rank
    would otherwise see a padded, partly duplicated shard, and the paired
    item-level statistics of section 8 require an exact, unduplicated item set.
    Evaluation is therefore run on rank 0 over the full split, which is cheap
    because the model is small and inputs are structured records rather than
    text.

    Args:
        cfg: Root configuration.
        dist_state: Resolved distributed state.

    Returns:
        Tuple ``(dataloaders, summary)``.
    """
    datasets, summary = build_datasets(cfg)
    generator = torch_generator(cfg.train.seed)
    pin = cfg.data.pin_memory and dist_state.device.type == "cuda"

    loaders: Dict[str, DataLoader] = {}
    for name, dataset in datasets.items():
        is_train = name == "train"
        sampler = None
        if is_train and dist_state.enabled:
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist_state.world_size,
                rank=dist_state.rank,
                shuffle=True,
                seed=cfg.train.seed,
                drop_last=False,
            )
        shuffle = is_train and sampler is None
        if not is_train and cfg.data.randomize_eval_order:
            # Section 4: run test items in randomised order.  The order is a
            # deterministic function of the seed and is recorded in the manifest.
            shuffle = True

        loaders[name] = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=pin,
            drop_last=False,
            collate_fn=collate_records,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=cfg.data.num_workers > 0,
        )
    return loaders, summary


def dataset_summary(dataset: CCARecordDataset) -> Dict[str, Any]:
    """Compute descriptive statistics for one split.

    Args:
        dataset: The split to describe.

    Returns:
        Counts by benchmark, label coverage, and mean option/claim counts.
    """
    by_benchmark: Dict[str, int] = {b: 0 for b in BENCHMARKS}
    labelled = 0
    option_total = 0
    claim_total = 0
    for item in dataset.items:
        by_benchmark[item.benchmark] += 1
        labelled += int(item.gold_option >= 0)
        option_total += item.n_options
        claim_total += sum(len(c.claims) for c in item.candidates)
    n = max(1, len(dataset))
    return {
        "n_items": len(dataset),
        "labelled_fraction": labelled / n,
        "mean_options": option_total / n,
        "mean_claims_per_item": claim_total / n,
        "by_benchmark": by_benchmark,
        "provenance": dataset.provenance,
    }
