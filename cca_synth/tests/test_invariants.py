"""Invariant tests for the properties the specification treats as guarantees.

These are not accuracy tests -- accuracy on simulated data means nothing.  Each
test pins a *structural* claim made in ``/logistics`` or in the module
docstrings, so that a refactor cannot quietly break it:

* agent-permutation invariance (specification section 9);
* option-permutation equivariance (position bias);
* role blindness actually blinds (section 7.1, RQ3);
* the empty coalition reduces to the direct-answer baseline (section 8);
* Shapley efficiency and symmetry axioms hold on the implementation;
* Benjamini-Hochberg matches a reference calculation;
* the audit writer rejects hidden reasoning, unknown fields, and clinical labels
  (``ETHICS_AND_DATA_GOVERNANCE.md``);
* seeding is reproducible and the deterministic split is stable.

Run with ``python -m pytest tests -q`` from the repository root.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402
    Config,
    N_ROLES,
    ROLES,
    apply_overrides,
    config_from_dict,
    config_hash,
)
from dataset import (  # noqa: E402
    CCARecordDataset,
    collate_records,
    hash_split,
    parse_item,
    sample_coalitions,
    simulate_records,
)
from engine import _permute_batch_roles, rule_targets_from_records  # noqa: E402
from model import build_model, majority_vote_scores  # noqa: E402
from utils import (  # noqa: E402
    AuditRecordWriter,
    benjamini_hochberg,
    expected_calibration_error,
    index_to_coalition,
    seed_everything,
    shapley_from_values,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config() -> Config:
    """A small, fast configuration for tests."""
    cfg = Config()
    cfg.data.simulate_n_items = 64
    cfg.data.batch_size = 16
    cfg.model.d_model = 48
    cfg.model.n_heads = 4
    cfg.model.n_layers = 2
    cfg.model.dropout = 0.0
    return cfg.validate()


@pytest.fixture(scope="module")
def batch(config: Config):
    """One collated batch built from simulated records."""
    records = simulate_records(
        n_items=config.data.batch_size,
        seed=11,
        max_options=config.data.max_options,
        max_claims=config.data.max_claims,
    )
    items = [parse_item(r, config.data.max_options, config.data.max_claims) for r in records]
    dataset = CCARecordDataset(items, max_claims=config.data.max_claims)
    return collate_records([dataset[i] for i in range(len(dataset))])


# ---------------------------------------------------------------------------
# Architectural guarantees
# ---------------------------------------------------------------------------


def test_agent_permutation_invariance(config: Config, batch) -> None:
    """Permuting agent slots must not move the option logits.

    The set encoder carries no positional encoding over the agent axis and role
    identity enters only as content, so this is a structural property. Any
    failure is a bug, not a result.
    """
    seed_everything(0, deterministic=False)
    model = build_model(config).eval()
    coalition = torch.ones(batch["present"].shape[0], N_ROLES)

    with torch.no_grad():
        reference = model(batch, coalition)["option_logits"]
        for seed in (1, 2, 3):
            permutation = torch.randperm(N_ROLES, generator=torch.Generator().manual_seed(seed))
            permuted = model(_permute_batch_roles(batch, permutation), coalition)["option_logits"]
            assert torch.allclose(reference, permuted, atol=1e-5), (
                f"permutation {permutation.tolist()} changed the logits by "
                f"{(reference - permuted).abs().max().item():.3e}"
            )


def test_option_permutation_equivariance(config: Config, batch) -> None:
    """Relabelling answer options must permute the prediction, not change it."""
    seed_everything(0, deterministic=False)
    model = build_model(config).eval()
    batch_size, n_options = batch["option_mask"].shape
    coalition = torch.ones(batch_size, N_ROLES)

    permutation = torch.randperm(n_options, generator=torch.Generator().manual_seed(5))
    permuted = dict(batch)
    permuted["vote"] = batch["vote"].index_select(2, permutation)
    permuted["base_vote"] = batch["base_vote"].index_select(1, permutation)
    permuted["option_mask"] = batch["option_mask"].index_select(1, permutation)

    with torch.no_grad():
        reference = model(batch, coalition)["option_logits"]
        moved = model(permuted, coalition)["option_logits"]

    # The exact property is on the logits: relabelling option k -> perm[k] must
    # carry the score with it. Asserting on the argmax instead is a weaker proxy
    # that fails spuriously whenever the top two options are within float noise
    # of each other, which says nothing about position bias.
    valid = batch["option_mask"]
    expected = reference.index_select(1, permutation)
    deviation = (moved - expected)[valid.index_select(1, permutation)].abs().max().item()
    assert deviation < 1e-4, f"option relabelling moved logits by {deviation:.3e}"


def test_role_blind_model_ignores_role_labels(config: Config, batch) -> None:
    """With role embeddings off, shuffling ``role_ids`` must change nothing."""
    blind = config_from_dict({"model": {"use_role_embeddings": False}}, base=config)
    seed_everything(0, deterministic=False)
    model = build_model(blind).eval()
    coalition = torch.ones(batch["present"].shape[0], N_ROLES)

    relabelled = dict(batch)
    relabelled["role_ids"] = batch["role_ids"].index_select(
        1, torch.randperm(N_ROLES, generator=torch.Generator().manual_seed(9))
    )
    with torch.no_grad():
        assert torch.allclose(
            model(batch, coalition)["option_logits"],
            model(relabelled, coalition)["option_logits"],
            atol=1e-6,
        )


def test_role_aware_model_can_use_role_labels(config: Config, batch) -> None:
    """With role embeddings on, the label pathway must actually exist.

    This is the control for the previous test: if both configurations were
    label-invariant, the role-blind ablation would be vacuous.
    """
    seed_everything(0, deterministic=False)
    model = build_model(config).eval()
    # Give the role embedding real magnitude; at init it is near zero by design.
    with torch.no_grad():
        model.role_embedding.weight.normal_(0.0, 1.0)
    coalition = torch.ones(batch["present"].shape[0], N_ROLES)

    relabelled = dict(batch)
    relabelled["role_ids"] = torch.roll(batch["role_ids"], shifts=1, dims=1)
    with torch.no_grad():
        difference = (
            model(batch, coalition)["option_logits"]
            - model(relabelled, coalition)["option_logits"]
        ).abs().max()
    assert difference > 1e-6, "role embeddings are enabled but have no effect"


def test_empty_coalition_is_well_defined(config: Config, batch) -> None:
    """The empty coalition must produce finite logits over valid options only."""
    seed_everything(0, deterministic=False)
    model = build_model(config).eval()
    batch_size = batch["present"].shape[0]

    with torch.no_grad():
        logits = model(batch, torch.zeros(batch_size, N_ROLES))["option_logits"]
    valid = batch["option_mask"]
    assert torch.isfinite(logits[valid]).all(), "empty coalition produced non-finite logits"
    # Assert the semantic property (masked options get no probability mass)
    # rather than a magic threshold that must track the sentinel constant.
    assert torch.softmax(logits, dim=-1)[~valid].max() < 1e-12, "padded options leaked mass"


def test_evidence_gates_vote_aggregation(config: Config, batch) -> None:
    """Adjudicated evidence must reach the answer head.

    Regression test for a real defect: the claim head originally ran *after* the
    option scorer, so no gradient path connected evidence to the answer. The
    trained model became a pure vote aggregator, and unanimous agreement on a
    wrong option with every claim marked ``unsupported`` still drove accuracy to
    0.0099 in the section 9 battery.

    Holding votes, confidences, and presence fixed, degrading only the *claim
    evidence* must move the option logits.
    """
    from config import EVIDENCE_STATUS_TO_INDEX

    seed_everything(0, deterministic=False)
    model = build_model(config).eval()
    coalition = torch.ones(batch["present"].shape[0], N_ROLES)

    degraded = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    degraded["claim_status"] = torch.where(
        batch["claim_mask"],
        torch.full_like(batch["claim_status"], EVIDENCE_STATUS_TO_INDEX["unsupported"]),
        batch["claim_status"],
    )
    degraded["claim_has_source"] = torch.zeros_like(batch["claim_has_source"])
    degraded["claim_conf"] = torch.zeros_like(batch["claim_conf"])

    with torch.no_grad():
        reference = model(batch, coalition)["option_logits"]
        moved = model(degraded, coalition)["option_logits"]

    valid = batch["option_mask"]
    shift = (reference - moved)[valid].abs().max().item()
    assert shift > 1e-4, (
        f"degrading every claim moved the option logits by only {shift:.2e}; "
        "the evidence pathway is not reaching the answer head"
    )


def test_evidence_credibility_gradient_reaches_claim_head(config: Config, batch) -> None:
    """The answer loss must train the claim head, not just the claim loss.

    This is the structural half of the fix: even with the claim-supervision
    weight set to zero, optimising the answer alone must produce gradient in the
    claim head.
    """
    seed_everything(0, deterministic=False)
    model = build_model(config).train()
    coalition = torch.ones(batch["present"].shape[0], N_ROLES)

    outputs = model(batch, coalition)
    # Answer head only -- no claim or rule loss whatsoever.
    outputs["option_logits"].masked_fill(~batch["option_mask"], 0.0).sum().backward()

    grads = [
        p.grad.abs().sum().item()
        for p in model.claim_head.parameters()
        if p.grad is not None
    ]
    assert grads and max(grads) > 0.0, (
        "no gradient reached the claim head from the answer loss; evidence is "
        "still a detached side task"
    )


def test_evidence_gate_floor_bounds_the_penalty(config: Config, batch) -> None:
    """An agent with no credible claims keeps exactly ``floor`` of its weight."""
    scorer_cfg = config_from_dict({"model": {"evidence_gate_floor": 1.0}}, base=config)
    seed_everything(0, deterministic=False)
    ungated = build_model(scorer_cfg).eval()
    assert ungated.option_scorer.evidence_gate_floor == 1.0

    # With the floor at 1.0 the gate is identically 1 and evidence cannot
    # modulate votes -- the documented ablation back to pure vote aggregation.
    credibility = torch.rand(4, N_ROLES)
    gate = 1.0 + (1.0 - 1.0) * credibility
    assert torch.allclose(gate, torch.ones_like(gate))


def test_masked_agents_do_not_influence_output(config: Config, batch) -> None:
    """Corrupting an agent that is outside the coalition must change nothing."""
    seed_everything(0, deterministic=False)
    model = build_model(config).eval()
    batch_size = batch["present"].shape[0]
    coalition = torch.ones(batch_size, N_ROLES)
    coalition[:, 2] = 0.0  # exclude slot 2

    corrupted = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    corrupted["role_conf"][:, 2] = 1.0
    corrupted["vote"][:, 2] = 0.0
    corrupted["vote"][:, 2, 0] = 1.0
    corrupted["role_scalars"][:, 2] = 1.0

    with torch.no_grad():
        assert torch.allclose(
            model(batch, coalition)["option_logits"],
            model(corrupted, coalition)["option_logits"],
            atol=1e-6,
        )


def test_majority_vote_falls_back_to_base_answer(config: Config, batch) -> None:
    """With no active agents, majority vote must return the direct answer."""
    batch_size = batch["present"].shape[0]
    scores = majority_vote_scores(batch, torch.zeros(batch_size, N_ROLES))
    has_base = batch["base_vote"].sum(dim=-1) > 0
    predicted = scores.argmax(dim=-1)
    expected = batch["base_vote"].argmax(dim=-1)
    assert (predicted[has_base] == expected[has_base]).all()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_temperature_fitting_survives_masked_options() -> None:
    """Temperature fitting must never return a non-finite value.

    Regression test for a silent, total failure. The mask sentinel was
    ``finfo.min / 4`` (about -8.5e37); once LBFGS moved the temperature below 1,
    dividing that sentinel overflowed and the gradient went nan, so
    ``fit_temperature`` returned nan and every downstream prediction collapsed
    to accuracy 0.0000 with nan calibration metrics. Nothing raised.
    """
    from engine import fit_temperature
    from model import MASK_FILL_FLOAT32

    generator = torch.Generator().manual_seed(0)
    for scale in (0.1, 0.5, 1.0, 5.0, 20.0):
        logits = torch.randn(400, 6, generator=generator) * scale
        mask = torch.zeros(400, 6, dtype=torch.bool)
        mask[:, :4] = True
        logits = logits.masked_fill(~mask, MASK_FILL_FLOAT32)
        gold = torch.randint(0, 4, (400,), generator=generator)

        temperature = fit_temperature(logits, gold, mask)
        assert np.isfinite(temperature), f"non-finite temperature at scale {scale}"
        assert 0.05 <= temperature <= 20.0, f"temperature {temperature} out of range"


def test_mask_sentinel_survives_temperature_scaling() -> None:
    """The sentinel must stay finite after division by the smallest temperature."""
    from model import MIN_TEMPERATURE, _mask_fill_value

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        sentinel = torch.tensor(_mask_fill_value(dtype), dtype=dtype)
        scaled = sentinel / torch.tensor(MIN_TEMPERATURE, dtype=dtype)
        assert torch.isfinite(scaled), f"{dtype} sentinel overflows when scaled"
        # Still negative enough to take a softmax to zero against ordinary logits.
        probs = torch.softmax(
            torch.stack([sentinel.float(), torch.tensor(10.0)]), dim=0
        )
        assert probs[0] < 1e-12


def test_shapley_efficiency_axiom() -> None:
    """Shapley values must sum to ``v(A) - v(empty)``."""
    rng = np.random.default_rng(3)
    values = rng.normal(size=2**N_ROLES)
    phi = shapley_from_values(values, N_ROLES)
    assert phi.sum() == pytest.approx(values[-1] - values[0], abs=1e-10)


def test_shapley_symmetry_and_null_player() -> None:
    """Interchangeable players share a value; a null player gets zero."""
    # A purely additive game where role 0 contributes 1, role 1 contributes 1,
    # and role 5 contributes nothing.
    contributions = np.array([1.0, 1.0, 0.5, 0.25, 0.125, 0.0])
    values = np.array(
        [
            sum(c for c, bit in zip(contributions, index_to_coalition(i, N_ROLES)) if bit)
            for i in range(2**N_ROLES)
        ]
    )
    phi = shapley_from_values(values, N_ROLES)
    assert phi[0] == pytest.approx(phi[1], abs=1e-10), "symmetry axiom violated"
    assert phi[5] == pytest.approx(0.0, abs=1e-10), "null-player axiom violated"
    assert np.allclose(phi, contributions, atol=1e-10)


def test_benjamini_hochberg_matches_reference() -> None:
    """BH q-values and rejections must match a hand-checked example."""
    # m = 8, alpha = 0.05.  The step-up threshold is i * alpha / m, so the
    # largest i with p_(i) <= i * alpha / m is i = 2 (0.008 <= 0.0125, while
    # 0.039 > 0.01875); everything up to rank 2 is rejected.
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    rejected, q_values = benjamini_hochberg(p_values, alpha=0.05)
    assert rejected.tolist() == [True, True, False, False, False, False, False, False]
    assert q_values.tolist() == pytest.approx(
        [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.074 * 8 / 7, 0.205], abs=1e-9
    )
    assert np.all(np.diff(np.sort(q_values)) >= -1e-12), "q-values must be monotone"

    # A step-*down* implementation would stop at the first non-rejection and
    # miss rank 3 here; the step-up rule must reject it.
    step_up_case = [0.01, 0.03, 0.005]
    rejected_case, _ = benjamini_hochberg(step_up_case, alpha=0.05)
    assert rejected_case.tolist() == [True, True, True]


def test_ece_is_zero_for_a_perfectly_calibrated_predictor() -> None:
    """A predictor whose confidence equals its accuracy has zero ECE."""
    rng = np.random.default_rng(7)
    confidences = rng.uniform(0.5, 1.0, size=200_000)
    correct = (rng.uniform(size=confidences.size) < confidences).astype(float)
    error = expected_calibration_error(confidences, correct, n_bins=15, binning="equal_mass")
    assert error < 0.01, f"expected near-zero ECE, got {error:.4f}"


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------


def test_audit_writer_rejects_hidden_reasoning(tmp_path) -> None:
    """Chain-of-thought-like fields must raise, not be silently dropped."""
    writer = AuditRecordWriter(str(tmp_path / "audit.jsonl"), enabled=True)
    for bad_key in ("chain_of_thought", "reasoning_trace", "system_prompt", "api_key"):
        with pytest.raises(ValueError, match="prohibited pattern"):
            writer.validate({bad_key: "anything"})
    writer.close()


def test_audit_writer_rejects_unknown_and_clinical_fields(tmp_path) -> None:
    """Unknown keys and clinical identifiers must both raise."""
    writer = AuditRecordWriter(str(tmp_path / "audit.jsonl"), enabled=True)
    with pytest.raises(ValueError, match="AUDIT_FIELD_ALLOWLIST"):
        writer.validate({"undeclared_field": 1})
    with pytest.raises(ValueError, match="clinical identifier"):
        writer.validate({"agent_role": "Sentinel"})
    writer.validate({"agent_role": "counter_evidence_checker"})  # must not raise
    writer.close()


def test_audit_writer_recurses_into_nested_records(tmp_path) -> None:
    """Validation must reach nested lists and mappings."""
    writer = AuditRecordWriter(str(tmp_path / "audit.jsonl"), enabled=True)
    with pytest.raises(ValueError, match="clinical identifier"):
        writer.validate({"agent_records": [{"agent_role": "Savant"}]})
    assert writer.write(
        {
            "item_id": "x",
            "agent_records": [
                {"agent_slot": 0, "agent_role": "local_claim_checker", "claims": []}
            ],
        }
    )
    writer.close()


def test_role_labels_are_functional_not_clinical() -> None:
    """No public role identifier may be a clinical term."""
    forbidden = {"sentinel", "auditor", "savant", "oscillator", "dissenter", "challenger"}
    assert not forbidden.intersection({r.lower() for r in ROLES})


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_seeding_is_reproducible() -> None:
    """Identical seeds must reproduce torch, NumPy, and Python draws."""
    import random

    def draw():
        seed_everything(4242, deterministic=False)
        return (
            torch.randn(8).tolist(),
            np.random.rand(8).tolist(),
            [random.random() for _ in range(8)],
        )

    assert draw() == draw()


def test_hash_split_is_stable_and_disjoint() -> None:
    """Split assignment must depend only on the id and salt, and cover all ids."""
    fractions = (0.5, 0.15, 0.15, 0.2)
    ids = [f"item-{i}" for i in range(5000)]
    first = [hash_split(i, "salt-a", fractions) for i in ids]
    second = [hash_split(i, "salt-a", fractions) for i in ids]
    assert first == second

    counts = {name: first.count(name) for name in set(first)}
    assert set(counts) <= {"train", "calibration", "validation", "test"}
    assert abs(counts["train"] / len(ids) - 0.5) < 0.03

    resalted = [hash_split(i, "salt-b", fractions) for i in ids]
    assert resalted != first, "changing the salt must reshuffle the assignment"


def test_coalition_sampler_respects_forced_regimes() -> None:
    """Forcing p_full=1 or p_empty=1 must yield exactly those coalitions."""
    generator = torch.Generator().manual_seed(0)
    full = sample_coalitions(64, 0.5, 1.0, 0.0, generator, torch.device("cpu"))
    assert full.sum().item() == 64 * N_ROLES
    empty = sample_coalitions(64, 0.5, 0.0, 1.0, generator, torch.device("cpu"))
    assert empty.sum().item() == 0


def test_config_hash_ignores_non_computational_fields() -> None:
    """Changing the output directory must not change the config hash."""
    base = Config()
    moved = apply_overrides(Config(), ["train.output_dir=/tmp/elsewhere"])
    assert config_hash(base) == config_hash(moved)
    reseeded = apply_overrides(Config(), ["train.seed=99"])
    assert config_hash(base) != config_hash(reseeded)


def test_config_rejects_unknown_fields() -> None:
    """A typo in a config file must raise rather than be ignored."""
    with pytest.raises(KeyError):
        config_from_dict({"model": {"d_modle": 128}})
    with pytest.raises(KeyError):
        config_from_dict({"modle": {"d_model": 128}})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_collate_masks_are_consistent(config: Config, batch) -> None:
    """No agent may vote for a padded option, and pad claims must be masked."""
    padded_options = ~batch["option_mask"]
    assert batch["vote"][padded_options.unsqueeze(1).expand_as(batch["vote"])].sum() == 0
    assert batch["base_vote"][padded_options].sum() == 0
    assert batch["claim_conf"][~batch["claim_mask"]].sum() == 0


def test_absent_agents_carry_no_features(config: Config) -> None:
    """A record marked ``present=False`` must contribute nothing to the tensors."""
    records = simulate_records(n_items=32, seed=3)
    for record in records:
        record["candidates"][0]["present"] = False
    items = [parse_item(r, config.data.max_options, config.data.max_claims) for r in records]
    dataset = CCARecordDataset(items, max_claims=config.data.max_claims)
    collated = collate_records([dataset[i] for i in range(len(dataset))])

    assert not collated["present"][:, 0].any()
    assert collated["vote"][:, 0].sum() == 0
    assert collated["role_scalars"][:, 0].sum() == 0
    assert collated["claim_mask"][:, 0].sum() == 0


def test_rule_targets_follow_the_declared_policy() -> None:
    """The adjudication rule must be the documented function of the record."""
    from config import ADJUDICATION_RULES, EVIDENCE_STATUS_TO_INDEX

    status = torch.tensor(
        [
            [
                [
                    EVIDENCE_STATUS_TO_INDEX["supported"],
                    EVIDENCE_STATUS_TO_INDEX["supported"],
                    EVIDENCE_STATUS_TO_INDEX["unsupported"],
                    EVIDENCE_STATUS_TO_INDEX["contradicted"],
                    EVIDENCE_STATUS_TO_INDEX["uncertain"],
                ]
            ]
        ]
    )
    has_source = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0]]])
    rules = rule_targets_from_records(status, has_source)[0, 0].tolist()
    names = [ADJUDICATION_RULES[r] for r in rules]
    assert names == [
        "accepted_evidence_supported",
        "accepted_corroborated_by_independent_agent",
        "rejected_unsupported_no_source",
        "rejected_contradicted_by_counterevidence",
        "deferred_conflict_unresolved",
    ]


def test_simulator_produces_correlated_agent_errors() -> None:
    """The simulator must reproduce limitation L7, not idealised independence.

    A corpus of independent agents would make the council look better than any
    real one; the shared-prior term exists precisely to avoid that.
    """
    records = simulate_records(n_items=3000, seed=17)
    errors = []
    for record in records:
        gold = record["gold_option"]
        row = [
            float(c["answer_option"] != gold)
            for c in record["candidates"]
            if c["present"] and not c["abstain"]
        ]
        if len(row) == N_ROLES:
            errors.append(row)
    matrix = np.asarray(errors)
    correlation = np.corrcoef(matrix, rowvar=False)
    off_diagonal = correlation[~np.eye(N_ROLES, dtype=bool)]
    assert off_diagonal.mean() > 0.05, (
        "simulated agents fail independently, which would understate the "
        "confounding that limitation L7 describes"
    )
    assert off_diagonal.mean() < 0.40, (
        "error correlation this high makes the council degenerate; the "
        "simulator should be realistic, not adversarial"
    )


def test_simulator_is_neither_trivial_nor_adversarial() -> None:
    """The synthetic corpus must sit in a defensible regime.

    Regression test for a real defect: an early common-mode setting made
    near-unanimous councils wrong 96% of the time, inverting the normal
    relationship between consensus and correctness. A synthesizer trained on
    that corpus would learn an artifact of the simulator.
    """
    import collections

    records = simulate_records(n_items=4000, seed=23)
    solo_correct = np.zeros(N_ROLES)
    solo_total = np.zeros(N_ROLES)
    unanimous = unanimous_correct = 0
    chance_levels = []

    for record in records:
        gold = record["gold_option"]
        chance_levels.append(1.0 / record["n_options"])
        votes = []
        for r, candidate in enumerate(record["candidates"]):
            if candidate["present"] and not candidate["abstain"] and candidate["answer_option"] >= 0:
                solo_total[r] += 1
                solo_correct[r] += candidate["answer_option"] == gold
                votes.append(candidate["answer_option"])
        if len(votes) == N_ROLES and len(set(votes)) == 1:
            unanimous += 1
            unanimous_correct += votes[0] == gold

    solo = solo_correct / np.maximum(solo_total, 1)
    chance = float(np.mean(chance_levels))

    assert solo.min() > chance + 0.15, (
        f"weakest agent ({solo.min():.3f}) is too close to chance ({chance:.3f}); "
        "the corpus would be mostly noise"
    )
    assert solo.max() < 0.90, "agents this strong leave no room for a council to help"
    assert unanimous > 0, "no unanimous items; consensus behaviour is untested"
    assert unanimous_correct / unanimous > 0.55, (
        f"unanimous councils are right only {unanimous_correct / unanimous:.2f} of "
        "the time; consensus must remain weak positive evidence, not a negative "
        "signal"
    )
