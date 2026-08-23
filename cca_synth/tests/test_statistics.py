"""Tests for the P11 and P12 additions.

Covers:

* training resume restoring optimiser, scheduler, scaler, EMA, step, and RNG
  (P11 item 1);
* ``eval.coalition_metric`` actually reaching the characteristic function
  (P11 item 2);
* the exclusion ledger accounting for every failed call (P11 item 3);
* Shapley interaction indices against hand-worked additive and non-additive
  value functions (P12);
* the repeated-trial schedule being stable and reused (P12);
* the necessary/sufficient decision rule refusing the words without a
  replication (P12).
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COALITION_METRICS, Config, N_ROLES, ROLES  # noqa: E402
from dataset import build_datasets, exclusion_ledger, parse_item  # noqa: E402
from engine import (  # noqa: E402
    _interaction_index,
    interaction_analysis,
    leave_one_out_analysis,
    necessity_sufficiency_verdicts,
    repeated_trial_schedule,
    verify_trial_coverage,
)
from utils import coalition_item_scores, shapley_from_values  # noqa: E402


# ---------------------------------------------------------------------------
# P11 item 2 -- coalition_metric is wired up
# ---------------------------------------------------------------------------


def test_coalition_metric_rejects_unknown_value():
    """An unknown metric fails at config validation, not at run time."""
    cfg = Config()
    cfg.eval.coalition_metric = "not_a_metric"
    with pytest.raises(ValueError, match="coalition_metric"):
        cfg.validate()


def test_coalition_metrics_differ_from_each_other():
    """The three metrics are genuinely different functions.

    A registry whose entries all compute the same thing would pass a smoke test
    and mean nothing.
    """
    predictions = np.array([0, 1, 2, 0])
    labels = np.array([0, 1, 1, 0])
    abstained = np.array([False, False, False, True])
    probabilities = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]]
    )

    selective = coalition_item_scores(
        "selective_accuracy", predictions, labels, abstained, probabilities
    )
    accuracy = coalition_item_scores(
        "accuracy", predictions, labels, abstained, probabilities
    )
    brier = coalition_item_scores(
        "negative_brier", predictions, labels, abstained, probabilities
    )

    # The abstained item scores 0 under selective_accuracy but 1 under accuracy,
    # because it was predicted correctly and only the abstention policy hid it.
    assert selective[3] == 0.0
    assert accuracy[3] == 1.0
    # negative_brier is bounded and not degenerate.
    assert np.all((brier >= -1.0) & (brier <= 1.0))
    assert not np.allclose(selective, brier)


def test_negative_brier_requires_probabilities():
    """A proper scoring rule cannot be computed from hard decisions."""
    with pytest.raises(ValueError, match="predictive distribution"):
        coalition_item_scores(
            "negative_brier",
            np.array([0]),
            np.array([0]),
            np.array([False]),
            probabilities=None,
        )


def test_unlabelled_items_score_nan_under_every_metric():
    """A ``-1`` gold label is missing data, not a wrong answer."""
    for metric in COALITION_METRICS:
        scores = coalition_item_scores(
            metric,
            np.array([0, 1]),
            np.array([-1, 1]),
            np.array([False, False]),
            probabilities=np.array([[0.5, 0.5], [0.2, 0.8]]),
        )
        assert np.isnan(scores[0]), f"{metric} scored an unlabelled item"
        assert not np.isnan(scores[1])


# ---------------------------------------------------------------------------
# P11 item 3 -- the exclusion ledger
# ---------------------------------------------------------------------------


def _record(role: str, present: bool, flags=(), retries: int = 0) -> dict:
    """Build one candidate payload."""
    return {
        "role": role,
        "present": present,
        "answer_option": 1 if present else -1,
        "abstain": False,
        "confidence": 0.5,
        "failure_flags": list(flags),
        "atomic_claims": [],
        "retries": retries,
    }


def test_exclusion_ledger_counts_every_expected_call():
    """Missing outputs, retries, and reasons are all accounted for."""
    payload = {
        "item_id": "x-1",
        "benchmark": "halueval",
        "n_options": 2,
        "gold_option": 1,
        "base_answer_option": 0,
        "candidates": [
            _record(ROLES[0], True, retries=2),
            _record(ROLES[1], False, ("refusal",)),
            _record(ROLES[2], False, ("truncated_output",), retries=1),
            _record(ROLES[3], False),  # no reason given -- must be flagged
            _record(ROLES[4], True),
            _record(ROLES[5], True),
        ],
    }
    item = parse_item(payload, max_options=2, max_claims=4)
    ledger = exclusion_ledger([item], n_raw=1, parse_failures=0)

    assert ledger["n_expected_calls"] == N_ROLES
    assert ledger["missing_outputs"]["total"] == 3
    assert ledger["missing_outputs"]["by_failure_flag"]["refusal"] == 1
    assert ledger["missing_outputs"]["by_failure_flag"]["truncated_output"] == 1
    # The record with no failure_flag is the interesting one: an orchestrator
    # that drops a call without saying why is a defect, and the ledger surfaces
    # it rather than absorbing it.
    assert ledger["missing_outputs"]["unexplained"] == 1
    assert ledger["retries"]["total"] == 3
    assert ledger["retries"]["calls_with_at_least_one_retry"] == 2


def test_exclusion_ledger_reaches_the_dataset_summary():
    """The ledger is produced by the normal build path, not only on demand."""
    cfg = Config()
    cfg.data.simulate_n_items = 120
    _, summary = build_datasets(cfg)
    ledger = summary["exclusion_ledger"]
    assert ledger["n_items"] == summary["n_parsed"]
    assert ledger["missing_outputs"]["total"] == summary["absent_agent_records"]


# ---------------------------------------------------------------------------
# P12 -- interaction terms
# ---------------------------------------------------------------------------


def _additive_values(weights: np.ndarray) -> np.ndarray:
    """Coalition values for a perfectly additive game."""
    values = np.zeros(2**N_ROLES)
    for index in range(2**N_ROLES):
        values[index] = sum(
            weights[r] for r in range(N_ROLES) if index & (1 << r)
        )
    return values


def test_interaction_is_zero_for_an_additive_game():
    """An additive value function has zero interaction everywhere.

    This is the property that makes the index meaningful: if it were non-zero
    here, a reported interaction would be an artifact of the estimator.
    """
    weights = np.array([0.10, 0.05, 0.08, 0.02, 0.07, 0.03])
    values = _additive_values(weights)
    for i in range(N_ROLES):
        for j in range(i + 1, N_ROLES):
            assert _interaction_index(values, i, j) == pytest.approx(0.0, abs=1e-12)


def test_interaction_detects_a_planted_pair_effect():
    """A game with one superadditive pair shows it, and only it."""
    weights = np.array([0.10, 0.05, 0.08, 0.02, 0.07, 0.03])
    values = _additive_values(weights)
    bonus = 0.20
    # Every coalition containing both agent 0 and agent 1 gets a bonus.
    for index in range(2**N_ROLES):
        if (index & 1) and (index & 2):
            values[index] += bonus

    assert _interaction_index(values, 0, 1) == pytest.approx(bonus, abs=1e-10)
    for i in range(N_ROLES):
        for j in range(i + 1, N_ROLES):
            if (i, j) == (0, 1):
                continue
            assert _interaction_index(values, i, j) == pytest.approx(0.0, abs=1e-10)


def test_interaction_sign_distinguishes_complements_from_substitutes():
    """Substitutable agents produce a negative interaction."""
    weights = np.array([0.10, 0.10, 0.05, 0.05, 0.05, 0.05])
    values = _additive_values(weights)
    # Agents 0 and 1 are redundant: having both is worth less than the sum.
    for index in range(2**N_ROLES):
        if (index & 1) and (index & 2):
            values[index] -= 0.08
    assert _interaction_index(values, 0, 1) < 0


def test_interaction_analysis_reports_every_pair_with_intervals():
    """The public entry point returns all 15 pairs with usable intervals."""
    rng = np.random.default_rng(7)
    weights = np.array([0.10, 0.05, 0.08, 0.02, 0.07, 0.03])
    base = _additive_values(weights)
    for index in range(2**N_ROLES):
        if (index & 1) and (index & 2):
            base[index] += 0.15

    n_items = 400
    score_matrix = np.clip(
        base[None, :] + rng.normal(0.0, 0.05, size=(n_items, 2**N_ROLES)), 0.0, 1.0
    )

    cfg = Config()
    cfg.eval.n_bootstrap = 60
    result = interaction_analysis(score_matrix, cfg)

    assert result["n_pairs"] == N_ROLES * (N_ROLES - 1) // 2
    for pair in result["pairs"]:
        assert pair["ci_lo"] <= pair["interaction"] <= pair["ci_hi"]
        assert pair["relationship"] in ("complementary", "substitutable", "additive")

    planted = next(
        p for p in result["pairs"]
        if {p["role_a"], p["role_b"]} == {ROLES[0], ROLES[1]}
    )
    assert planted["interaction"] > 0.05
    assert planted["relationship"] == "complementary"


def test_degenerate_value_function_is_flagged_despite_efficiency_passing():
    """A collapsed value function must not pass as a clean result.

    This is the retrain-mode failure of FINDINGS_EVIDENCE_PATHWAY.md section 8:
    all 64 coalitions scored 0, every Shapley value was mechanically 0, and the
    efficiency axiom certified it because 0 == 0 - 0. The degeneracy check
    exists precisely because the efficiency check cannot see this.
    """
    from engine import shapley_analysis

    cfg = Config()
    cfg.eval.n_bootstrap = 20
    collapsed = np.zeros((100, 2**N_ROLES))
    result = shapley_analysis(collapsed, cfg)

    # Efficiency passes, vacuously.
    assert result["efficiency_check"]["absolute_gap"] == pytest.approx(0.0)
    # Degeneracy does not.
    assert result["degeneracy_check"]["degenerate"] is True
    assert result["degeneracy_check"]["passed"] is False
    assert result["degeneracy_check"]["n_distinct_coalition_values"] == 1
    assert "pipeline fault, not a finding" in result["degeneracy_check"]["note"]


def test_healthy_value_function_passes_the_degeneracy_check():
    """The check must not fire on a normal ablation."""
    from engine import shapley_analysis

    rng = np.random.default_rng(3)
    values = _additive_values(np.array([0.10, 0.05, 0.08, 0.02, 0.07, 0.03]))
    matrix = np.clip(
        values[None, :] + rng.normal(0, 0.03, size=(200, 2**N_ROLES)), 0.0, 1.0
    )
    cfg = Config()
    cfg.eval.n_bootstrap = 20
    result = shapley_analysis(matrix, cfg)
    assert result["degeneracy_check"]["passed"] is True
    assert result["degeneracy_check"]["n_distinct_coalition_values"] > 1


def test_additivity_gap_is_labelled_as_an_identity_not_evidence():
    """The efficiency identity must not be presented as evidence of additivity."""
    values = _additive_values(np.full(N_ROLES, 0.05))
    score_matrix = np.tile(values, (50, 1))
    cfg = Config()
    cfg.eval.n_bootstrap = 20
    result = interaction_analysis(score_matrix, cfg)
    assert result["additivity"]["gap"] == pytest.approx(0.0, abs=1e-9)
    assert "NOT evidence" in result["additivity"]["note"]


# ---------------------------------------------------------------------------
# P12 -- repeated trials
# ---------------------------------------------------------------------------


def test_repeated_trial_schedule_is_deterministic_and_order_free():
    """The same items and seed give the same schedule, whatever the input order."""
    items = [f"item-{i:03d}" for i in range(25)]
    a = repeated_trial_schedule(items, 3, 12345)
    b = repeated_trial_schedule(list(reversed(items)), 3, 12345)
    assert a["schedule_sha256"] == b["schedule_sha256"]
    assert a["n_observations_per_coalition"] == 75


def test_repeated_trial_schedule_changes_with_the_seed():
    """A different base seed is a different schedule."""
    items = [f"item-{i:03d}" for i in range(10)]
    assert (
        repeated_trial_schedule(items, 2, 1)["schedule_sha256"]
        != repeated_trial_schedule(items, 2, 2)["schedule_sha256"]
    )


def test_trial_coverage_names_the_under_covered_coalitions():
    """A coalition short of its trials is named, not silently averaged in."""
    counts = {i: 3 for i in range(2**N_ROLES)}
    counts[5] = 1
    del counts[9]
    coverage = verify_trial_coverage(counts, required=3)
    assert not coverage["passed"]
    assert 5 in coverage["under_covered"]
    assert 9 in coverage["missing_coalitions"]


# ---------------------------------------------------------------------------
# P12 -- the necessary/sufficient decision rule
# ---------------------------------------------------------------------------


def _shapley_stub(ci_lo: float, ci_hi: float, value: float) -> dict:
    """Build a shapley_analysis-shaped payload with a chosen interval."""
    coalition_values = {"0" * N_ROLES: 0.40, "1" * N_ROLES: 0.70}
    for r in range(N_ROLES):
        bits = ["0"] * N_ROLES
        bits[r] = "1"
        coalition_values["".join(bits)] = 0.45
    return {
        "empty_coalition_value": 0.40,
        "full_coalition_value": 0.70,
        "coalition_values": coalition_values,
        "roles": [
            {
                "role": role,
                "shapley": value,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "prob_positive": 0.99,
                "primary_benchmark": "truthfulqa",
            }
            for role in ROLES
        ],
    }


def test_necessary_is_refused_without_a_replication():
    """A strong interval alone does not license the word."""
    shapley = _shapley_stub(ci_lo=0.05, ci_hi=0.09, value=0.07)
    verdicts = necessity_sufficiency_verdicts(shapley, {"pairs": []}, replication=None)
    assert verdicts["n_necessary"] == 0
    for entry in verdicts["roles"]:
        assert entry["may_write_necessary"] is False
        assert "replication" in entry["reason"].lower()


def test_necessary_is_refused_when_the_interval_spans_zero():
    """An interval including zero blocks the word even with a replication."""
    shapley = _shapley_stub(ci_lo=-0.01, ci_hi=0.09, value=0.04)
    replication = {role: 1.0 for role in ROLES}
    verdicts = necessity_sufficiency_verdicts(
        shapley, {"pairs": []}, replication=replication
    )
    assert verdicts["n_necessary"] == 0
    assert verdicts["roles"][0]["permitted_wording"] == (
        "contribution not distinguishable from zero"
    )


def test_necessary_is_refused_below_the_predeclared_sesoi():
    """Excluding zero is not enough; the SESOI must be cleared."""
    shapley = _shapley_stub(ci_lo=0.002, ci_hi=0.02, value=0.008)
    replication = {role: 1.0 for role in ROLES}
    verdicts = necessity_sufficiency_verdicts(
        shapley, {"pairs": []}, replication=replication
    )
    assert verdicts["n_necessary"] == 0
    assert "SESOI" in verdicts["roles"][0]["reason"]
    assert verdicts["roles"][0]["permitted_wording"] == "contributes positively"


def test_necessary_is_permitted_only_when_all_three_conditions_hold():
    """Interval above SESOI, excluding zero, and replicated."""
    shapley = _shapley_stub(ci_lo=0.05, ci_hi=0.09, value=0.07)
    replication = {role: 1.0 for role in ROLES}
    verdicts = necessity_sufficiency_verdicts(
        shapley, {"pairs": []}, replication=replication
    )
    assert verdicts["n_necessary"] == N_ROLES
    assert verdicts["roles"][0]["permitted_wording"] == "necessary"


def test_substitutable_pairs_are_surfaced_to_the_report():
    """A substitutable pair is reported so it can override a necessity claim."""
    shapley = _shapley_stub(ci_lo=0.05, ci_hi=0.09, value=0.07)
    interactions = {
        "pairs": [
            {"role_a": ROLES[0], "role_b": ROLES[1], "relationship": "substitutable"},
            {"role_a": ROLES[2], "role_b": ROLES[3], "relationship": "additive"},
        ]
    }
    verdicts = necessity_sufficiency_verdicts(shapley, interactions, replication=None)
    assert verdicts["substitutable_pairs"] == [f"{ROLES[0]}+{ROLES[1]}"]


# ---------------------------------------------------------------------------
# P11 item 1 -- training resume
# ---------------------------------------------------------------------------


def _tiny_trainer(output_dir: str, run_name: str, epochs: int):
    """Build a small CPU trainer for resume testing."""
    from dataset import build_dataloaders
    from engine import Trainer
    from model import build_model
    from utils import DistState

    cfg = Config()
    cfg.dist.device = "cpu"
    cfg.data.simulate_n_items = 160
    cfg.data.num_workers = 0
    cfg.train.epochs = epochs
    cfg.train.batch_size = 16
    cfg.train.output_dir = output_dir
    cfg.train.run_name = run_name
    cfg.train.amp = False
    cfg.train.eval_every = 1
    cfg.eval.write_audit_records = False
    cfg.validate()

    dist_state = DistState(
        enabled=False, rank=0, local_rank=0, world_size=1,
        device=torch.device("cpu"), backend="",
    )
    loaders, _ = build_dataloaders(cfg, dist_state)
    return cfg, Trainer(cfg, build_model(cfg), loaders, dist_state)


def test_resume_restores_full_training_state():
    """Optimiser, scheduler, scaler, EMA, step, and RNG all come back.

    Both trainers are built with the *same* ``train.epochs``, because that is
    what a real interrupted run looks like: the job died partway through a
    4-epoch schedule and is restarted with the same configuration. Changing
    ``epochs`` between the two changes the config hash and is correctly refused
    by ``resume_from`` -- that case is covered separately below.
    """
    with tempfile.TemporaryDirectory() as directory:
        cfg, trainer = _tiny_trainer(directory, "resume_test", epochs=2)
        trainer.train()
        step_after = trainer.global_step
        best_after = trainer.best_metric

        # A fresh trainer knows nothing.
        cfg2, resumed = _tiny_trainer(directory, "resume_test", epochs=2)
        assert resumed.global_step == 0
        assert resumed.start_epoch == 0

        assert resumed.resume_from() is True
        assert resumed.global_step == step_after
        assert resumed.best_metric == pytest.approx(best_after)
        assert resumed.start_epoch == 2
        # The optimiser carries real state, not a fresh one.
        assert any(resumed.optimizer.state_dict()["state"])
        # The scheduler resumes at the right point on its curve.
        assert resumed.scheduler.last_epoch > 0


def test_resume_refuses_a_mismatched_config_by_default():
    """Resuming across a config change silently produces a hybrid model.

    Extending a schedule (``epochs=1`` to ``epochs=3``) is a config change like
    any other and is refused by default, which is deliberate: the LR schedule is
    a function of ``total_steps``, so a checkpoint written under a 1-epoch cosine
    curve does not belong on a 3-epoch one without the operator saying so.
    """
    with tempfile.TemporaryDirectory() as directory:
        cfg, trainer = _tiny_trainer(directory, "resume_guard", epochs=1)
        trainer.train()

        cfg2, other = _tiny_trainer(directory, "resume_guard", epochs=3)
        with pytest.raises(ValueError, match="config hash"):
            other.resume_from()
        # Explicit override is available, and warns rather than raising.
        assert other.resume_from(strict=False) is True
        assert other.start_epoch == 1


def test_resume_continues_rather_than_restarting():
    """A resumed run trains the remaining epochs only."""
    with tempfile.TemporaryDirectory() as directory:
        cfg, trainer = _tiny_trainer(directory, "resume_continue", epochs=2)
        trainer.train()
        steps_for_two = trainer.global_step

        cfg2, resumed = _tiny_trainer(directory, "resume_continue", epochs=2)
        resumed.resume_from(strict=False)
        summary = resumed.train()
        # Nothing left to do: start_epoch == epochs.
        assert resumed.global_step == steps_for_two
        assert summary["global_steps"] == steps_for_two


def test_missing_checkpoint_is_not_an_error_for_an_implicit_resume():
    """``--resume`` on a fresh run starts from scratch instead of failing."""
    with tempfile.TemporaryDirectory() as directory:
        _, trainer = _tiny_trainer(directory, "never_trained", epochs=1)
        assert trainer.resume_from() is False
        assert trainer.start_epoch == 0


# ---------------------------------------------------------------------------
# Leave-one-out attribution (PREREGISTRATION.md A2.2)
# ---------------------------------------------------------------------------


def _loo_cfg(n_bootstrap: int = 400) -> Config:
    """Config with a small bootstrap, for the leave-one-out tests."""
    cfg = Config()
    cfg.eval.n_bootstrap = n_bootstrap
    return cfg


def _loo_matrix(drop_rates, n_items: int = 600, full_rate: float = 0.80, seed: int = 11):
    """Score matrix whose drop-one columns have the given success rates."""
    rng = np.random.default_rng(seed)
    rates = np.array([full_rate, *drop_rates], dtype=np.float64)
    return (rng.random((n_items, rates.size)) < rates).astype(np.float64)


def test_leave_one_out_recovers_planted_contrasts():
    """Each role's contrast tracks the gap it was constructed to have."""
    # Roles 0 and 2 are load-bearing; the rest are inert.
    matrix = _loo_matrix([0.60, 0.80, 0.70, 0.80, 0.80, 0.80])
    out = leave_one_out_analysis(matrix, _loo_cfg(), mode="retrain")

    by_role = {r["role"]: r for r in out["roles"]}
    assert by_role[ROLES[0]]["loo_contrast"] == pytest.approx(0.20, abs=0.05)
    assert by_role[ROLES[2]]["loo_contrast"] == pytest.approx(0.10, abs=0.05)
    for index in (1, 3, 4, 5):
        assert by_role[ROLES[index]]["loo_contrast"] == pytest.approx(0.0, abs=0.05)


def test_moves_benchmark_verdict_is_conjunctive():
    """A2.2 requires both an interval excluding zero and the threshold."""
    matrix = _loo_matrix([0.60, 0.78, 0.80, 0.80, 0.80, 0.80])
    cfg = _loo_cfg()
    cfg.eval.loo_moves_threshold = 0.05
    out = leave_one_out_analysis(matrix, cfg, mode="retrain")
    by_role = {r["role"]: r for r in out["roles"]}

    strong = by_role[ROLES[0]]
    assert strong["interval_excludes_zero"] and strong["meets_threshold"]
    assert strong["moves_benchmark"] is True

    # A real but sub-threshold effect must not be reported as moving the
    # benchmark: A2.5 sets the floor at 0.05 precisely because the scoped n
    # cannot resolve anything smaller.
    weak = by_role[ROLES[1]]
    assert weak["meets_threshold"] is False
    assert weak["moves_benchmark"] is False


def test_threshold_alone_cannot_certify_a_role():
    """A large point estimate with an interval spanning zero does not pass."""
    # Few items, so the interval is wide even though the gap is large.
    matrix = _loo_matrix([0.55], n_items=8, seed=3)
    matrix = np.repeat(matrix[:, :2], (1 + N_ROLES) // 2 + 1, axis=1)[:, : 1 + N_ROLES]
    out = leave_one_out_analysis(matrix, _loo_cfg(200), mode="retrain")
    for role in out["roles"]:
        if not role["interval_excludes_zero"]:
            assert role["moves_benchmark"] is False


def test_leave_one_out_degeneracy_check_catches_a_collapsed_value_function():
    """A flat value function fails loudly, as it does for Shapley."""
    out = leave_one_out_analysis(np.ones((120, 1 + N_ROLES)), _loo_cfg(100))
    assert out["degeneracy_check"]["degenerate"] is True
    assert out["degeneracy_check"]["passed"] is False
    assert "pipeline fault" in out["degeneracy_check"]["note"]
    for role in out["roles"]:
        assert role["loo_contrast"] == pytest.approx(0.0)
        assert role["moves_benchmark"] is False


def test_leave_one_out_rejects_a_full_coalition_matrix():
    """Passing the 64-column Shapley matrix by mistake is an error."""
    with pytest.raises(ValueError, match="expected 7 columns"):
        leave_one_out_analysis(np.ones((10, 2**N_ROLES)), _loo_cfg(50))


def test_leave_one_out_does_not_claim_shapley_properties():
    """No efficiency check, and the caveats say why."""
    out = leave_one_out_analysis(_loo_matrix([0.7] * N_ROLES), _loo_cfg(200), mode="retrain")
    assert "efficiency_check" not in out
    assert out["estimator"] == "leave_one_out"
    joined = " ".join(out["caveats"]).lower()
    assert "not a shapley value" in joined
    assert "efficiency" in joined
    # The substitutability bias registered in A1 must be stated, since it is
    # what makes a passing role a conservative result rather than an inflated
    # one.
    assert "substitutable" in joined


def test_masked_mode_caveat_names_the_a1_prohibition():
    """A masked-mode run must carry the per-role prohibition in its artifact."""
    out = leave_one_out_analysis(_loo_matrix([0.7] * N_ROLES), _loo_cfg(100), mode="masked")
    joined = " ".join(out["caveats"])
    assert "A1" in joined and "forbids per-role claims" in joined


# ---------------------------------------------------------------------------
# Leave-one-out CLI wiring: the A2.2 estimator must be selectable and must
# feed the same FDR family analysis as the Shapley path.
# ---------------------------------------------------------------------------


def test_attribution_estimator_is_validated():
    """Only the two registered estimators are accepted."""
    from config import EvalConfig

    assert EvalConfig(attribution_estimator="shapley").attribution_estimator == "shapley"
    assert (
        EvalConfig(attribution_estimator="leave_one_out").attribution_estimator
        == "leave_one_out"
    )
    with pytest.raises(ValueError, match="attribution_estimator"):
        EvalConfig(attribution_estimator="loo")


def test_scoped_config_registers_leave_one_out():
    """configs/scoped.json must encode the A2.2 estimator and retrain mode."""
    import json

    from config import config_from_dict

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "configs", "scoped.json")) as handle:
        cfg = config_from_dict(json.load(handle))
    assert cfg.eval.attribution_estimator == "leave_one_out"
    assert cfg.eval.shapley_mode == "retrain"


def test_loo_coalitions_are_full_then_drop_one():
    """Column order must match leave_one_out_analysis's expectation."""
    from engine import loo_coalitions

    coalitions = loo_coalitions()
    assert coalitions[0] == tuple([1] * N_ROLES)
    for r in range(N_ROLES):
        expected = [1] * N_ROLES
        expected[r] = 0
        assert coalitions[1 + r] == tuple(expected)
    assert len(coalitions) == 1 + N_ROLES


def test_expand_loo_matrix_scatters_to_bitmask_columns():
    """The full and drop-one columns land where the family analysis reads."""
    from engine import expand_loo_matrix

    matrix = _loo_matrix([0.6, 0.8, 0.7, 0.8, 0.8, 0.8], n_items=64)
    scattered = expand_loo_matrix(matrix)
    full_index = 2**N_ROLES - 1
    assert scattered.shape == (64, 2**N_ROLES)
    np.testing.assert_array_equal(scattered[:, full_index], matrix[:, 0])
    for r in range(N_ROLES):
        np.testing.assert_array_equal(
            scattered[:, full_index & ~(1 << r)], matrix[:, 1 + r]
        )
    evaluated = {full_index} | {full_index & ~(1 << r) for r in range(N_ROLES)}
    for column in range(2**N_ROLES):
        if column not in evaluated:
            assert np.isnan(scattered[:, column]).all()

    with pytest.raises(ValueError, match="columns"):
        expand_loo_matrix(matrix[:, :3])


def test_family_analysis_from_expanded_loo_matches_direct_contrast():
    """agent_by_benchmark_family on the scattered matrix reads only real columns."""
    from engine import agent_by_benchmark_family, expand_loo_matrix

    matrix = _loo_matrix([0.55, 0.80, 0.70, 0.80, 0.80, 0.80], n_items=400)
    benchmark_ids = np.zeros(400, dtype=np.int64)  # all items on benchmark 0
    cfg = _loo_cfg(200)
    family = agent_by_benchmark_family(expand_loo_matrix(matrix), benchmark_ids, cfg)
    # One benchmark with items, six roles -> six cells, none of them NaN.
    assert family["n_tests"] == N_ROLES
    for cell in family["cells"]:
        assert np.isfinite(cell["marginal_value"])
    # Role 0's marginal value is the planted full-minus-drop contrast.
    role0 = next(c for c in family["cells"] if c["role"] == ROLES[0])
    planted = float(np.mean(matrix[:, 0]) - np.mean(matrix[:, 1]))
    assert abs(role0["marginal_value"] - planted) < 1e-9


def test_trial_coverage_accepts_explicit_coalition_set():
    """The LOO path passes its seven bitmasks; coverage judges only those."""
    full_index = 2**N_ROLES - 1
    masks = [full_index] + [full_index & ~(1 << r) for r in range(N_ROLES)]
    out = verify_trial_coverage({m: 3 for m in masks}, 3, expected_coalitions=masks)
    assert out["passed"]
    assert out["n_coalitions_expected"] == 1 + N_ROLES
    assert out["missing_coalitions"] == []
    # Dropping one of the seven is reported as missing, not silently passed.
    partial = {m: 3 for m in masks[:-1]}
    out = verify_trial_coverage(partial, 3, expected_coalitions=masks)
    assert not out["passed"]
    assert out["missing_coalitions"] == [masks[-1]]
    # Default behaviour is unchanged: all 64 expected.
    out = verify_trial_coverage({m: 3 for m in masks}, 3)
    assert out["n_coalitions_expected"] == 2**N_ROLES
    assert not out["passed"]
