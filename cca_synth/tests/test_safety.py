"""Tests for the operational safety controls (P20, ethics addendum).

The design principle these pin down: **a degraded run must be visibly
degraded.** A fallback that silently produces a plausible answer is worse than
a failure, because the failure would have been noticed. Every assertion here is
ultimately about whether the record says what happened.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROLES  # noqa: E402
from orchestrator.safety import (  # noqa: E402
    CouncilHealth,
    DegradationState,
    KillSwitch,
    KillSwitchTripped,
    degradation_report,
    preserve_dissent,
    single_agent_fallback,
)


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_starts_untripped(tmp_path, monkeypatch):
    """A fresh switch must not halt a run."""
    monkeypatch.delenv("CCA_KILL_SWITCH", raising=False)
    switch = KillSwitch(sentinel_path=str(tmp_path / "STOP"), env_var="CCA_KILL_SWITCH")
    assert switch.tripped is False
    switch.check()  # must not raise


def test_kill_switch_trips_on_sentinel_file(tmp_path, monkeypatch):
    """The filesystem trigger works when the process is otherwise unreachable."""
    monkeypatch.delenv("CCA_KILL_SWITCH", raising=False)
    sentinel = tmp_path / "STOP"
    switch = KillSwitch(sentinel_path=str(sentinel), env_var="CCA_KILL_SWITCH")
    sentinel.write_text("")
    assert switch.tripped is True
    with pytest.raises(KillSwitchTripped, match="sentinel file"):
        switch.check()


def test_kill_switch_trips_on_environment(tmp_path, monkeypatch):
    """The supervisor trigger works for a restartable process."""
    switch = KillSwitch(sentinel_path=str(tmp_path / "STOP"), env_var="CCA_KILL_SWITCH")
    monkeypatch.setenv("CCA_KILL_SWITCH", "1")
    assert switch.tripped is True
    with pytest.raises(KillSwitchTripped, match="environment variable"):
        switch.check()


def test_kill_switch_trips_programmatically(tmp_path, monkeypatch):
    """The in-process trigger works from a signal handler or another thread."""
    monkeypatch.delenv("CCA_KILL_SWITCH", raising=False)
    switch = KillSwitch(sentinel_path=str(tmp_path / "STOP"), env_var="CCA_KILL_SWITCH")
    switch.trip("operator halted")
    assert switch.tripped is True
    assert switch.reason == "operator halted"
    switch.reset()
    assert switch.tripped is False


def test_halted_run_reports_itself_incomplete(tmp_path, monkeypatch):
    """A partial run must not be mistaken for a complete one."""
    monkeypatch.delenv("CCA_KILL_SWITCH", raising=False)
    switch = KillSwitch(sentinel_path=str(tmp_path / "STOP"), env_var="CCA_KILL_SWITCH")
    switch.trip("out of budget")
    report = degradation_report(
        [DegradationState() for _ in range(40)], switch, n_items_planned=200
    )
    assert report["run_complete"] is False
    assert report["kill_switch_tripped"] is True
    assert report["n_items_processed"] == 40
    assert report["n_items_planned"] == 200


# ---------------------------------------------------------------------------
# Health assessment and fallback
# ---------------------------------------------------------------------------


def test_healthy_council_is_not_degraded():
    """The happy path must be reachable or the guard is useless."""
    health = CouncilHealth()
    state = health.assess(list(ROLES), [False] * 6)
    assert state.degraded is False
    assert state.as_flags() == []


def test_quorum_failure_degrades():
    """Too few surviving agents means the synthesiser has too little evidence."""
    state = CouncilHealth(min_quorum=3).assess(list(ROLES[:2]), [False, False])
    assert state.degraded is True
    assert state.reason == "quorum_not_met"
    assert "quorum_not_met" in state.as_flags()
    assert "degraded_to_single_agent" in state.as_flags()


def test_injection_rate_degrades():
    """A council relaying an attack between members must not be aggregated."""
    state = CouncilHealth(max_injection_rate=0.34).assess(
        list(ROLES), [True, True, True, False, False, False]
    )
    assert state.degraded is True
    assert state.reason == "injection_rate_exceeded"


def test_fallback_marks_confidence_uncalibrated():
    """A single agent's confidence was calibrated in a council context, if at all."""
    state = DegradationState(degraded=True, reason="quorum_not_met",
                             surviving_roles=[ROLES[0]])
    candidates = [
        {"role": ROLES[0], "present": True, "abstain": False,
         "answer_option": 2, "confidence": 0.9},
    ]
    out = single_agent_fallback(candidates, state)
    assert out["fallback"] is True
    assert out["answer_option"] == 2
    assert out["confidence_is_calibrated"] is False
    assert "must not be pooled with council results" in out["note"]


def test_fallback_abstains_when_nothing_is_usable():
    """No usable answer is a refusal, not a low-confidence guess."""
    state = DegradationState(degraded=True, reason="quorum_not_met")
    out = single_agent_fallback(
        [{"role": ROLES[0], "present": False}], state
    )
    assert out["abstain"] is True
    assert out["answer_option"] == -1
    assert "Abstaining" in out["note"]


def test_fallback_does_not_salvage_a_council_answer():
    """Aggregating survivors would be out of distribution for the synthesiser."""
    state = DegradationState(degraded=True, reason="quorum_not_met")
    candidates = [
        {"role": ROLES[0], "present": True, "abstain": False,
         "answer_option": 1, "confidence": 0.4},
        {"role": ROLES[1], "present": True, "abstain": False,
         "answer_option": 3, "confidence": 0.8},
    ]
    out = single_agent_fallback(candidates, state)
    # One agent's answer, not a vote between the two survivors.
    assert out["answer_option"] == 3
    assert out["fallback_role"] == ROLES[1]


# ---------------------------------------------------------------------------
# Dissent preservation
# ---------------------------------------------------------------------------


def test_dissent_is_preserved_with_its_evidence():
    """A rejected dissent must remain reconstructible from the record."""
    candidates = [
        {"role": ROLES[0], "present": True, "abstain": False, "answer_option": 1,
         "confidence": 0.8, "counterevidence": [], "atomic_claims": []},
        {"role": ROLES[1], "present": True, "abstain": False, "answer_option": 3,
         "confidence": 0.6, "counterevidence": ["the source contradicts this"],
         "failure_flags": ["unsupported_claim"],
         "atomic_claims": [
             {"evidence_status": "contradicted"},
             {"evidence_status": "unsupported"},
             {"evidence_status": "supported"},
         ]},
    ]
    preserved = preserve_dissent(candidates, accepted_option=1)
    assert len(preserved) == 1
    entry = preserved[0]
    assert entry["role"] == ROLES[1]
    assert entry["dissenting_option"] == 3
    assert entry["counterevidence"] == ["the source contradicts this"]
    assert entry["n_claims_contradicted"] == 1
    assert entry["n_claims_unsupported"] == 1


def test_unanimous_council_preserves_nothing():
    """No dissent, nothing to preserve."""
    candidates = [
        {"role": r, "present": True, "abstain": False, "answer_option": 2,
         "confidence": 0.7, "atomic_claims": []}
        for r in ROLES
    ]
    assert preserve_dissent(candidates, accepted_option=2) == []


def test_abstainers_are_not_counted_as_dissenters():
    """Declining to answer is not disagreeing."""
    candidates = [
        {"role": ROLES[0], "present": True, "abstain": True, "answer_option": -1,
         "confidence": 0.1, "atomic_claims": []},
    ]
    assert preserve_dissent(candidates, accepted_option=2) == []


# ---------------------------------------------------------------------------
# Degradation reporting
# ---------------------------------------------------------------------------


def test_degradation_report_states_the_reduced_effective_sample():
    """Degraded items are excluded from council contrasts, so n is smaller."""
    states = [DegradationState(degraded=True, reason="quorum_not_met")] * 5
    states += [DegradationState() for _ in range(15)]
    report = degradation_report(states)
    assert report["n_degraded_items"] == 5
    assert report["degradation_rate"] == pytest.approx(0.25)
    assert report["degraded_by_reason"]["quorum_not_met"] == 5
    assert "effective sample" in report["note"]
