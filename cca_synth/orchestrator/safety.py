"""Operational safety controls (ETHICS_AND_DATA_GOVERNANCE.md, "Safety and
accountability").

Four requirements from the ethics addendum that are engineering, not policy:

* **a kill switch** — a way to stop the council immediately, from outside the
  process, without waiting for an in-flight batch to finish;
* **a single-agent fallback** — when the council is degraded or under attack,
  degrade to one agent rather than to nothing, and *say so in the record*;
* **preservation of the evidence behind rejected dissenting outputs** — a
  dissent that the synthesiser overrules must remain reconstructible;
* **an accountable owner** — recorded in every run so responsibility is not
  diffused across six agents.

The design principle throughout: a degraded run must be *visibly* degraded. A
fallback that silently produces a plausible answer is worse than a failure,
because the failure would have been noticed. Every control here writes a flag
into the record, and :func:`degradation_report` surfaces those flags as a
first-class result rather than a log line.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "KillSwitch",
    "KillSwitchTripped",
    "DegradationState",
    "CouncilHealth",
    "single_agent_fallback",
    "preserve_dissent",
    "degradation_report",
    "SAFETY_FLAGS",
]

#: Flags written into a record when a safety control fires. All are reported;
#: none is silent.
SAFETY_FLAGS = (
    "kill_switch_tripped",
    "degraded_to_single_agent",
    "quorum_not_met",
    "injection_rate_exceeded",
    "dissent_overruled",
)


class KillSwitchTripped(RuntimeError):
    """Raised when a run is halted by the kill switch."""


@dataclass
class KillSwitch:
    """An out-of-band stop for a running council.

    Three independent triggers, because the one that is reachable depends on how
    the run failed:

    * a **sentinel file** — works when the process is unreachable but the
      filesystem is not, which is the usual case for a long batch run;
    * an **environment variable** — works for a supervisor that can restart the
      process;
    * a **programmatic trip** — works from a signal handler or another thread.

    Checked between items, never mid-item, so that a halted run leaves complete
    records rather than a truncated one.

    Attributes:
        sentinel_path: File whose existence trips the switch.
        env_var: Environment variable whose truthy value trips the switch.
        _tripped: Programmatic trip flag.
        _reason: Why the switch tripped.
    """

    sentinel_path: str = "STOP_COUNCIL"
    env_var: str = "CCA_KILL_SWITCH"
    _tripped: bool = field(default=False, repr=False)
    _reason: str = field(default="", repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def trip(self, reason: str = "programmatic") -> None:
        """Trip the switch from within the process.

        Args:
            reason: Recorded in the run artifact.
        """
        with self._lock:
            self._tripped = True
            self._reason = reason

    def reset(self) -> None:
        """Clear a programmatic trip. Does not remove the sentinel file."""
        with self._lock:
            self._tripped = False
            self._reason = ""

    @property
    def tripped(self) -> bool:
        """Whether any trigger is active.

        Returns:
            ``True`` when the run should stop.
        """
        if self._tripped:
            return True
        if os.environ.get(self.env_var, "").strip().lower() in ("1", "true", "yes"):
            return True
        return os.path.exists(self.sentinel_path)

    @property
    def reason(self) -> str:
        """Why the switch is tripped.

        Returns:
            A short human-readable reason, or the empty string.
        """
        if self._tripped:
            return self._reason
        if os.environ.get(self.env_var, "").strip().lower() in ("1", "true", "yes"):
            return f"environment variable {self.env_var} set"
        if os.path.exists(self.sentinel_path):
            return f"sentinel file {self.sentinel_path} present"
        return ""

    def check(self) -> None:
        """Raise if the switch is tripped.

        Called between items by the orchestrator.

        Raises:
            KillSwitchTripped: When any trigger is active.
        """
        if self.tripped:
            raise KillSwitchTripped(
                f"council halted by kill switch: {self.reason}. "
                "Records written so far are complete and usable; the run is "
                "partial and must be reported as such."
            )


@dataclass
class DegradationState:
    """Why and how a council run was degraded, carried into the record.

    Attributes:
        degraded: Whether any fallback fired.
        reason: Short machine-readable reason.
        surviving_roles: Roles that produced a usable record.
        fallback_role: Role used for the single-agent fallback, if any.
    """

    degraded: bool = False
    reason: str = ""
    surviving_roles: List[str] = field(default_factory=list)
    fallback_role: str = ""

    def as_flags(self) -> List[str]:
        """Safety flags implied by this state.

        Returns:
            A list drawn from :data:`SAFETY_FLAGS`.
        """
        flags: List[str] = []
        if self.degraded:
            flags.append("degraded_to_single_agent")
        if self.reason == "quorum_not_met":
            flags.append("quorum_not_met")
        if self.reason == "injection_rate_exceeded":
            flags.append("injection_rate_exceeded")
        return flags


@dataclass
class CouncilHealth:
    """Health thresholds that trigger the single-agent fallback.

    Attributes:
        min_quorum: Minimum agents that must produce a usable record before the
            council's output is trusted. Below this, the synthesiser is
            aggregating too little evidence to be doing what it claims.
        max_injection_rate: Fraction of agent outputs flagged as containing
            suspected injection above which the council is treated as under
            attack. Inter-agent text is untrusted input; a council whose members
            are relaying an attack to each other should not be aggregated.
    """

    min_quorum: int = 3
    max_injection_rate: float = 0.34

    def assess(
        self, present_roles: Sequence[str], injection_flags: Sequence[bool]
    ) -> DegradationState:
        """Decide whether a council result is trustworthy.

        Args:
            present_roles: Roles that produced a usable record.
            injection_flags: Per-agent suspected-injection flags.

        Returns:
            The degradation state; ``degraded=False`` when the council is healthy.
        """
        n = len(injection_flags)
        rate = sum(1 for f in injection_flags if f) / n if n else 0.0

        if rate > self.max_injection_rate:
            return DegradationState(
                degraded=True,
                reason="injection_rate_exceeded",
                surviving_roles=list(present_roles),
            )
        if len(present_roles) < self.min_quorum:
            return DegradationState(
                degraded=True,
                reason="quorum_not_met",
                surviving_roles=list(present_roles),
            )
        return DegradationState(degraded=False, surviving_roles=list(present_roles))


def single_agent_fallback(
    candidates: Sequence[Mapping[str, Any]],
    state: DegradationState,
    *,
    preferred_role: str = "",
) -> Dict[str, Any]:
    """Produce a single-agent answer when the council is not usable.

    The fallback deliberately does **not** try to salvage a council answer from
    the surviving agents. Aggregating two records with a synthesiser trained on
    six is out of distribution, and the resulting confidence would be
    meaningless. One agent's answer, clearly labelled as such, is honest; a
    council answer built from a broken council is not.

    Args:
        candidates: Candidate records from the attempted council run.
        state: The degradation state that triggered the fallback.
        preferred_role: Role to prefer as the fallback agent, when present.

    Returns:
        A record carrying the fallback answer, the safety flags, and the reason.
        Returns an abstention when no candidate is usable at all.
    """
    usable = [c for c in candidates if c.get("present") and not c.get("abstain")]
    chosen: Optional[Mapping[str, Any]] = None

    if preferred_role:
        chosen = next((c for c in usable if c.get("role") == preferred_role), None)
    if chosen is None and usable:
        # Highest declared confidence among survivors. Not a quality judgement:
        # it is a tie-break, and the record says the answer is single-agent.
        chosen = max(usable, key=lambda c: float(c.get("confidence", 0.0)))

    flags = list(state.as_flags())
    if chosen is None:
        return {
            "answer_option": -1,
            "abstain": True,
            "confidence": 0.0,
            "fallback": True,
            "fallback_role": "",
            "failure_flags": flags,
            "degradation_reason": state.reason or "no_usable_candidate",
            "note": (
                "Council degraded and no single agent produced a usable answer. "
                "Abstaining. This is a refusal, not a low-confidence answer."
            ),
        }

    return {
        "answer_option": chosen.get("answer_option", -1),
        "abstain": False,
        # Confidence is NOT carried over from the agent. A single agent's
        # self-reported confidence was calibrated (if at all) in the context of
        # a council; it does not transfer. It is passed through unchanged but
        # explicitly marked uncalibrated so no downstream metric treats it as a
        # calibrated probability.
        "confidence": float(chosen.get("confidence", 0.0)),
        "confidence_is_calibrated": False,
        "fallback": True,
        "fallback_role": chosen.get("role", ""),
        "failure_flags": flags,
        "degradation_reason": state.reason,
        "surviving_roles": list(state.surviving_roles),
        "note": (
            "Single-agent fallback. This answer was produced by one agent, not "
            "by the council, and must not be pooled with council results."
        ),
    }


def preserve_dissent(
    candidates: Sequence[Mapping[str, Any]],
    accepted_option: int,
) -> List[Dict[str, Any]]:
    """Retain the evidence behind every overruled dissenting candidate.

    The ethics addendum requires preserving the evidence behind dissenting
    outputs when the synthesiser rejects them. The failure mode this prevents is
    a council that looked unanimous in the artifact because the disagreement was
    dropped at aggregation time -- which would also make the error-correlation
    and interaction analyses wrong, since both need the dissent.

    Args:
        candidates: Candidate records considered by the synthesiser.
        accepted_option: The option the synthesiser accepted.

    Returns:
        One entry per dissenting candidate, carrying its answer, its
        counterevidence, and its unsupported-claim flags. Empty when the council
        agreed.
    """
    preserved: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.get("present") or candidate.get("abstain"):
            continue
        option = candidate.get("answer_option", -1)
        if option == accepted_option or option < 0:
            continue
        claims = candidate.get("atomic_claims", []) or []
        preserved.append(
            {
                "role": candidate.get("role", ""),
                "dissenting_option": option,
                "confidence": float(candidate.get("confidence", 0.0)),
                "counterevidence": list(candidate.get("counterevidence", []) or []),
                "failure_flags": list(candidate.get("failure_flags", []) or []),
                "n_claims": len(claims),
                "n_claims_contradicted": sum(
                    1 for c in claims if c.get("evidence_status") == "contradicted"
                ),
                "n_claims_unsupported": sum(
                    1 for c in claims if c.get("evidence_status") == "unsupported"
                ),
            }
        )
    return preserved


def degradation_report(
    per_item_states: Sequence[DegradationState],
    kill_switch: Optional[KillSwitch] = None,
    *,
    n_items_planned: int = 0,
) -> Dict[str, Any]:
    """Summarise degradation across a run, as a first-class result.

    Args:
        per_item_states: Degradation state per item.
        kill_switch: The run's kill switch, if one was used.
        n_items_planned: Items the run intended to process, so that a halted
            run reports its own incompleteness.

    Returns:
        JSON-serialisable report.
    """
    degraded = [s for s in per_item_states if s.degraded]
    by_reason: Dict[str, int] = {}
    for state in degraded:
        by_reason[state.reason] = by_reason.get(state.reason, 0) + 1

    n = len(per_item_states)
    halted = bool(kill_switch and kill_switch.tripped)
    return {
        "n_items_processed": n,
        "n_items_planned": n_items_planned or n,
        "run_complete": (not halted) and (not n_items_planned or n >= n_items_planned),
        "kill_switch_tripped": halted,
        "kill_switch_reason": kill_switch.reason if kill_switch else "",
        "n_degraded_items": len(degraded),
        "degradation_rate": len(degraded) / n if n else float("nan"),
        "degraded_by_reason": by_reason,
        "note": (
            "Items marked degraded were answered by a single agent, not by the "
            "council, and are excluded from council-versus-control contrasts. "
            "A run with a non-zero degradation rate has a smaller effective "
            "sample than its item count suggests, and the primary contrast must "
            "report the reduced n."
        ),
    }
