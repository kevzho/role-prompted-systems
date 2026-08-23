"""Tests for the §3 comparator conditions.

The properties here are the ones that would confound the primary contrast if
they broke: a matched control that is not matched, a self-consistency arm whose
samples are identical, a debate arm that communicates when it claims not to, and
a baseline join that imputes a value for `v(∅)` instead of recording it missing.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import N_ROLES, ROLES  # noqa: E402

# Scratch directory for arm-record IO.
#
# These tests previously wrote to hardcoded ``/tmp/cca_t_*.jsonl`` paths. That is
# a defect independent of any disk-space problem: two concurrent runs of the
# suite share those paths and silently overwrite each other's fixtures. Writing
# under the repository keeps the artifacts on the same volume as the code and
# makes each run self-contained.
_SCRATCH_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".pytest-scratch"
)
os.makedirs(_SCRATCH_ROOT, exist_ok=True)
_TMP = tempfile.mkdtemp(prefix="conditions-", dir=_SCRATCH_ROOT)
from dataset import parse_item  # noqa: E402
from orchestrator import Council, Item, StubProvider  # noqa: E402
from orchestrator.conditions import (  # noqa: E402
    ArmRecord,
    DebateCouncil,
    DirectAnswer,
    MatchedDeliberation,
    SelfConsistency,
    load_arm_records,
    merge_base_answers,
    write_arm_records,
)
from orchestrator.providers import Provider, ProviderResponse  # noqa: E402


def _items(n: int = 12) -> list:
    """Build n fixed-option items."""
    return [
        Item(item_id=f"x-{i:04d}", benchmark="math_aime", question=f"q{i}",
             options=["a", "b", "c", "d"], gold=i % 4)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Condition 1: direct answer, and the v(empty) join
# ---------------------------------------------------------------------------


def test_direct_answer_produces_one_record_per_item() -> None:
    """Condition 1 emits a single-answer arm record per item."""
    condition = DirectAnswer(StubProvider(seed=1, failure_rate=0.0, malformed_rate=0.0))
    manifest = condition.run(_items(10), f"{_TMP}/cca_t_c1.jsonl")
    assert manifest["n_items"] == 10
    records = load_arm_records(f"{_TMP}/cca_t_c1.jsonl")
    assert len(records) == 10
    assert all(r.condition == "direct_answer" for r in records.values())


def test_direct_answer_concurrency_and_resume_are_ordered() -> None:
    """The paid baseline arm should restart without duplicates or reordering."""
    path = f"{_TMP}/cca_t_c1_resume.jsonl"
    DirectAnswer(StubProvider(seed=21), max_workers=2).run(_items(4), path)
    condition = DirectAnswer(StubProvider(seed=21), max_workers=4)
    manifest = condition.run(_items(7), path, resume=True)

    with open(path, encoding="utf-8") as handle:
        ids = [json.loads(line)["item_id"] for line in handle]
    assert ids == [item.item_id for item in _items(7)]
    assert manifest["n_items"] == 7
    assert manifest["n_items_written"] == 3
    assert manifest["n_items_skipped_existing"] == 4
    assert manifest["counts"]["calls"] == 3
    assert manifest["max_workers"] == 4


def test_base_answer_join_populates_v_empty() -> None:
    """The join must fill base_answer_option, which defines v(empty).

    Without it the field stays -1 and the empty coalition falls back to nothing,
    silently replacing the pre-registered null policy with "no answer at all".
    """
    provider = StubProvider(seed=2, failure_rate=0.0, malformed_rate=0.0, truncation_rate=0.0)
    items = _items(60)
    Council({r: provider for r in ROLES}).run(items, f"{_TMP}/cca_t_council.jsonl")
    DirectAnswer(provider).run(items, f"{_TMP}/cca_t_base.jsonl")

    summary = merge_base_answers(
        f"{_TMP}/cca_t_council.jsonl", f"{_TMP}/cca_t_base.jsonl", f"{_TMP}/cca_t_merged.jsonl"
    )
    # Assert the accounting invariant rather than a coverage number tuned to the
    # stub's built-in abstention rate: every council record must be accounted
    # for as matched, missing, or unusable, with none silently dropped.
    assert (
        summary["base_answers_matched"]
        + summary["base_answers_missing"]
        + summary["base_answers_unusable"]
        == summary["n_council_records"]
    )
    assert summary["base_answers_matched"] > 0
    assert summary["base_answers_missing"] == 0  # every item has a baseline row
    with open(f"{_TMP}/cca_t_merged.jsonl", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    assert sum(1 for r in records if r["base_answer_option"] >= 0) == summary[
        "base_answers_matched"
    ]
    # And the merged records must still parse.
    for record in records:
        parse_item(record, max_options=6, max_claims=12)


def test_refusing_baseline_is_recorded_not_imputed() -> None:
    """A baseline that refused leaves v(empty) undefined, not guessed."""
    provider = StubProvider(seed=3, failure_rate=0.0, malformed_rate=0.0)
    items = _items(8)
    Council({r: provider for r in ROLES}).run(items, f"{_TMP}/cca_t_council2.jsonl")
    DirectAnswer(StubProvider(seed=4, failure_rate=1.0)).run(items, f"{_TMP}/cca_t_base2.jsonl")

    summary = merge_base_answers(
        f"{_TMP}/cca_t_council2.jsonl", f"{_TMP}/cca_t_base2.jsonl", f"{_TMP}/cca_t_merged2.jsonl"
    )
    assert summary["base_answers_matched"] == 0
    assert summary["base_answers_unusable"] == 8
    with open(f"{_TMP}/cca_t_merged2.jsonl", encoding="utf-8") as handle:
        assert all(json.loads(line)["base_answer_option"] == -1 for line in handle)


def test_missing_baseline_items_are_counted() -> None:
    """Items with no baseline at all are reported as missing."""
    provider = StubProvider(seed=5, failure_rate=0.0, malformed_rate=0.0)
    Council({r: provider for r in ROLES}).run(_items(10), f"{_TMP}/cca_t_council3.jsonl")
    write_arm_records(
        [ArmRecord(item_id="x-0000", condition="direct_answer", present=True, answer_option=1)],
        f"{_TMP}/cca_t_base3.jsonl",
    )
    summary = merge_base_answers(
        f"{_TMP}/cca_t_council3.jsonl", f"{_TMP}/cca_t_base3.jsonl", f"{_TMP}/cca_t_merged3.jsonl"
    )
    assert summary["base_answers_missing"] == 9
    assert summary["base_answers_matched"] == 1


# ---------------------------------------------------------------------------
# Condition 2: matched deliberation
# ---------------------------------------------------------------------------


def test_matched_deliberation_requires_a_measured_budget() -> None:
    """A guessed budget confounds the control, so it must be supplied."""
    provider = StubProvider(seed=6)
    with pytest.raises(ValueError, match="measured from a council run"):
        MatchedDeliberation(provider, matched_output_tokens=0)
    MatchedDeliberation(provider, matched_output_tokens=600)  # must not raise


def test_matched_deliberation_uses_the_supplied_budget() -> None:
    """The budget must reach the provider call, not merely be stored."""

    class _BudgetProbe(Provider):
        def __init__(self) -> None:
            super().__init__("probe-2026-07-28")
            self.max_tokens_seen: list = []

        def complete(self, system_prompt, user_content, max_tokens=1024,
                     temperature=0.0, seed=None):
            self.max_tokens_seen.append(max_tokens)
            return ProviderResponse(
                text=json.dumps({"answer_option": 0, "abstain": False, "confidence": 0.5,
                                 "atomic_claims": [], "counterevidence": [],
                                 "failure_flags": []}),
                model_snapshot=self.model_snapshot,
            )

    probe = _BudgetProbe()
    MatchedDeliberation(probe, matched_output_tokens=777).run(_items(3), f"{_TMP}/cca_t_c2.jsonl")
    assert probe.max_tokens_seen == [777, 777, 777]


# ---------------------------------------------------------------------------
# Condition 3: self-consistency
# ---------------------------------------------------------------------------


def test_self_consistency_rejects_zero_temperature() -> None:
    """At temperature 0 the k samples are identical and this is condition 1."""
    with pytest.raises(ValueError, match="temperature > 0"):
        SelfConsistency(StubProvider(seed=7), k=6, temperature=0.0)


def test_self_consistency_draws_k_samples_per_item() -> None:
    """k=6 must issue six calls per item, matching the council's agent count."""
    condition = SelfConsistency(StubProvider(seed=8, failure_rate=0.0), k=6, temperature=0.7)
    manifest = condition.run(_items(5), f"{_TMP}/cca_t_c3.jsonl")
    assert manifest["counts"]["calls"] == 30


def test_self_consistency_reports_agreement_as_confidence() -> None:
    """Sample agreement is the arm's calibrated signal, not self-report.

    Using the mean self-reported confidence instead would hand this arm a
    calibration advantage the method does not actually have.
    """
    condition = SelfConsistency(
        StubProvider(seed=9, failure_rate=0.0, malformed_rate=0.0), k=6, temperature=0.7
    )
    condition.run(_items(10), f"{_TMP}/cca_t_c3b.jsonl")
    for record in load_arm_records(f"{_TMP}/cca_t_c3b.jsonl").values():
        if record.present:
            assert record.confidence == pytest.approx(record.agreement)
            assert 0.0 < record.agreement <= 1.0
            assert record.n_samples == 6


def test_self_consistency_records_total_failure() -> None:
    """If every sample refuses, the item is absent rather than guessed."""
    condition = SelfConsistency(StubProvider(seed=10, failure_rate=1.0), k=6, temperature=0.7)
    condition.run(_items(4), f"{_TMP}/cca_t_c3c.jsonl")
    for record in load_arm_records(f"{_TMP}/cca_t_c3c.jsonl").values():
        assert record.present is False
        assert record.answer_option == -1
        assert "refusal" in record.failure_flags


# ---------------------------------------------------------------------------
# Condition 4: generic debate
# ---------------------------------------------------------------------------


def test_debate_emits_council_shaped_records() -> None:
    """Condition 4 must be comparable like-for-like with the CCA council."""
    condition = DebateCouncil(StubProvider(seed=11, failure_rate=0.0, malformed_rate=0.0))
    manifest = condition.run(_items(8), f"{_TMP}/cca_t_c4.jsonl")
    assert manifest["n_items"] == 8
    with open(f"{_TMP}/cca_t_c4.jsonl", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            assert record["condition"] == "debate_council"
            item = parse_item(record, max_options=6, max_claims=12)
            assert len(item.candidates) == N_ROLES


def test_debate_declares_its_communication_round() -> None:
    """The one place agents see each other must be explicit in the manifest."""
    two_round = DebateCouncil(StubProvider(seed=12), rounds=2)
    manifest = two_round.run(_items(3), f"{_TMP}/cca_t_c4b.jsonl")
    assert manifest["agents_isolated"] is False
    assert "declared communication round" in manifest["communication_round"]

    one_round = DebateCouncil(StubProvider(seed=13), rounds=1)
    manifest = one_round.run(_items(3), f"{_TMP}/cca_t_c4c.jsonl")
    assert manifest["agents_isolated"] is True
    assert "isolated throughout" in manifest["communication_round"]


def test_debate_round_one_is_isolated() -> None:
    """Agents must not see peers before the declared round."""

    class _Recorder(Provider):
        def __init__(self) -> None:
            super().__init__("recorder-2026-07-28")
            self.seen: list = []
            self.calls = 0

        def complete(self, system_prompt, user_content, max_tokens=1024,
                     temperature=0.0, seed=None):
            self.seen.append(user_content)
            self.calls += 1
            return ProviderResponse(
                text=json.dumps({"answer_option": 2, "abstain": False, "confidence": 0.7,
                                 "atomic_claims": [], "counterevidence": [],
                                 "failure_flags": []}),
                model_snapshot=self.model_snapshot,
            )

    recorder = _Recorder()
    DebateCouncil(recorder, rounds=2).run_item(_items(1)[0])

    first_round = recorder.seen[:N_ROLES]
    second_round = recorder.seen[N_ROLES:]
    assert all("Other assistants" not in content for content in first_round)
    assert all("Other assistants" in content for content in second_round)


def test_debate_rounds_must_be_one_or_two() -> None:
    """An undeclared multi-round protocol must not be silently accepted."""
    with pytest.raises(ValueError, match="rounds must be"):
        DebateCouncil(StubProvider(seed=14), rounds=3)


# ---------------------------------------------------------------------------
# Arm-record IO
# ---------------------------------------------------------------------------


def test_arm_records_round_trip() -> None:
    """Writing then loading arm records must preserve every field."""
    original = [
        ArmRecord(item_id="a", condition="direct_answer", answer_option=2,
                  confidence=0.7, present=True, output_tokens=120, retries=1),
        ArmRecord(item_id="b", condition="direct_answer", present=False,
                  failure_flags=["refusal"]),
    ]
    write_arm_records(original, f"{_TMP}/cca_t_arm.jsonl")
    loaded = load_arm_records(f"{_TMP}/cca_t_arm.jsonl")
    assert loaded["a"].answer_option == 2
    assert loaded["a"].retries == 1
    assert loaded["b"].present is False
    assert loaded["b"].failure_flags == ["refusal"]


def test_load_arm_records_reports_a_missing_file() -> None:
    """A missing arm file must raise rather than yield an empty comparison."""
    with pytest.raises(FileNotFoundError):
        load_arm_records(f"{_TMP}/definitely-not-here-cca.jsonl")
