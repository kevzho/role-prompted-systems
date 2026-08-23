"""Tests for the role prompts, their controls, and the manipulation checks.

The governance properties here are enforceable in code and therefore tested in
code: no prompt may contain clinical vocabulary, no schema may accept a
chain-of-thought field, and the controls must be length-matched, since an
unmatched control confounds "the operation helped" with "more tokens helped".
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROLES  # noqa: E402
from prompts import (  # noqa: E402
    CLINICAL_TERMS,
    PROMPT_VARIANTS,
    ROLE_PROMPTS,
    assert_no_clinical_language,
    load_prompt,
    output_schema,
    prompt_manifest,
)
from prompts.manipulation_check import MIN_EFFECT, check_role  # noqa: E402


def test_every_role_has_all_three_variants() -> None:
    """The six functional roles each need operation, neutral, and persona."""
    assert set(ROLE_PROMPTS) == set(ROLES)
    for role in ROLES:
        for variant in PROMPT_VARIANTS:
            assert load_prompt(role, variant).strip()


def test_no_prompt_contains_clinical_language() -> None:
    """Function-first vocabulary is mandatory in every rendered prompt."""
    for role in ROLES:
        for variant in PROMPT_VARIANTS:
            text = load_prompt(role, variant).lower()
            for term in CLINICAL_TERMS:
                assert term not in text, f"{role}/{variant} contains '{term}'"


def test_clinical_language_guard_actually_fires() -> None:
    """The guard must reject prohibited vocabulary, not merely exist."""
    for bad in ("You are an OCD agent.", "the manic phase", "a paranoid reasoner"):
        with pytest.raises(ValueError, match="prohibited clinical vocabulary"):
            assert_no_clinical_language(bad, "test")
    assert_no_clinical_language("You are a careful checker.", "test")


def test_controls_are_length_matched() -> None:
    """Neutral and persona controls must be within 10% of the operation prompt.

    Specification conditions 2 and 6 both require length matching; without it,
    any measured benefit could be bought with tokens rather than with the
    operation.
    """
    for role in ROLES:
        lengths = {v: len(load_prompt(role, v)) for v in PROMPT_VARIANTS}
        for control in ("neutral", "persona"):
            ratio = lengths[control] / lengths["operation"]
            assert 0.9 <= ratio <= 1.1, f"{role}/{control} length ratio {ratio:.2f}"


def test_persona_control_carries_no_operational_content() -> None:
    """The persona variant must supply a label and no procedure.

    If the persona prompt leaked the procedure, condition 6 would no longer
    isolate the label from the operation and RQ3 would be untestable.
    """
    for role in ROLES:
        persona = load_prompt(role, "persona")
        assert "Procedure:" not in persona, f"{role} persona control leaks a procedure"
        assert "No specific procedure is prescribed" in persona


def test_prompts_never_request_reasoning() -> None:
    """No prompt may ask for chain-of-thought, and the schema must reject it."""
    requests_for_reasoning = (
        "think step by step",
        "show your reasoning",
        "show your work",
        "explain your thought process",
        "let's think",
        "before answering, reason",
    )
    for role in ROLES:
        for variant in PROMPT_VARIANTS:
            text = load_prompt(role, variant).lower()
            # The prohibition must be present ...
            assert "do not include any field describing your reasoning" in text, (
                f"{role}/{variant} does not forbid reasoning fields"
            )
            # ... and nothing may solicit chain-of-thought.
            for phrase in requests_for_reasoning:
                assert phrase not in text, f"{role}/{variant} solicits '{phrase}'"

    schema = output_schema(4)
    assert schema["additionalProperties"] is False
    assert "reasoning" not in schema["properties"]
    assert schema["properties"]["atomic_claims"]["items"]["additionalProperties"] is False


def test_output_schema_bounds_the_answer_index() -> None:
    """The schema must not admit an option index outside the item's range."""
    schema = output_schema(3)
    answer = schema["properties"]["answer_option"]
    assert answer["minimum"] == -1  # -1 is abstention
    assert answer["maximum"] == 2


def test_manipulation_checks_discriminate() -> None:
    """Each check must separate a record exhibiting its behaviour from one not.

    This is a test of the *checks*, not of any model: a check that cannot
    distinguish its target behaviour is unable to validate a role.
    """
    strong = {
        "counter_evidence_checker": {
            "counterevidence": ["a", "b", "c"],
            "atomic_claims": [
                {"claim": "x", "evidence_status": "unsupported", "source_or_check": "check"},
                {"claim": "y", "evidence_status": "contradicted", "source_or_check": "check"},
            ],
        },
        "consistency_checker": {
            "failure_flags": ["contradiction"], "abstain": False, "atomic_claims": [],
        },
        "local_claim_checker": {
            "atomic_claims": [
                {"claim": f"c{i}", "evidence_status": "supported", "source_or_check": "local"}
                for i in range(6)
            ],
        },
        "explore_then_prune": {
            "atomic_claims": [
                {"claim": f"h{i}", "evidence_status": "supported"} for i in range(4)
            ],
        },
        "rare_hypothesis_checker": {
            "atomic_claims": [
                {"claim": "alt", "evidence_status": "supported", "source_or_check": "src"},
                {"claim": "alt2", "evidence_status": "supported", "source_or_check": "src"},
            ],
        },
        "confidence_calibrator": {"confidence": 0.9, "correct": True, "atomic_claims": []},
    }
    weak = {
        "counter_evidence_checker": {"counterevidence": [], "atomic_claims": []},
        "consistency_checker": {"failure_flags": [], "abstain": True, "atomic_claims": []},
        "local_claim_checker": {"atomic_claims": [{"claim": "c", "evidence_status": "supported"}]},
        "explore_then_prune": {
            "atomic_claims": [{"claim": "h", "evidence_status": "uncertain"}]
        },
        "rare_hypothesis_checker": {
            "atomic_claims": [{"claim": "alt", "evidence_status": "supported"}]
        },
        "confidence_calibrator": {"confidence": 0.9, "correct": False, "atomic_claims": []},
    }

    for role, spec in ROLE_PROMPTS.items():
        high = spec.check([strong[role]])
        low = spec.check([weak[role]])
        assert high > low, (
            f"{role}: check '{spec.check_name}' does not separate target "
            f"behaviour ({high}) from its absence ({low})"
        )


def test_calibration_check_rejects_uniform_underconfidence() -> None:
    """Lowering confidence everywhere must not pass the calibration check.

    The specification is explicit: "Lower expected calibration error on held-out
    calibration data, not merely lower average confidence."
    """
    spec = ROLE_PROMPTS["confidence_calibrator"]
    # An agent right 80% of the time. Well calibrated at 0.8, badly at 0.2.
    calibrated = [{"confidence": 0.8, "correct": i < 80} for i in range(100)]
    timid = [{"confidence": 0.2, "correct": i < 80} for i in range(100)]
    assert spec.check(calibrated) > spec.check(timid)


def test_check_role_rejects_misaligned_variants() -> None:
    """Unequal item counts across variants must raise, not silently truncate."""
    with pytest.raises(ValueError, match="not item-aligned"):
        check_role(
            "counter_evidence_checker",
            {"operation": [{}], "neutral": [{}, {}], "persona": [{}]},
        )


def test_check_role_produces_a_paired_verdict() -> None:
    """A role whose operation clearly moves its check must pass."""
    n = 60
    records = {
        "operation": [
            {"counterevidence": ["a", "b"], "atomic_claims": [
                {"claim": "x", "evidence_status": "unsupported"}]}
            for _ in range(n)
        ],
        "neutral": [{"counterevidence": [], "atomic_claims": []} for _ in range(n)],
        "persona": [{"counterevidence": ["a"], "atomic_claims": []} for _ in range(n)],
    }
    result = check_role("counter_evidence_checker", records)
    assert result["passed"]
    assert result["vs_neutral"]["diff"] >= MIN_EFFECT
    assert result["vs_neutral"]["prob_positive"] >= 0.95
    # Persona carries some of the effect but not all; the share is the RQ3
    # quantity at prompt level.
    assert 0.0 < result["operation_over_label_share"] < 1.0


def test_prompt_manifest_is_stable_and_versioned() -> None:
    """The manifest must pin a version and a hash per rendered prompt."""
    manifest = prompt_manifest()
    assert manifest["prompt_version"]
    assert set(manifest["roles"]) == set(ROLES)
    for role in ROLES:
        for variant in PROMPT_VARIANTS:
            assert len(manifest["roles"][role][variant]) == 16
    assert prompt_manifest() == manifest, "manifest must be deterministic"
