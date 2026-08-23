# Role prompts and manipulation checks

Six functional roles, three prompt variants each, and one executable check per
role that decides whether the operation actually happened.

> These roles are functional AI processing constraints. They are not diagnostic
> models, simulations of lived experience, or claims that mental health
> conditions confer a single cognitive mechanism.

## The gate

**A role that fails its manipulation check is not implementing its operation,
whatever it scores on a benchmark.** `PREREGISTRATION.md` §1 treats passing as a
precondition for interpreting any agent-level result, and
`prompts/manipulation_check.py` exits non-zero when any role fails, so it can
gate a collection run rather than being a report nobody reads.

This matters because limitation L3 is that the IPA formalism is descriptive:
`f(C_i, y, x)` is unconstrained, so almost any prompt behaviour could be
relabelled an IPA after the fact. The check is what makes each role falsifiable
*before* its benchmark number is interpreted.

## Three variants, differing in exactly one respect

| Variant | What it contains | Which condition it serves |
| --- | --- | --- |
| `operation` | The predeclared processing instruction | 5 — the council under test |
| `neutral` | Same output contract, no processing instruction, length-matched | 2 — matched deliberation tokens; also the reference for every check |
| `persona` | A label only ("you are a careful checker"), no procedure, length-matched | 6 — persona-label-only council |

Length matching is enforced by test to within 10%. Without it, a measured
benefit could be bought with tokens rather than with the operation.

The `persona` variant deliberately contains no procedure. If it leaked one,
condition 6 would stop isolating the label from the operation and RQ3 would be
untestable. A test asserts the string `Procedure:` never appears in it.

## What each check measures

Taken from the "Manipulation check" column of `RESEARCH_SPECIFICATION.md` §1,
implemented literally rather than in whatever form would be easiest to pass.

| Role | Check | Guard against gaming |
| --- | --- | --- |
| `counter_evidence_checker` | Counterclaims + unsupported/contradicted flags per record | — |
| `consistency_checker` | Contradiction-flag rate **minus abstention rate** | Cannot pass by flagging everything and refusing |
| `local_claim_checker` | Atomic claims weighted by fraction carrying a named local check | Claim count alone is not enough |
| `explore_then_prune` | Distinct claims × evidence concentration | Measures both passes, not just breadth |
| `rare_hypothesis_checker` | Sourced supported claims **minus unsourced** ones | Fabricated alternatives subtract |
| `confidence_calibrator` | Negative expected calibration error | Uniform under-confidence does not improve ECE, so it cannot pass by hedging |

The last two guards are the ones that matter most. `REFERENCE_CORRECTIONS.md`
notes that the proposed uniform confidence penalty is an engineering hypothesis
that the psychological source does not validate; scoring on ECE rather than on
mean confidence is what keeps that hypothesis testable instead of assumed. A
test asserts that an agent hedging at 0.2 while being right 80% of the time
scores *worse* than one calibrated at 0.8.

## Running the checks

```bash
python -m prompts.manipulation_check --records dev_records.jsonl --out checks.json
```

Input is one JSON line per item:

```json
{"item_id": "dev-001",
 "counter_evidence_checker": {"operation": {...}, "neutral": {...}, "persona": {...}},
 "consistency_checker": {"operation": {...}, "neutral": {...}, "persona": {...}}}
```

Each variant value is a §7 candidate record. For `confidence_calibrator` the
records must also carry `"correct": true|false`, since ECE needs outcomes.

Contrasts are paired at the item level and bootstrapped — the same statistics
used for benchmark contrasts, so a check result is directly comparable. A role
passes at `diff ≥ 0.10` with `P(diff > 0) ≥ 0.95`, both pre-registered.

The report also gives `operation_over_label_share`: the operation-versus-persona
contrast divided by the operation-versus-neutral contrast. That is the RQ3
quantity at the prompt level — how much of the effect survives when the label is
held constant and only operational content is removed.

## Governance properties, enforced in code

- **No clinical vocabulary.** `assert_no_clinical_language` runs at import over
  every rendered prompt and raises on any of 33 prohibited terms. A prompt
  cannot be added that reintroduces it.
- **No chain-of-thought.** No prompt solicits reasoning, and `output_schema`
  sets `additionalProperties: false` at both levels, so a model-emitted
  `reasoning` field is a validation error rather than something stripped
  downstream. This is the substitution `ETHICS_AND_DATA_GOVERNANCE.md` requires.
- **Injection boundary.** Every prompt declares that supplied text is untrusted
  data and never instruction, and requires `injection_suspected` to be flagged.
  Inter-agent text gets the same boundary as external content.
- **Versioned.** `PROMPT_VERSION` plus a per-variant content hash reach the run
  manifest, and `prompt_version` is already in the audit-record allowlist, so a
  stored record traces to the exact instruction that produced it.

## Changing a prompt

Bump `PROMPT_VERSION`. Records collected under different versions are not
comparable and must not be pooled. If a change is made after any test-set result
has been observed, `PREREGISTRATION.md` §11 requires it to be recorded as a
dated amendment and the affected analyses reported as exploratory.
