# Minimum research specification

This document converts the PDF's proposal into a falsifiable, reproducible study. It is a protocol, not a report of completed work.

## 1. Scope and versions

### Portable V1: prompt and context operations

V1 may use only operations available through ordinary model APIs:

| PDF role | Functional label | Predeclared operation | Manipulation check |
| --- | --- | --- | --- |
| Sentinel | Counter-evidence checker | For every material factual claim, provide supporting evidence status and the strongest plausible counterclaim before recommending acceptance. | More explicit counterclaims and unsupported-claim flags than a length-matched neutral prompt. |
| Auditor | Consistency checker | Produce an answer, run two fixed independent invariant checks, and revise only when a named contradiction or calculation error is found. | Higher contradiction detection on seeded inconsistencies without excessive abstention. |
| Savant | Local-claim checker | Decompose the input into atomic claims, judge each from its local evidence, then reconcile with the full context. | Higher detection of locally unsupported claims; expected risk on long-range integration is also measured. |
| Oscillator | Explore-then-prune | Generate a fixed number of diverse hypotheses, then score them against a fixed evidence rubric and select or abstain. | Higher hypothesis diversity in pass one and improved evidence concentration in pass two. |
| Dissenter | Rare-hypothesis checker | Generate one plausible non-canonical alternative and require explicit evidence before it can displace the leading answer. | More valid alternative hypotheses without an unacceptable rise in fabricated alternatives. |
| Challenger | Confidence calibrator | Give an initial answer, list disconfirming conditions, and output a probability after checking those conditions. | Lower expected calibration error on held-out calibration data, not merely lower average confidence. |

The original clinical names remain identifiers for traceability only. Product-facing systems should use the functional labels.

### Optional V2: open-weight interventions

Decoding or activation-level mechanisms require a separate protocol and open-weight implementation. Record exact code for temperature schedules, logits transforms, attention interventions, context segmentation, and stopping. Do not describe early-layer attention temperature, low-frequency token upweighting, or beam diversity as implemented when only a system prompt is used.

## 2. Research questions

- RQ1: Does a full six-role council outperform the strongest single-agent and generic-ensemble controls on predeclared primary metrics?
- RQ2: Does each role provide positive marginal value on its predicted target after multiplicity correction?
- RQ3: Are gains attributable to processing operations rather than role labels, extra tokens, or the synthesizer?
- RQ4: What accuracy, calibration, safety, cost, and latency regressions accompany any gain?
- RQ5: Do results generalize across at least three model families and both homogeneous and heterogeneous councils?

Pre-register directional hypotheses and smallest effect sizes of interest before observing test results.

## 3. Experimental conditions

At minimum, compare:

1. Strongest single model, direct answer
2. Single model with matched deliberation tokens
3. Self-consistency with six samples
4. Generic six-agent debate or council
5. Functionally diverse CCA council
6. Persona-label-only council with matched text length
7. CCA operations with all clinical language removed
8. CCA candidates plus simple majority vote
9. CCA candidates plus the proposed synthesizer
10. Heterogeneous-model council with otherwise identical routing

Match budgets twice:

- Resource-matched: same maximum input tokens, output tokens, calls, and synthesis allowance
- Cost-matched: same total provider cost and comparable latency ceiling

Also report the unconstrained best result so the quality-cost frontier is visible.

## 4. Models and execution

- Use at least three model families from at least two providers when licenses and budget permit.
- Freeze exact model snapshot IDs. Do not use a moving alias as the only identifier.
- Store prompts, context assembly, tool access, sampling parameters, maximum tokens, stop rules, seed where supported, timeout, retry, and refusal handling.
- Separate development, calibration, validation, and final test sets.
- Run test items in randomized order. Cache no answer across conditions.
- Isolate agents from one another until the declared communication round.
- Disclose retrieval, browsing, calculators, and code execution per condition.

## 5. Outcomes

| Domain | Primary outcome | Secondary and harm outcomes |
| --- | --- | --- |
| TruthfulQA | Dataset-native truthful-and-informative score or a precisely versioned equivalent | Abstention, verbosity, judge disagreement |
| HaluEval | Dataset-native accuracy/F1 for the selected task and split | Claim-level false-positive/false-negative rates |
| MATH/AIME | Exact final-answer accuracy using official normalization | Invalid format, reasoning-token cost |
| AdvConsistency | Accuracy on the seeded local falsehood, with hidden labels | Long-range integration accuracy, position bias |
| Sycophancy | Correct answer retention after a controlled user challenge | Inappropriate stubbornness when the user is correct |
| Fresh or temporal QA | Accuracy against timestamped authoritative sources | Unsupported answer rate and retrieval dependence |
| Creative divergence | Blinded human originality score under a written rubric | Usefulness, feasibility, safety, inter-rater agreement |

Across all tasks, report calibration error, Brier score when probabilistic labels exist, selective accuracy/coverage, safety incidents, input/output tokens, p50/p95 latency, and dollar cost.

## 6. AdvConsistency construction

Before calling AdvConsistency a benchmark:

- Define the phenomenon independently of the proposed Savant role.
- Construct paired passages that differ only in a controlled local claim.
- Include positive controls where global context legitimately resolves local ambiguity.
- Balance answer position, topic, passage length, and contradiction location.
- Use domain experts for label creation where factual expertise is required.
- Pilot for ambiguity, then freeze a hidden test split.
- Deduplicate against public corpora and model development prompts as far as practical.
- Document licenses, personal-data handling, annotator instructions, pay, agreement, and adjudication.
- Publish a data card and baseline results. Keep a non-public test set or periodically refresh it.

## 7. Synthesis protocol

The synthesizer receives only structured candidate records:

```json
{
  "candidate_answer": "...",
  "atomic_claims": [
    {
      "claim": "...",
      "evidence_status": "supported|unsupported|contradicted|uncertain",
      "source_or_check": "...",
      "confidence": 0.0
    }
  ],
  "counterevidence": ["..."],
  "failure_flags": ["..."],
  "abstain": false
}
```

It must:

1. Normalize claims without seeing condition names or clinical labels.
2. Resolve factual conflicts using cited evidence or a declared verification tool.
3. Treat agent confidence as an input feature, not ground truth.
4. Record the rule used for every accepted or rejected material claim.
5. Abstain when evidence cannot resolve a high-impact conflict.

Fit any learned router only on the training/calibration data. Compare it with majority vote, confidence-weighted vote, and an oracle upper bound that is never presented as deployable performance.

## 8. Ablation and statistics

For the six-role set `A`, define the empty coalition value before the run. A defensible default is the performance of the direct-answer base model under the same per-item input.

- Evaluate all 64 coalitions only if each coalition receives enough repeated item-level trials.
- Reuse a fixed item/seed schedule across coalitions to reduce variance.
- Compute Shapley values per bootstrap resample from item-level observations.
- Report 95% confidence intervals and the probability that each contribution is positive.
- Test pre-registered primary contrasts with paired item-level methods.
- Control the false discovery rate across agent-by-benchmark secondary tests.
- Report effect sizes and intervals, not only p-values.
- Analyze pairwise error correlations and interaction terms; independent Shapley additivity must not be assumed.
- Publish all exclusions, failed calls, retries, and missing outputs.

Do not describe an agent as necessary or sufficient unless a predeclared decision rule, uncertainty interval, and replication support that wording.

## 9. Robustness and safety tests

Test at least:

- prompt injection delivered through another agent's output
- plausible but false minority hypotheses
- a confidently wrong majority
- evidence-free agreement and evidence-backed disagreement
- order and role-label permutations
- prompt paraphrases and neutral functional labels
- malicious or privacy-sensitive user inputs
- long context, irrelevant context, and contradictory context
- provider refusal and truncated output
- synthesizer capture by one verbose agent

The council must not be deployed in medical, legal, financial, or other high-impact decisions solely on benchmark results. Domain evaluation and accountable human review are separate requirements.

## 10. Release checklist

- [ ] Pre-registration with hypotheses, primary outcomes, sample-size rationale, and analysis plan
- [ ] Exact prompts and functional role definitions
- [ ] Model snapshots and execution manifest
- [ ] Versioned benchmark and data card
- [ ] Raw structured outputs with sensitive fields removed
- [ ] Evaluation and statistics code
- [ ] Full results, uncertainty, failure analysis, and negative findings
- [ ] Cost and latency report
- [ ] Ethics, privacy, and threat-model review
- [ ] Independent replication or at least a held-out second-model-family replication

