# Limitations and dispositions

Reviewed against `disorders_ml_ai.pdf` on 2026-07-26. "Fixed" below means that a companion document supplies a bounded protocol, correction, or safety requirement. It does not mean that missing experiments have been run.

## High-priority limitations

| ID | Limitation | Why it matters | Disposition |
| --- | --- | --- | --- |
| L1 | No empirical results are reported. | The abstract says "We evaluate," but the paper contains only an evaluation plan. There are no scores, confidence intervals, statistical tests, examples, error analysis, or costs. | `RESEARCH_SPECIFICATION.md` defines the minimum experiment. Until it is run, rewrite the abstract as "We propose to evaluate" and label every performance statement as a hypothesis. |
| L2 | Novelty claims are not supported by a systematic search. | Statements such as "CCA is the first" and "categorical" inversion can be invalidated by adjacent work. Several cited comparators appeared in 2025-2026 and the bibliography is incomplete. | `REFERENCE_CORRECTIONS.md` verifies known adjacent works. Replace "first" with "to our knowledge" and publish the search databases, queries, date, inclusion rules, and screened results. |
| L3 | The IPA formalism is descriptive rather than operational. | The function `f(C_i, y, x)` is unconstrained and unmeasured, so nearly any prompt behavior could be described after the fact as an IPA. | `RESEARCH_SPECIFICATION.md` requires a predeclared transformation, measurable behavior check, and falsifiable success criterion for each agent. |
| L4 | The subtitle conflicts with the proposed implementation. | The PDF says "Prompt-Level Implementation" but also relies on decoding controls, context manipulation, and "early-layer self-attention temperature," which hosted model APIs generally do not expose. | The supplement separates portable prompt-level V1 from optional open-weight V2. Claims about hidden-layer interventions are out of scope until code and model access exist. |
| L5 | Clinical analogies are broader than their sources. | OCD, bipolar disorder, schizophrenia, autism/weak central coherence, paranoia, and impostor phenomenon are heterogeneous human experiences, not single algorithms. Some cited works motivate a construct but do not validate the proposed LLM operation. | `ETHICS_AND_DATA_GOVERNANCE.md` requires functional public names and prohibits diagnostic or simulation claims. `REFERENCE_CORRECTIONS.md` distinguishes inspiration from evidentiary support. |
| L6 | The paper proposes logging chain-of-thought. | Hidden reasoning may be unavailable, unreliable as an explanation, privacy-sensitive, or disallowed by a model provider. It can also expose prompts, personal data, or security-relevant details. | Replace chain-of-thought with the structured evidence and decision record in `ETHICS_AND_DATA_GOVERNANCE.md`. Never require private reasoning tokens. |
| L7 | Agent diversity may be superficial or correlated. | Six instances of one base model can share training priors and failure modes. Different role prompts do not establish independent evidence. | Measure pairwise error correlation, output similarity, disagreement, and marginal utility. Include heterogeneous-model and homogeneous-model conditions. |
| L8 | The synthesizer is a major uncontrolled confound. | A strong synthesizer may create the gain; a weak or biased one may erase it. Self-reported confidence is usually not calibrated merely because it is called calibrated. | Factor synthesis method independently, compare with simple voting and best-single-agent controls, and calibrate confidence on a separate validation set. |
| L9 | Benchmark and metric definitions are incomplete. | "Creative divergence," "sycophancy battery," and the proposed AdvConsistency benchmark lack item construction, rubrics, annotator procedures, licenses, and primary metrics. | The supplement predeclares task-specific outcomes, blinded judging, leakage checks, and an AdvConsistency release checklist. |
| L10 | Fairness under compute is not guaranteed. | A six-agent council plus synthesis uses more calls and tokens than a single agent. Matching only the number of samples does not match context, output, latency, or cost. | Report both unconstrained performance and budget-matched performance using equal input tokens, output tokens, wall-clock limits, and dollar cost. |

## Methodological limitations

| ID | Limitation | Required correction |
| --- | --- | --- |
| L11 | The full `2^6 = 64` subset sweep includes the empty set, whose value is undefined unless a null policy is specified. A subset sweep alone is not a statistical analysis. | Define the empty-set baseline, repeat each item across fixed seeds, compute item-level Shapley values, and report bootstrap intervals. Do not call an agent "necessary" or "sufficient" from a point estimate alone. |
| L12 | Agent roles are linked one-to-one with benchmarks, inviting confirmation bias. | Pre-register all agent-by-benchmark predictions, including expected harms. Score every agent and subset on every benchmark. Correct for multiple comparisons. |
| L13 | "Primary target" assignments omit plausible trade-offs. | Include long-context, instruction-following, safety, calibration, latency, and over-refusal outcomes. An agent passes only if its target benefit is not purchased with unacceptable regressions. |
| L14 | No prompt, model, or decoding versions are frozen. | Record provider, exact model ID and snapshot, system/user prompts, parameters, context assembly, random seed where supported, date, and retry policy. |
| L15 | No contamination or temporal-validity plan is given. | Keep private held-out items, use post-training-cutoff items where relevant, hash release splits, and distinguish closed-book knowledge from retrieval-assisted performance. |
| L16 | LLM judging and human judging are underspecified. | Blind judges to condition, randomize order, use at least two independent human ratings for subjective primary outcomes, report agreement, and audit a stratified sample of model-judge decisions. |
| L17 | Safety failures from adversarial roles are not evaluated. | Test prompt injection, false minority hypotheses, fabricated counter-evidence, consensus capture, harmful-content escalation, and leakage between agents. |
| L18 | Cost, latency, and environmental burden are absent. | Report calls, input/output tokens, p50/p95 latency, retries, provider/model prices at run time, and estimated compute or energy when measurable. |
| L19 | The work has no reproducibility artifact. | Release prompts, orchestration code, environment lockfile, raw per-item outputs subject to privacy rules, evaluation scripts, seeds, and an executable analysis notebook. |

## Conceptual and presentation limitations

- "Mental disorder vs. emotion" is an overly sharp dichotomy. Psychiatric conditions can involve affect, cognition, behavior, physiology, environment, and social context. Reframe the distinction as functional process constraints versus output personas.
- The paper shifts between disorder, syndrome, profile, construct, and computational mechanism. Those terms are not interchangeable.
- "Savant syndrome / weak central coherence" combines distinct concepts and relies on an autism monograph as if it directly established an LLM mechanism.
- The bipolar citation concerns positive emotion and reward disturbance; it does not establish a two-pass high-temperature/low-temperature algorithm.
- The Dunning-Kruger citation does not support impostor phenomenon.
- The ADHD citation is misattributed and the proposed six-agent council contains no ADHD-derived agent.
- The title and repeated clinical mappings may imply psychiatric endorsement or faithful simulation. The supplement recommends a function-first title and role vocabulary.
- The placeholder `[email]`, incomplete `[TBD]` citations, split contents page, sparse final page, and absence of appendices make the PDF unsuitable as a submission-ready manuscript.
- Venue targets and claims about "recent" work are time-sensitive. Verify them at each revision rather than treating this July 2026 review as permanently current.

## Claim language until evidence exists

Use:

- "We propose..."
- "We hypothesize..."
- "This operation is inspired by..."
- "The clinical literature does not establish that the LLM operation reproduces the human condition."

Avoid:

- "We demonstrate..."
- "CCA reduces..."
- "empirically validating each IPA..."
- "CCA is the first..."
- any suggestion that an agent has, experiences, or simulates a psychiatric condition

