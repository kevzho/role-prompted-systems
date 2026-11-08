# Ethics and data governance addendum

## Position

CCA agents are software configurations. They do not have diagnoses, symptoms, lived experience, or a clinically validated equivalence to people. The psychiatric literature may inspire hypotheses about information processing, but it does not by itself validate a prompt, decoding rule, or model activation as a representation of a disorder.

## Terminology

Use function-first names in code, user interfaces, tables, and demonstrations:

| Original identifier | Preferred public label |
| --- | --- |
| Sentinel | Counter-evidence checker |
| Auditor | Consistency checker |
| Savant | Local-claim checker |
| Oscillator | Explore-then-prune |
| Dissenter | Rare-hypothesis checker |
| Challenger | Confidence calibrator |

In scholarly history sections, the original inspirations may be discussed with citations and with explicit limits. Avoid phrases such as "paranoid reasoner," "OCD agent," "schizophrenic agent," "manic phase," or "depressive phase" in prompts and product-facing copy. They collapse heterogeneous conditions into stereotypes and imply fidelity the implementation has not established.

Prefer "impostor phenomenon" over "impostor syndrome" unless quoting a source that uses the latter term. Do not equate impostor phenomenon with the Dunning-Kruger effect.

## Required disclaimer

Use this or equivalent language wherever the clinical inspiration is presented:

> These roles are functional AI processing constraints. They are not diagnostic models, simulations of lived experience, or claims that mental health conditions confer a single cognitive mechanism.

## Replace chain-of-thought logging

The PDF's "cognitive audit trail" must not store or expose private chain-of-thought. Collect a concise decision record instead:

- final candidate answer
- atomic factual claims
- evidence status for each claim
- citations or named verification checks
- counterevidence considered
- contradiction and safety flags
- declared confidence and abstention
- synthesizer acceptance/rejection decision and short rationale
- model/version, prompt version, timestamp, parameters, token counts, latency, and retry count

This record is an audit artifact, not a faithful window into the model's internal cognition. Explanations can be post-hoc and must be validated against interventions or evidence where faithfulness matters.

## Data minimization and retention

- Do not send personal or confidential data to six agents merely because one-agent processing was authorized.
- State which providers and regions receive data, whether providers retain it, and whether it can be used for training.
- Strip secrets, direct identifiers, and unnecessary free text before dispatch.
- Use per-field allowlists for logs. Do not log full system prompts, credentials, tool tokens, or hidden reasoning.
- Encrypt data in transit and at rest. Separate content data from experiment metadata.
- Use role-based access, access logs, and a named data owner.
- Set a short default retention period; document deletion and legal-hold exceptions.
- Obtain consent or another valid basis before using human conversation data for research.
- Provide a process to correct or delete records when applicable.
- Red-team membership inference, prompt leakage, cross-agent data propagation, and accidental release in research artifacts.

## Human-subject and clinical input

If experiments include people, private conversations, annotations about mental health, or recruitment based on diagnosis:

- obtain institutional ethics review or a documented determination that it is not required
- use informed consent and an age-appropriate process
- do not infer or assign diagnoses from model behavior
- avoid recruiting a stigmatized group merely to validate an analogy
- involve compensated people with relevant lived experience and clinical/research expertise in terminology review
- publish foreseeable discomfort, withdrawal, support, and debrief procedures

No mental-health advice or diagnosis should be generated or routed differently because a user's language appears to match one of the paper's analogies.

## Safety and accountability

- Maintain one accountable system owner; a council must not diffuse responsibility.
- Preserve the evidence behind dissenting outputs when the synthesizer rejects them.
- Measure false alarms, over-refusal, fabricated alternatives, and stubbornness, not only target-task gains.
- Require human review for consequential actions and give reviewers the evidence record, uncertainty, and unresolved disagreements.
- Provide a kill switch and a single-agent fallback when the council is degraded or under attack.
- Treat inter-agent text as untrusted input and apply the same tool and prompt-injection boundaries used for external content.
- Prohibit agents from diagnosing users, role-playing a disorder as entertainment, or presenting stereotypes as clinical facts.

## Publication reporting

Report:

- why each human clinical construct was selected
- which part of the source is used as inspiration
- what was deliberately not modeled
- expert and lived-experience review, if any
- negative results and regressions
- privacy incidents and safety failures
- demographic or language subgroup effects where the task involves people
- funding, conflicts, provider relationships, and model access

The ethical adequacy of the work depends on the actual prompts, data, deployment context, and results. This addendum is a minimum, not ethics approval.

