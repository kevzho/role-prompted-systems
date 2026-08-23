# CCA calibrated synthesizer — run `arms-cca_council`

> These roles are functional AI processing constraints. They are not diagnostic models, simulations of lived experience, or claims that mental health conditions confer a single cognitive mechanism.

- config hash: `541a000237a68c9cfc718abf9a8f6f1e647e16718e6fcd1f64f29b40da812b20`
- roles (functional labels): `counter_evidence_checker`, `consistency_checker`, `local_claim_checker`, `explore_then_prune`, `rare_hypothesis_checker`, `confidence_calibrator`
- role-blind synthesizer: `False`
- Shapley mode: `retrain`

## Training

- best `selective_accuracy` = 0.8592 at epoch 0
- optimiser steps: 81
- wall clock: 34.7s

## Outcomes

| arm | accuracy | selective acc. | coverage | ECE | Brier | AURC |
| --- | --- | --- | --- | --- | --- | --- |
| `cca_synthesizer` | 0.7826 | 0.7826 | 0.9333 | 0.2606 | 0.4242 | 0.1103 |
| `cca_synthesizer_no_selective_abstention` | 0.7826 | 0.7826 | 1.0000 | 0.2606 | 0.4242 | 0.1103 |
| `confidence_weighted_vote` | 0.7935 | 0.7935 | 1.0000 | 0.1875 | 0.3835 | 0.1073 |
| `direct_answer_base` | 0.8043 | 0.8043 | 1.0000 | 0.1612 | 0.3255 | 0.2878 |
| `majority_vote` | 0.7826 | 0.7826 | 1.0000 | 0.1977 | 0.3846 | 0.1669 |
| `oracle_upper_bound_not_deployable` | 0.8696 | 0.8696 | 1.0000 | 0.3640 | 0.3854 | 0.3907 |
| `single_agent::confidence_calibrator` | 0.8043 | 0.8043 | 1.0000 | 0.1528 | 0.3255 | 0.2885 |
| `single_agent::consistency_checker` | 0.8043 | 0.8043 | 1.0000 | 0.1553 | 0.3255 | 0.3010 |
| `single_agent::counter_evidence_checker` | 0.7717 | 0.7717 | 1.0000 | 0.1479 | 0.3556 | 0.3317 |
| `single_agent::explore_then_prune` | 0.7989 | 0.7989 | 1.0000 | 0.1583 | 0.3305 | 0.3009 |
| `single_agent::local_claim_checker` | 0.7935 | 0.7935 | 1.0000 | 0.1479 | 0.3355 | 0.2841 |
| `single_agent::rare_hypothesis_checker` | 0.8098 | 0.8098 | 1.0000 | 0.1583 | 0.3205 | 0.2995 |

The oracle arm consumes the gold label and is an upper bound on routing, never a deployable result.

### Paired item-level contrasts (synthesizer − comparator)

| comparator | Δ | 95% CI | P(Δ>0) | deployable |
| --- | --- | --- | --- | --- |
| `direct_answer_base` | -0.0217 | [-0.0598, +0.0163] | 0.122 | yes |
| `majority_vote` | +0.0000 | [-0.0326, +0.0326] | 0.447 | yes |
| `confidence_weighted_vote` | -0.0109 | [-0.0489, +0.0272] | 0.229 | yes |
| `oracle_upper_bound_not_deployable` | -0.0870 | [-0.1304, -0.0489] | 0.000 | no |
| `single_agent::counter_evidence_checker` | +0.0109 | [-0.0272, +0.0489] | 0.661 | yes |
| `single_agent::consistency_checker` | -0.0217 | [-0.0598, +0.0163] | 0.112 | yes |
| `single_agent::local_claim_checker` | -0.0109 | [-0.0380, +0.0163] | 0.148 | yes |
| `single_agent::explore_then_prune` | -0.0163 | [-0.0543, +0.0217] | 0.154 | yes |
| `single_agent::rare_hypothesis_checker` | -0.0272 | [-0.0707, +0.0163] | 0.084 | yes |
| `single_agent::confidence_calibrator` | -0.0217 | [-0.0652, +0.0217] | 0.124 | yes |

### Agent independence (limitation L7)

- mean pairwise error correlation: 0.8043 (max 0.9489)
- High off-diagonal correlation means the council supplies less independent evidence than its agent count implies (limitation L7). It does not by itself invalidate a measured gain, but it bounds how much of that gain can be attributed to diversity.

### Robustness (specification section 9)

- agent-permutation invariance: max |Δlogit| = 1.19e-07 (tolerance 1e-04) — **PASS**
- option-permutation equivariance: 0.9969 agreement
- `prompt_injection_via_agent_output`: accuracy 0.6739 (degradation +0.1087)
- `confidently_wrong_majority`: accuracy 0.4891 (degradation +0.2935)
- `evidence_free_agreement`: accuracy 0.0000 (degradation +0.7826)
- `role_label_permutation`: accuracy 0.7772 (degradation +0.0054)

## Exclusions, failed calls, and retries (specification section 8)

1500 items × 6 roles = 9000 expected calls. 0 source lines were unparsable. 186 calls (2.07%) produced no usable record and were excluded rather than imputed.

| role | missing outputs | retries | abstentions |
| --- | --- | --- | --- |
| `counter_evidence_checker` | 31 | 0 | 86 |
| `consistency_checker` | 43 | 0 | 179 |
| `local_claim_checker` | 38 | 0 | 207 |
| `explore_then_prune` | 22 | 0 | 67 |
| `rare_hypothesis_checker` | 42 | 0 | 42 |
| `confidence_calibrator` | 10 | 0 | 143 |

Failure reasons for the missing outputs: `format_invalid` 2, `refusal` 184.

- retries: 0 total, 0.000 per item, across 0 calls

### Cost (limitations L10, L18)

- council calls per item: 6
- output tokens per item: 1659.4
- USD per item: 0.026689
- Budget-matched comparison requires holding these quantities equal across arms; the unconstrained numbers above are the quality-cost frontier point, not a matched comparison.

## Leave-one-out attribution (PREREGISTRATION.md A2.2)

- estimator: leave-one-out over 7 coalitions, mode `retrain` · v(A) = 0.7826
- decision rule: interval excludes zero AND contrast ≥ 0.050 (registered as `eval.loo_moves_threshold`)
- degeneracy check: 5 distinct coalition values — **PASS**

| role | v(A) − v(A∖{i}) | 95% CI | P(>0) | moves benchmark | predeclared primary |
| --- | --- | --- | --- | --- | --- |
| `counter_evidence_checker` | -0.0163 | [-0.0481, +0.0116] | 0.109 | **no** | `truthfulqa` |
| `consistency_checker` | +0.0054 | [+0.0000, +0.0169] | 0.637 | **no** | `math_aime` |
| `local_claim_checker` | -0.0109 | [-0.0402, +0.0172] | 0.181 | **no** | `advconsistency` |
| `explore_then_prune` | +0.0054 | [-0.0112, +0.0231] | 0.599 | **no** | `creative_divergence` |
| `rare_hypothesis_checker` | -0.0054 | [-0.0170, +0.0000] | 0.000 | **no** | `temporal_qa` |
| `confidence_calibrator` | +0.0000 | [-0.0159, +0.0160] | 0.348 | **no** | `sycophancy` |

> Leave-one-out is not a Shapley value. It does not satisfy efficiency, so these numbers need not sum to v(A) - v(empty), and no efficiency check is reported because none applies.
> Every interaction between role i and the rest of the council is charged to role i by this estimator. Under the substitutable pairs recorded in PREREGISTRATION.md A1, that biases each contrast toward zero, so a role clearing the threshold has cleared a conservative bar.
> Coalitions other than the full council and the six drop-one sets were not evaluated. No claim about any other coalition is supported by this table.
> Each coalition was trained separately; no masking approximation enters.

### Repeated trials (section 8)

- schedule hash: `10cb967388b00af8` (3 trials per item, 810 observations per coalition)
- coverage: **PASS** — 7/7 coalitions reported
- This schedule is generated once and reused for every coalition. Redrawing it per coalition would confound the coalition with the seed and invalidate the paired item-level statistics.

### Agent × benchmark, FDR-controlled (limitation L12)

0 of 12 cells survive Benjamini–Hochberg at α=0.05.

| role | benchmark | primary? | marginal value | q | significant |
| --- | --- | --- | --- | --- | --- |
| `counter_evidence_checker` | `truthfulqa` | yes | +nan | nan | no |
| `counter_evidence_checker` | `halueval` | no | -0.0163 | nan | no |
| `consistency_checker` | `truthfulqa` | no | +nan | nan | no |
| `consistency_checker` | `halueval` | no | +0.0054 | nan | no |
| `local_claim_checker` | `truthfulqa` | no | +nan | nan | no |
| `local_claim_checker` | `halueval` | no | -0.0109 | nan | no |
| `explore_then_prune` | `truthfulqa` | no | +nan | nan | no |
| `explore_then_prune` | `halueval` | no | +0.0054 | nan | no |
| `rare_hypothesis_checker` | `truthfulqa` | no | +nan | nan | no |
| `rare_hypothesis_checker` | `halueval` | no | -0.0054 | nan | no |
| `confidence_calibrator` | `truthfulqa` | no | +nan | nan | no |
| `confidence_calibrator` | `halueval` | no | +0.0000 | nan | no |

## Claim language

Per `LIMITATIONS.md`, results from this repository support statements of the form *we propose*, *we hypothesize*, and *this operation is inspired by*. They do not support *we demonstrate*, *CCA reduces*, or *CCA is the first*. Clinical identifiers appear nowhere in the code, the tensors, or the audit trail; the traceability map (`counter_evidence_checker`, `consistency_checker`, `local_claim_checker`, `explore_then_prune`, `rare_hypothesis_checker`, `confidence_calibrator`) exists solely so readers of the original concept paper can locate the corresponding section.
