# CCA calibrated synthesizer — run `pilot-halueval`

> These roles are functional AI processing constraints. They are not diagnostic models, simulations of lived experience, or claims that mental health conditions confer a single cognitive mechanism.

- config hash: `97da97b6a0339949c0a1ba614616dd1ca325a0aeb9d706440c78440d9a686a61`
- roles (functional labels): `counter_evidence_checker`, `consistency_checker`, `local_claim_checker`, `explore_then_prune`, `rare_hypothesis_checker`, `confidence_calibrator`
- role-blind synthesizer: `False`
- Shapley mode: `retrain`

## Training

- best `selective_accuracy` = 0.8966 at epoch 1
- optimiser steps: 20
- wall clock: 29.4s

## Outcomes

| arm | accuracy | selective acc. | coverage | ECE | Brier | AURC |
| --- | --- | --- | --- | --- | --- | --- |
| `cca_synthesizer` | 0.7234 | 0.6383 | 0.7872 | 0.2647 | 0.4291 | 0.1305 |
| `cca_synthesizer_no_selective_abstention` | 0.7234 | 0.7234 | 1.0000 | 0.2647 | 0.4291 | 0.1305 |
| `confidence_weighted_vote` | 0.7447 | 0.7447 | 1.0000 | 0.2360 | 0.4625 | 0.1462 |
| `direct_answer_base` | 0.3404 | 0.3404 | 1.0000 | 0.2872 | 0.5000 | 0.7113 |
| `majority_vote` | 0.7447 | 0.7447 | 1.0000 | 0.2354 | 0.4644 | 0.2022 |
| `oracle_upper_bound_not_deployable` | 0.8511 | 0.8511 | 1.0000 | 0.3546 | 0.4195 | 0.4115 |
| `single_agent::confidence_calibrator` | 0.7447 | 0.7447 | 1.0000 | 0.2153 | 0.3806 | 0.3637 |
| `single_agent::consistency_checker` | 0.7872 | 0.7872 | 1.0000 | 0.2070 | 0.3413 | 0.3243 |
| `single_agent::counter_evidence_checker` | 0.7447 | 0.7447 | 1.0000 | 0.2070 | 0.3806 | 0.3569 |
| `single_agent::explore_then_prune` | 0.7660 | 0.7660 | 1.0000 | 0.2283 | 0.3610 | 0.3282 |
| `single_agent::local_claim_checker` | 0.7447 | 0.7447 | 1.0000 | 0.2070 | 0.3806 | 0.3529 |
| `single_agent::rare_hypothesis_checker` | 0.7447 | 0.7447 | 1.0000 | 0.2496 | 0.3806 | 0.3645 |

The oracle arm consumes the gold label and is an upper bound on routing, never a deployable result.

### Paired item-level contrasts (synthesizer − comparator)

| comparator | Δ | 95% CI | P(Δ>0) | deployable |
| --- | --- | --- | --- | --- |
| `direct_answer_base` | +0.2979 | [+0.1064, +0.4894] | 0.998 | yes |
| `majority_vote` | -0.1064 | [-0.1915, -0.0213] | 0.000 | yes |
| `confidence_weighted_vote` | -0.1064 | [-0.1915, -0.0213] | 0.000 | yes |
| `oracle_upper_bound_not_deployable` | -0.2128 | [-0.3404, -0.1064] | 0.000 | no |
| `single_agent::counter_evidence_checker` | -0.1064 | [-0.1915, -0.0213] | 0.000 | yes |
| `single_agent::consistency_checker` | -0.1489 | [-0.2553, -0.0638] | 0.000 | yes |
| `single_agent::local_claim_checker` | -0.1064 | [-0.2128, -0.0213] | 0.000 | yes |
| `single_agent::explore_then_prune` | -0.1277 | [-0.2340, -0.0426] | 0.000 | yes |
| `single_agent::rare_hypothesis_checker` | -0.1064 | [-0.1915, -0.0213] | 0.000 | yes |
| `single_agent::confidence_calibrator` | -0.1064 | [-0.1915, -0.0213] | 0.000 | yes |

### Agent independence (limitation L7)

- mean pairwise error correlation: 0.7996 (max 0.9440)
- High off-diagonal correlation means the council supplies less independent evidence than its agent count implies (limitation L7). It does not by itself invalidate a measured gain, but it bounds how much of that gain can be attributed to diversity.

### Robustness (specification section 9)

- agent-permutation invariance: max |Δlogit| = 1.01e-07 (tolerance 1e-04) — **PASS**
- option-permutation equivariance: 1.0000 agreement
- `prompt_injection_via_agent_output`: accuracy 0.6170 (degradation +0.1064)
- `confidently_wrong_majority`: accuracy 0.3617 (degradation +0.3617)
- `evidence_free_agreement`: accuracy 0.0000 (degradation +0.7234)
- `role_label_permutation`: accuracy 0.7234 (degradation +0.0000)

## Exclusions, failed calls, and retries (specification section 8)

200 items × 6 roles = 1200 expected calls. 0 source lines were unparsable. 0 calls (0.00%) produced no usable record and were excluded rather than imputed.

| role | missing outputs | retries | abstentions |
| --- | --- | --- | --- |
| `counter_evidence_checker` | 0 | 0 | 0 |
| `consistency_checker` | 0 | 0 | 0 |
| `local_claim_checker` | 0 | 0 | 0 |
| `explore_then_prune` | 0 | 0 | 0 |
| `rare_hypothesis_checker` | 0 | 0 | 0 |
| `confidence_calibrator` | 0 | 0 | 0 |

- retries: 0 total, 0.000 per item, across 0 calls

### Cost (limitations L10, L18)

- council calls per item: 6
- output tokens per item: 1304.6
- USD per item: 0.023000
- Budget-matched comparison requires holding these quantities equal across arms; the unconstrained numbers above are the quality-cost frontier point, not a matched comparison.

## Leave-one-out attribution (PREREGISTRATION.md A2.2)

- estimator: leave-one-out over 7 coalitions, mode `retrain` · v(A) = 0.6383
- decision rule: interval excludes zero AND contrast ≥ 0.050 (registered as `eval.loo_moves_threshold`)
- degeneracy check: 5 distinct coalition values — **PASS**

| role | v(A) − v(A∖{i}) | 95% CI | P(>0) | moves benchmark | predeclared primary |
| --- | --- | --- | --- | --- | --- |
| `counter_evidence_checker` | +0.0000 | [+0.0000, +0.0000] | 0.000 | **no** | `truthfulqa` |
| `consistency_checker` | +0.0213 | [+0.0000, +0.0638] | 0.647 | **no** | `math_aime` |
| `local_claim_checker` | -0.0426 | [-0.1064, +0.0000] | 0.000 | **no** | `advconsistency` |
| `explore_then_prune` | +0.0426 | [+0.0000, +0.1064] | 0.875 | **no** | `creative_divergence` |
| `rare_hypothesis_checker` | +0.0213 | [+0.0000, +0.0638] | 0.647 | **no** | `temporal_qa` |
| `confidence_calibrator` | -0.0213 | [-0.0638, +0.0000] | 0.000 | **no** | `sycophancy` |

> Leave-one-out is not a Shapley value. It does not satisfy efficiency, so these numbers need not sum to v(A) - v(empty), and no efficiency check is reported because none applies.
> Every interaction between role i and the rest of the council is charged to role i by this estimator. Under the substitutable pairs recorded in PREREGISTRATION.md A1, that biases each contrast toward zero, so a role clearing the threshold has cleared a conservative bar.
> Coalitions other than the full council and the six drop-one sets were not evaluated. No claim about any other coalition is supported by this table.
> Each coalition was trained separately; no masking approximation enters.

### Repeated trials (section 8)

- schedule hash: `bb1d106d48c982f9` (3 trials per item, 141 observations per coalition)
- coverage: **PASS** — 7/7 coalitions reported
- This schedule is generated once and reused for every coalition. Redrawing it per coalition would confound the coalition with the seed and invalidate the paired item-level statistics.

### Agent × benchmark, FDR-controlled (limitation L12)

0 of 6 cells survive Benjamini–Hochberg at α=0.05.

| role | benchmark | primary? | marginal value | q | significant |
| --- | --- | --- | --- | --- | --- |
| `counter_evidence_checker` | `halueval` | no | +0.0000 | 1.000 | no |
| `consistency_checker` | `halueval` | no | +0.0213 | 0.761 | no |
| `local_claim_checker` | `halueval` | no | -0.0426 | 0.761 | no |
| `explore_then_prune` | `halueval` | no | +0.0426 | 0.761 | no |
| `rare_hypothesis_checker` | `halueval` | no | +0.0213 | 0.761 | no |
| `confidence_calibrator` | `halueval` | no | -0.0213 | 0.761 | no |

## Claim language

Per `LIMITATIONS.md`, results from this repository support statements of the form *we propose*, *we hypothesize*, and *this operation is inspired by*. They do not support *we demonstrate*, *CCA reduces*, or *CCA is the first*. Clinical identifiers appear nowhere in the code, the tensors, or the audit trail; the traceability map (`counter_evidence_checker`, `consistency_checker`, `local_claim_checker`, `explore_then_prune`, `rare_hypothesis_checker`, `confidence_calibrator`) exists solely so readers of the original concept paper can locate the corresponding section.
