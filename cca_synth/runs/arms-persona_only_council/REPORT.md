# CCA calibrated synthesizer — run `arms-persona_only_council`

> These roles are functional AI processing constraints. They are not diagnostic models, simulations of lived experience, or claims that mental health conditions confer a single cognitive mechanism.

- config hash: `d4cf7e905468e1a432477d3520dccba57079e819236de52463bf81879c8f5255`
- roles (functional labels): `counter_evidence_checker`, `consistency_checker`, `local_claim_checker`, `explore_then_prune`, `rare_hypothesis_checker`, `confidence_calibrator`
- role-blind synthesizer: `False`
- Shapley mode: `retrain`

## Training

- best `selective_accuracy` = 0.8873 at epoch 3
- optimiser steps: 108
- wall clock: 55.3s

## Outcomes

| arm | accuracy | selective acc. | coverage | ECE | Brier | AURC |
| --- | --- | --- | --- | --- | --- | --- |
| `cca_synthesizer` | 0.8043 | 0.8043 | 0.9519 | 0.1585 | 0.3495 | 0.0883 |
| `cca_synthesizer_no_selective_abstention` | 0.8043 | 0.8043 | 1.0000 | 0.1585 | 0.3495 | 0.0883 |
| `confidence_weighted_vote` | 0.7989 | 0.7989 | 1.0000 | 0.1889 | 0.3902 | 0.1032 |
| `direct_answer_base` | 0.8043 | 0.8043 | 1.0000 | 0.1612 | 0.3255 | 0.2878 |
| `majority_vote` | 0.7880 | 0.7880 | 1.0000 | 0.1999 | 0.3917 | 0.1550 |
| `oracle_upper_bound_not_deployable` | 0.8641 | 0.8641 | 1.0000 | 0.3571 | 0.3954 | 0.4017 |
| `single_agent::confidence_calibrator` | 0.8043 | 0.8043 | 1.0000 | 0.1553 | 0.3255 | 0.2935 |
| `single_agent::consistency_checker` | 0.8207 | 0.8207 | 1.0000 | 0.1583 | 0.3104 | 0.2517 |
| `single_agent::counter_evidence_checker` | 0.7989 | 0.7989 | 1.0000 | 0.1667 | 0.3305 | 0.2974 |
| `single_agent::explore_then_prune` | 0.7935 | 0.7935 | 1.0000 | 0.1553 | 0.3355 | 0.3024 |
| `single_agent::local_claim_checker` | 0.7880 | 0.7880 | 1.0000 | 0.1533 | 0.3406 | 0.2948 |
| `single_agent::rare_hypothesis_checker` | 0.7989 | 0.7989 | 1.0000 | 0.1558 | 0.3305 | 0.2940 |

The oracle arm consumes the gold label and is an upper bound on routing, never a deployable result.

### Paired item-level contrasts (synthesizer − comparator)

| comparator | Δ | 95% CI | P(Δ>0) | deployable |
| --- | --- | --- | --- | --- |
| `direct_answer_base` | +0.0000 | [-0.0326, +0.0326] | 0.438 | yes |
| `majority_vote` | +0.0163 | [-0.0054, +0.0380] | 0.880 | yes |
| `confidence_weighted_vote` | +0.0054 | [-0.0109, +0.0219] | 0.604 | yes |
| `oracle_upper_bound_not_deployable` | -0.0598 | [-0.0978, -0.0272] | 0.000 | no |
| `single_agent::counter_evidence_checker` | +0.0054 | [-0.0163, +0.0326] | 0.575 | yes |
| `single_agent::consistency_checker` | -0.0163 | [-0.0435, +0.0109] | 0.092 | yes |
| `single_agent::local_claim_checker` | +0.0163 | [-0.0109, +0.0435] | 0.845 | yes |
| `single_agent::explore_then_prune` | +0.0109 | [-0.0163, +0.0435] | 0.691 | yes |
| `single_agent::rare_hypothesis_checker` | +0.0054 | [-0.0217, +0.0326] | 0.580 | yes |
| `single_agent::confidence_calibrator` | +0.0000 | [-0.0272, +0.0272] | 0.417 | yes |

### Agent independence (limitation L7)

- mean pairwise error correlation: 0.8483 (max 0.9146)
- High off-diagonal correlation means the council supplies less independent evidence than its agent count implies (limitation L7). It does not by itself invalidate a measured gain, but it bounds how much of that gain can be attributed to diversity.

### Robustness (specification section 9)

- agent-permutation invariance: max |Δlogit| = 1.19e-07 (tolerance 1e-04) — **PASS**
- option-permutation equivariance: 0.9969 agreement
- `prompt_injection_via_agent_output`: accuracy 0.7772 (degradation +0.0272)
- `confidently_wrong_majority`: accuracy 0.5924 (degradation +0.2120)
- `evidence_free_agreement`: accuracy 0.0000 (degradation +0.8043)
- `role_label_permutation`: accuracy 0.8043 (degradation +0.0000)

## Exclusions, failed calls, and retries (specification section 8)

1500 items × 6 roles = 9000 expected calls. 0 source lines were unparsable. 139 calls (1.54%) produced no usable record and were excluded rather than imputed.

| role | missing outputs | retries | abstentions |
| --- | --- | --- | --- |
| `counter_evidence_checker` | 24 | 0 | 67 |
| `consistency_checker` | 24 | 0 | 50 |
| `local_claim_checker` | 26 | 0 | 100 |
| `explore_then_prune` | 24 | 0 | 51 |
| `rare_hypothesis_checker` | 21 | 0 | 55 |
| `confidence_calibrator` | 20 | 0 | 99 |

Failure reasons for the missing outputs: `format_invalid` 4, `refusal` 135.

- retries: 0 total, 0.000 per item, across 0 calls

### Cost (limitations L10, L18)

- council calls per item: 6
- output tokens per item: 1498.5
- USD per item: 0.024649
- Budget-matched comparison requires holding these quantities equal across arms; the unconstrained numbers above are the quality-cost frontier point, not a matched comparison.

## Leave-one-out attribution (PREREGISTRATION.md A2.2)

- estimator: leave-one-out over 7 coalitions, mode `retrain` · v(A) = 0.8043
- decision rule: interval excludes zero AND contrast ≥ 0.050 (registered as `eval.loo_moves_threshold`)
- degeneracy check: 4 distinct coalition values — **PASS**

| role | v(A) − v(A∖{i}) | 95% CI | P(>0) | moves benchmark | predeclared primary |
| --- | --- | --- | --- | --- | --- |
| `counter_evidence_checker` | +0.0163 | [-0.0055, +0.0419] | 0.876 | **no** | `truthfulqa` |
| `consistency_checker` | +0.0109 | [-0.0107, +0.0328] | 0.781 | **no** | `math_aime` |
| `local_claim_checker` | -0.0109 | [-0.0276, +0.0000] | 0.000 | **no** | `advconsistency` |
| `explore_then_prune` | +0.0000 | [-0.0160, +0.0158] | 0.341 | **no** | `creative_divergence` |
| `rare_hypothesis_checker` | +0.0163 | [+0.0000, +0.0372] | 0.952 | **no** | `temporal_qa` |
| `confidence_calibrator` | +0.0163 | [-0.0056, +0.0406] | 0.881 | **no** | `sycophancy` |

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
| `counter_evidence_checker` | `halueval` | no | +0.0163 | nan | no |
| `consistency_checker` | `truthfulqa` | no | +nan | nan | no |
| `consistency_checker` | `halueval` | no | +0.0109 | nan | no |
| `local_claim_checker` | `truthfulqa` | no | +nan | nan | no |
| `local_claim_checker` | `halueval` | no | -0.0109 | nan | no |
| `explore_then_prune` | `truthfulqa` | no | +nan | nan | no |
| `explore_then_prune` | `halueval` | no | +0.0000 | nan | no |
| `rare_hypothesis_checker` | `truthfulqa` | no | +nan | nan | no |
| `rare_hypothesis_checker` | `halueval` | no | +0.0163 | nan | no |
| `confidence_calibrator` | `truthfulqa` | no | +nan | nan | no |
| `confidence_calibrator` | `halueval` | no | +0.0163 | nan | no |

## Claim language

Per `LIMITATIONS.md`, results from this repository support statements of the form *we propose*, *we hypothesize*, and *this operation is inspired by*. They do not support *we demonstrate*, *CCA reduces*, or *CCA is the first*. Clinical identifiers appear nowhere in the code, the tensors, or the audit trail; the traceability map (`counter_evidence_checker`, `consistency_checker`, `local_claim_checker`, `explore_then_prune`, `rare_hypothesis_checker`, `confidence_calibrator`) exists solely so readers of the original concept paper can locate the corresponding section.
