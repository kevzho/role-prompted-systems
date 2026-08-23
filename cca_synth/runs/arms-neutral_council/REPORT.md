# CCA calibrated synthesizer — run `arms-neutral_council`

> These roles are functional AI processing constraints. They are not diagnostic models, simulations of lived experience, or claims that mental health conditions confer a single cognitive mechanism.

- config hash: `a905600db20304a22ab1a8876eb5c8b245131998d0e8d3d258aa92e22844c2ea`
- roles (functional labels): `counter_evidence_checker`, `consistency_checker`, `local_claim_checker`, `explore_then_prune`, `rare_hypothesis_checker`, `confidence_calibrator`
- role-blind synthesizer: `False`
- Shapley mode: `retrain`

## Training

- best `selective_accuracy` = 0.8451 at epoch 2
- optimiser steps: 99
- wall clock: 54.5s

## Outcomes

| arm | accuracy | selective acc. | coverage | ECE | Brier | AURC |
| --- | --- | --- | --- | --- | --- | --- |
| `cca_synthesizer` | 0.7989 | 0.7989 | 0.9296 | 0.1748 | 0.3615 | 0.0954 |
| `cca_synthesizer_no_selective_abstention` | 0.7989 | 0.7989 | 1.0000 | 0.1748 | 0.3615 | 0.0954 |
| `confidence_weighted_vote` | 0.8043 | 0.8043 | 1.0000 | 0.1835 | 0.3839 | 0.1039 |
| `direct_answer_base` | 0.8043 | 0.8043 | 1.0000 | 0.1612 | 0.3255 | 0.2885 |
| `majority_vote` | 0.7989 | 0.7989 | 1.0000 | 0.1881 | 0.3871 | 0.1559 |
| `oracle_upper_bound_not_deployable` | 0.8424 | 0.8424 | 1.0000 | 0.3633 | 0.4355 | 0.4434 |
| `single_agent::confidence_calibrator` | 0.8098 | 0.8098 | 1.0000 | 0.1583 | 0.3205 | 0.2792 |
| `single_agent::consistency_checker` | 0.8043 | 0.8043 | 1.0000 | 0.1420 | 0.3255 | 0.2941 |
| `single_agent::counter_evidence_checker` | 0.7935 | 0.7935 | 1.0000 | 0.1504 | 0.3355 | 0.2859 |
| `single_agent::explore_then_prune` | 0.7880 | 0.7880 | 1.0000 | 0.1449 | 0.3406 | 0.3065 |
| `single_agent::local_claim_checker` | 0.8043 | 0.8043 | 1.0000 | 0.1444 | 0.3255 | 0.2851 |
| `single_agent::rare_hypothesis_checker` | 0.7935 | 0.7935 | 1.0000 | 0.1420 | 0.3355 | 0.2965 |

The oracle arm consumes the gold label and is an upper bound on routing, never a deployable result.

### Paired item-level contrasts (synthesizer − comparator)

| comparator | Δ | 95% CI | P(Δ>0) | deployable |
| --- | --- | --- | --- | --- |
| `direct_answer_base` | -0.0054 | [-0.0435, +0.0326] | 0.325 | yes |
| `majority_vote` | +0.0000 | [-0.0272, +0.0272] | 0.416 | yes |
| `confidence_weighted_vote` | -0.0054 | [-0.0326, +0.0217] | 0.270 | yes |
| `oracle_upper_bound_not_deployable` | -0.0435 | [-0.0761, -0.0163] | 0.000 | no |
| `single_agent::counter_evidence_checker` | +0.0054 | [-0.0163, +0.0326] | 0.560 | yes |
| `single_agent::consistency_checker` | -0.0054 | [-0.0380, +0.0272] | 0.317 | yes |
| `single_agent::local_claim_checker` | -0.0054 | [-0.0380, +0.0272] | 0.307 | yes |
| `single_agent::explore_then_prune` | +0.0109 | [-0.0163, +0.0380] | 0.715 | yes |
| `single_agent::rare_hypothesis_checker` | +0.0054 | [-0.0163, +0.0326] | 0.570 | yes |
| `single_agent::confidence_calibrator` | -0.0109 | [-0.0380, +0.0217] | 0.185 | yes |

### Agent independence (limitation L7)

- mean pairwise error correlation: 0.8852 (max 0.9837)
- High off-diagonal correlation means the council supplies less independent evidence than its agent count implies (limitation L7). It does not by itself invalidate a measured gain, but it bounds how much of that gain can be attributed to diversity.

### Robustness (specification section 9)

- agent-permutation invariance: max |Δlogit| = 1.34e-07 (tolerance 1e-04) — **PASS**
- option-permutation equivariance: 0.9969 agreement
- `prompt_injection_via_agent_output`: accuracy 0.7826 (degradation +0.0163)
- `confidently_wrong_majority`: accuracy 0.6685 (degradation +0.1304)
- `evidence_free_agreement`: accuracy 0.0000 (degradation +0.7989)
- `role_label_permutation`: accuracy 0.8043 (degradation -0.0054)

## Exclusions, failed calls, and retries (specification section 8)

1500 items × 6 roles = 9000 expected calls. 0 source lines were unparsable. 167 calls (1.86%) produced no usable record and were excluded rather than imputed.

| role | missing outputs | retries | abstentions |
| --- | --- | --- | --- |
| `counter_evidence_checker` | 29 | 0 | 65 |
| `consistency_checker` | 32 | 0 | 60 |
| `local_claim_checker` | 30 | 0 | 60 |
| `explore_then_prune` | 26 | 0 | 63 |
| `rare_hypothesis_checker` | 28 | 0 | 53 |
| `confidence_calibrator` | 22 | 0 | 68 |

Failure reasons for the missing outputs: `format_invalid` 2, `refusal` 165.

- retries: 0 total, 0.000 per item, across 0 calls

### Cost (limitations L10, L18)

- council calls per item: 6
- output tokens per item: 1428.2
- USD per item: 0.023883
- Budget-matched comparison requires holding these quantities equal across arms; the unconstrained numbers above are the quality-cost frontier point, not a matched comparison.

## Leave-one-out attribution (PREREGISTRATION.md A2.2)

- estimator: leave-one-out over 7 coalitions, mode `retrain` · v(A) = 0.8043
- decision rule: interval excludes zero AND contrast ≥ 0.050 (registered as `eval.loo_moves_threshold`)
- degeneracy check: 6 distinct coalition values — **PASS**

| role | v(A) − v(A∖{i}) | 95% CI | P(>0) | moves benchmark | predeclared primary |
| --- | --- | --- | --- | --- | --- |
| `counter_evidence_checker` | +0.0163 | [-0.0056, +0.0395] | 0.884 | **no** | `truthfulqa` |
| `consistency_checker` | +0.0109 | [-0.0157, +0.0370] | 0.736 | **no** | `math_aime` |
| `local_claim_checker` | +0.0054 | [+0.0000, +0.0169] | 0.645 | **no** | `advconsistency` |
| `explore_then_prune` | -0.0109 | [-0.0279, +0.0000] | 0.000 | **no** | `creative_divergence` |
| `rare_hypothesis_checker` | -0.0163 | [-0.0383, +0.0000] | 0.000 | **no** | `temporal_qa` |
| `confidence_calibrator` | +0.0000 | [+0.0000, +0.0000] | 0.000 | **no** | `sycophancy` |

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
| `local_claim_checker` | `halueval` | no | +0.0054 | nan | no |
| `explore_then_prune` | `truthfulqa` | no | +nan | nan | no |
| `explore_then_prune` | `halueval` | no | -0.0109 | nan | no |
| `rare_hypothesis_checker` | `truthfulqa` | no | +nan | nan | no |
| `rare_hypothesis_checker` | `halueval` | no | -0.0163 | nan | no |
| `confidence_calibrator` | `truthfulqa` | no | +nan | nan | no |
| `confidence_calibrator` | `halueval` | no | +0.0000 | nan | no |

## Claim language

Per `LIMITATIONS.md`, results from this repository support statements of the form *we propose*, *we hypothesize*, and *this operation is inspired by*. They do not support *we demonstrate*, *CCA reduces*, or *CCA is the first*. Clinical identifiers appear nowhere in the code, the tensors, or the audit trail; the traceability map (`counter_evidence_checker`, `consistency_checker`, `local_claim_checker`, `explore_then_prune`, `rare_hypothesis_checker`, `confidence_calibrator`) exists solely so readers of the original concept paper can locate the corresponding section.
