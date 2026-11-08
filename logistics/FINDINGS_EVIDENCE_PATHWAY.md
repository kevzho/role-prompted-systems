# Findings: the evidence pathway

Dated 2026-07-28. All numbers below come from **simulated** records and are
statements about the *architecture*, not about CCA. They are recorded here
because some are design constraints that should be pre-registered, and some are
defects that were found and fixed.

**§§6–8 were added on 2026-07-28** after executing code paths that had never
been run. Two of the three are defects that would have corrupted the registered
analysis — and neither crashed, failed a test, or produced obviously wrong
output. Both were certified by the pipeline's own checks:

| § | What | Severity |
| --- | --- | --- |
| 6 | All 15 agent pairs are substitutable, at the same magnitude as the main effects | A finding, and a constraint on H1 |
| 7 | `--strict-real` did not actually exclude simulated data | **Critical** — the artifact could have misreported its own provenance |
| 8 | `shapley_mode="retrain"` returned all zeros while its efficiency check passed | **Critical** — this is the mode the pre-registration names for the camera-ready ablation |

## 1. The claim head was a detached side task (fixed)

**Symptom.** In the section 9 battery, forcing all six agents to agree
unanimously on a wrong option *while simultaneously* marking every claim
`unsupported` with no sources drove accuracy to 0.0099. The synthesizer was
behaving as a pure vote aggregator; the maximally-suspicious evidence
configuration produced no resistance at all. Corroborating this, role-label
permutation moved accuracy by only 0.0016, so the role embeddings were near
inert too.

**Cause.** `CCASynthesizer.forward` computed `claim_accept_logit` *after* the
option scorer, from `summary`. There was no gradient path from adjudicated
evidence to the answer decision, so the claim head was trained only by its own
weakly-weighted auxiliary loss (`w_claim=0.25`) and the answer head never had a
reason to consult it.

**Fix.** Claim adjudication now runs upstream of the answer head, and its output
gates vote aggregation in `OptionScorer` with weight
`floor + (1 - floor) * credibility_r`, `floor = model.evidence_gate_floor = 0.25`.
Two tests pin this: `test_evidence_gates_vote_aggregation` (degrading only claim
evidence must move the option logits) and
`test_evidence_credibility_gradient_reaches_claim_head` (the answer loss alone
must produce gradient in the claim head).

**Effect on the probe: none.** Clean accuracy 0.8514 -> 0.8227;
`evidence_free_agreement` 0.0099 -> 0.0000. The architectural defect is real and
is fixed, but fixing it did not change the outcome, for the reason in §2.

## 2. Relative gating cannot express "nobody has evidence"

The gate is **relative between agents**. When every agent loses credibility at
once, all vote weights scale by the same factor and the option ranking is
unchanged. Gating can discount a low-credibility *minority*; it cannot express
"no one here has evidence, so distrust the whole council."

Expressing that requires an absolute signal routed to abstention rather than to
option ranking. `supporter_credibility` and `base_vote` are both available to
the scorer, so the capacity exists; what is missing is a training signal that
rewards using it (see §3).

## 3. The abstention head does not fire under attack

Measured abstention rate under every section 9 probe: **0.0000**.

| probe | accuracy | abstention | baseline fallback |
| --- | --- | --- | --- |
| clean | 0.8227 | — | — |
| prompt injection via agent output | 0.7126 | 0.0000 | 0.2619 |
| confidently wrong majority | 0.2800 | 0.0000 | 0.1708 |
| evidence-free agreement | 0.0000 | 0.0000 | 0.2291 |
| role-label permutation | 0.8177 | 0.0000 | 0.2849 |

Accuracy alone cannot distinguish "followed the attack" from "correctly
refused": under probes that make the gold answer unreachable, low accuracy is
unavoidable and the meaningful question is whether the system abstains or falls
back. The robustness suite now reports all three columns for exactly this
reason.

**A rebalancing attempt was tried and reverted.** The abstention target fires on
only a few percent of items, so a BCE `pos_weight` seemed indicated. It cost 4.4
points of clean accuracy (0.8227 -> 0.7791) and nearly tripled ECE
(0.0418 -> 0.1165) while leaving abstention at 0.0008. The head is not merely
under-trained: the probes are **out of distribution**. Nothing in training ever
showed the model an input with all evidence stripped, so a head fitted on
in-distribution data has no reason to fire on one.

**Candidate fix, to pre-register rather than patch in:** adversarial training,
augmenting the training distribution with the section 9 perturbations. That
changes what the model is trained to do and would invalidate a naive comparison
against the current arm, so it belongs in `PREREGISTRATION.md` as a declared
condition with its own hypothesis, not as an undeclared improvement.

## 4. Two latent numerical bugs surfaced on the way

Both were invisible until the logit scale changed, and both are now tested.

**Temperature fitting returned `nan`.** The attention/option mask sentinel was
`finfo.min / 4` (about -8.5e37 in float32). Post-hoc temperature scaling divides
logits by `T`; once LBFGS moved `T` below 1, that sentinel overflowed to `-inf`
and the backward pass returned `nan`. `fit_temperature` then returned `nan`,
which silently drove **every** downstream prediction to accuracy 0.0000 with
`nan` calibration metrics. Nothing raised. The sentinel is now derived from the
temperature floor (`MIN_TEMPERATURE = 0.05`) instead of guessed, masked entries
are zeroed before the division, and a non-finite fit falls back to `T = 1.0`
with a warning.

**The float32 sentinel overflowed float16.** The first replacement (-1e9) is
safe in float32 and bfloat16 but not in float16, whose maximum is 65504:
`-1e9 / 0.05` is `inf` there. This would have been a CUDA-only failure, invisible
on MPS and CPU. The sentinel is now computed per dtype as
`-min(1e9, finfo.max * MIN_TEMPERATURE / 4)`.

## 5. What this means for the pilot (P13)

Three things to check first on real records:

1. Whether real claim evidence carries more signal relative to votes than the
   simulator's does. If it does, the gate will matter more than it does here; if
   it does not, the synthesizer is a fancy majority vote and should be reported
   as one.
2. Whether abstention fires on genuinely unresolvable real items, as opposed to
   the out-of-distribution probes.
3. The real pairwise error correlation, against the simulator's 0.10 assumption.

## 6. Every agent pair is substitutable, at the same magnitude as the main effects

Added 2026-07-28, from the interaction analysis P12 introduced
(`engine.interaction_analysis`). Simulated data, 800 items, masked mode.

Section 8 says independent Shapley additivity "must not be assumed". It is now
measured, via the exact Shapley–Owen interaction index over all 64 coalitions.
The result is not marginal:

| | value |
| --- | --- |
| Pairs with an interval excluding zero | **15 of 15** |
| Direction | **all negative** (substitutable) |
| Mean \|interaction\| | 0.0964 |
| Max \|interaction\| | 0.1371 (`local_claim_checker` + `explore_then_prune`) |
| Per-role Shapley values, for scale | 0.0356 – 0.0836 |

**The interactions are the same order of magnitude as the main effects, and in
several cases larger.** `local_claim_checker` + `explore_then_prune` interact at
−0.1371 while the largest single-role Shapley value is +0.0836.

**What this means.** Every pair of agents contributes *less together than
separately*. This is the coalition-side signature of limitation L7: the agents
are largely redundant, so a coalition that already contains one of them gains
less from adding a second. It is the same phenomenon the pairwise error
correlation measures, seen through the value function instead of the error
vectors — and it is consistent with the co-failure ceiling reported across 67
frontier models (arXiv:2606.27288, see `NOVELTY_SEARCH.md` §3.5).

**Three consequences that are now enforced in code, not prose:**

1. **The per-role Shapley table alone misdescribes the council.** Reporting six
   positive Shapley values without the interaction table implies six independent
   contributions summing to the total. They do not; the sum is only exact
   because the efficiency axiom forces it to be, which is an identity rather
   than evidence. `REPORT.md` now says exactly that under the additivity gap.
2. **No role can be called necessary.** `necessity_sufficiency_verdicts` lists
   every substitutable pair and states that a role substituting for another
   cannot be necessary whatever its own interval shows. With all 15 pairs
   substitutable, that rules out the word for all six roles independently of the
   replication requirement.
3. **It bounds H1 before the real run.** If the same pattern holds on real data,
   the council's headline gain is capped well below the sum of its parts, and
   the budget-matched comparison — six calls versus one — becomes correspondingly
   harder to win. This is a reason to expect H1 to be tight, registered here
   before any real result is observed.

**Caveat, and it is the important one.** These are simulated records. The
simulator deliberately induces correlated common-mode failure (error correlation
≈ 0.10), so finding redundancy in it is close to finding what was put there.
What is *not* built in is the magnitude, the uniformity across all 15 pairs, or
the fact that interactions rival main effects. Those are properties of the
value function the synthesiser induces, and the pilot (P13) should check whether
real agents reproduce them. If real interactions are materially smaller, the
council has more genuine diversity than the simulator assumes and H1 is more
reachable than this suggests.

## 7. `--strict-real` did not actually exclude simulated data (fixed)

Found and fixed 2026-07-28. This is the most consequential defect in this
document, because it was a hole in the one guarantee the repository makes about
its own honesty.

**Symptom.** A run driven entirely by `StubProvider` — the offline fake used for
testing, which fabricates every response — produced records stamped
`"provenance": "real"`. Feeding those records to
`python main.py all --records ... --strict-real` produced a complete
`REPORT.md` with `provenance: real`, no banner, and no warning. Every number in
it would have been fabricated.

**Cause, in two independent places.**

1. `orchestrator/council.py` and `orchestrator/conditions.py` hardcoded
   `"provenance": "real"` into every emitted record, regardless of which
   provider produced it.
2. `dataset.build_datasets` set `provenance = "real"` on the basis that
   `cfg.data.records_path` was non-empty — that is, it inferred "real" from
   *"this came from a file"* and never read the per-record stamp at all.

Either one alone defeats the guard. `README.md` states that "`--strict-real`
refuses to run at all on synthetic data"; that was not true of stub output.

**Fix.**

- `Provider.synthetic` is a declared class attribute defaulting to **`True`**.
  The fail-safe direction is deliberate: a provider that forgets to declare
  itself gets its output stamped *simulated*, costing a refused run, whereas the
  opposite default silently admits fabricated records. Only `AnthropicProvider`
  and `OpenAIProvider` set it `False`.
- `Council.provenance` is derived from the providers, and **any** synthetic
  provider taints the whole record. Five-sixths real is not real.
- `build_datasets` reads the stamps and treats the corpus as real only if every
  record says `"real"`. A missing stamp, a `"simulated"` stamp, or a mixture all
  yield `"simulated"`, with a `RuntimeWarning` naming what it saw.

**Four tests pin it**: stub output is stamped simulated; one synthetic provider
taints a council; `Provider.synthetic` defaults to `True`; and a stub-generated
corpus makes `cfg.validate()` raise under `eval.strict_real`.

**Why it mattered more than the others in this document.** The other findings
are about whether the architecture works. This one was about whether the
artifact tells the truth about where its numbers came from — and the whole
`/logistics` package rests on that distinction holding.

## 8. `shapley_mode="retrain"` returned all zeros (fixed)

Found 2026-07-28, on the first end-to-end execution of that mode. It had been
implemented but never run — `REMAINING_PROMPTS.md` P11 item 5 asked for exactly
this validation, and it was warranted.

**Symptom.** The retrain sweep completed all 64 coalitions without error and
produced:

| | retrain | masked (same config) |
| --- | --- | --- |
| Distinct coalition values across all 64 | **1** | 30 |
| v(∅) | 0.0 | 0.0 |
| v(A) | **0.0** | 0.3314 |
| Every role's φ | **0.0000, CI [0, 0], P(φ>0) = 0** | +0.036 … +0.084 |
| Efficiency check | 0.00e+00 — **passing vacuously** | 0.00e+00 |

Every item in every coalition scored exactly zero.

**Cause.** `cmd_ablate` fits the logit temperature and the selective-prediction
threshold **once**, on the calibration split, using the main
coalition-dropout-trained model — then passes that single threshold into
`_retrain_coalition_sweep`, which applies it to all 64 *independently retrained*
models. Those models never receive temperature fitting at all, and each has its
own selection-score scale. The inherited threshold sat above every retrained
model's scores, so every item abstained, and with `abstain_credit = 0.0` every
`v(S)` collapsed to zero.

**Why nothing caught it.** The efficiency axiom check — the ablation's main
self-test — compares `Σφ` against `v(A) − v(∅)`. When everything is zero that is
`0 == 0 − 0`, which passes. A test that only checks an identity cannot detect
the case where every term is degenerate.

**Fix.** Each coalition's model is now calibrated on the calibration split
individually, inside the loop: fit its own temperature, fit its own abstention
threshold, then evaluate on test. Calibration never touches the test split, per
§7.

**Why this one matters most for the paper.** `PREREGISTRATION.md` §7 designates
`retrain` as the mode for the **camera-ready ablation number**, with masked
reported alongside as a methods comparison. Had the full run been executed
against the registered plan without this fix, the headline ablation table would
have reported that no role contributes anything — a clean, symmetric,
efficiency-checked table of zeros that looks like a finding and is an artifact.

**Generalisable lesson, worth stating in the methods section.** Any ablation
that retrains per condition must recalibrate per condition. Sharing a threshold
across independently trained models is not a minor approximation; it silently
destroys the comparison. This applies to the real run as much as the simulated
one.

### What the corrected comparison shows

With the fix, both modes run on identical configuration and disagree by more
than the pre-registration's per-role effect size:

| role | masked | retrain | gap |
| --- | ---: | ---: | ---: |
| `counter_evidence_checker` | +0.0618 | +0.0356 | −0.0262 |
| `consistency_checker` | +0.0381 | +0.0510 | +0.0129 |
| `local_claim_checker` | +0.0566 | +0.0711 | +0.0145 |
| `explore_then_prune` | +0.0836 | +0.0607 | −0.0230 |
| `rare_hypothesis_checker` | +0.0556 | +0.0560 | +0.0004 |
| `confidence_calibrator` | +0.0356 | +0.0570 | +0.0214 |

**mean |gap| 0.0164 · max |gap| 0.0262.** H2's SESOI is **+0.01**.

1. **The approximation error exceeds the effect H2 seeks.** The masked mode
   cannot support per-role conclusions at the registered effect size. It is
   still fine for iteration and for the full-council contrast, where both modes
   agree exactly (v(∅) = 0.0000, v(A) = 0.3314 in both).
2. **Rank order is not preserved.** Masked ranks `explore_then_prune` first and
   `counter_evidence_checker` second; retrain ranks `local_claim_checker` first
   and `counter_evidence_checker` **last**. "Role X contributes most" is an
   artifact of the mode unless computed under `retrain`.

This gives §7's choice of `retrain` for the camera-ready number a quantitative
justification it previously lacked — and it is itself the "reportable methods
finding" `REMAINING_PROMPTS.md` P11 item 5 asked for.

## Verified unaffected

Agent-permutation invariance (max |Δlogit| 4.8e-07), option-permutation
equivariance (agreement 1.0, max |Δlogit| 0.0), Shapley efficiency
(|Σφ − (v(A) − v(∅))| = 1.1e-16), and all 30 invariant tests.
