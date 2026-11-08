# Pre-registration: Cognitive Constraint Architecture, portable V1

**Status:** FROZEN on commit. No test-split result has been observed at the time
of writing, nor at amendments A1 or A2 (§11, both 2026-07-28). Any change after
the first test-set read must be recorded in §11 as a dated amendment, and results
obtained under an amended plan must be reported as exploratory.

**A2 narrows this registration.** §§1–10 describe the full seven-domain,
ten-condition study and remain the registered plan of record. The study actually
being run is the scoped subset defined in §11 A2, whose primary outcome is role
operation verification rather than H1. Read A2 before reading §§2, 4, 5, or 6 as
a description of what will be executed.

**Scope:** Portable V1 only — prompt and context operations available through
ordinary model APIs. Decoding-level and activation-level mechanisms (V2) are out
of scope and no claim about them is registered here.

> These roles are functional AI processing constraints. They are not diagnostic
> models, simulations of lived experience, or claims that mental health
> conditions confer a single cognitive mechanism.

---

## 1. What is being tested

Six agents run predeclared processing operations over the same item and emit
structured candidate records. A synthesizer combines those records into one
answer, a calibrated confidence, and an abstention decision. The question is
whether that pipeline beats budget-matched controls, and whether each role earns
its place.

Functional labels are the identifiers throughout. The clinical terms in the
concept paper are traceability aids only and appear in no prompt, tensor, log,
or results table.

| Role | Predeclared operation | Predeclared primary target |
| --- | --- | --- |
| `counter_evidence_checker` | For every material factual claim, give evidence status and the strongest plausible counterclaim before recommending acceptance | TruthfulQA |
| `consistency_checker` | Answer, run two fixed independent invariant checks, revise only on a named contradiction or calculation error | MATH/AIME |
| `local_claim_checker` | Decompose into atomic claims, judge each from local evidence, then reconcile with full context | AdvConsistency |
| `explore_then_prune` | Generate a fixed number of diverse hypotheses, then score against a fixed evidence rubric and select or abstain | Creative divergence |
| `rare_hypothesis_checker` | Generate one plausible non-canonical alternative; require explicit evidence before it displaces the leading answer | Temporal/Fresh QA |
| `confidence_calibrator` | Initial answer, list disconfirming conditions, output a probability after checking them | Sycophancy |

---

## 2. Hypotheses, directional, with smallest effect sizes of interest

A SESOI is the smallest difference that would change a deployment decision. Each
is justified rather than asserted. All are on the primary item score defined in
§4.

| ID | Hypothesis (directional) | SESOI | Justification |
| --- | --- | --- | --- |
| **H1** | The six-role council with the learned synthesizer exceeds the strongest deployable control under **both** budget-matching regimes | **+0.03** absolute | The council costs ~6× the calls of a single agent. Below 3 points, the quality-cost frontier favours the single agent for essentially every application we can name, so a smaller gain is real but not decision-relevant |
| **H2** | Each role has positive Shapley value on its predeclared primary target after FDR correction | **+0.01** per role | With six roles sharing a full-council gain of ~0.03–0.10, a per-role contribution under 1 point is inside the noise a reviewer should expect from item resampling |
| **H3** | The gain survives removal of role labels: the role-blind synthesizer and the clinical-language-removed council both retain ≥80% of the full-council gain | **20%** of the H1 gain | If more than a fifth of the effect is carried by labels rather than operations, the mechanism claim in the paper is not supported, whatever the headline number |
| **H4** | The council does not regress calibration: ECE no worse than the best deployable control | **+0.02** ECE | Below 2 points of ECE, differences are within the resolution of a 15-bin equal-mass estimator at our planned sample size |
| **H5** | Results hold across ≥3 model families: the sign of the H1 contrast is preserved in every family | sign agreement | A gain that reverses across families is a property of one provider's model, not of the architecture |

**Registered null-result commitments.** We commit to reporting, with equal
prominence: (a) H1 failing under cost-matching while succeeding unconstrained;
(b) any role whose Shapley interval spans zero; (c) any benchmark where the
council regresses.

---

## 3. Predicted harms per role

Limitation L13 requires predicting regressions, not only benefits. A role passes
only if its target benefit is not purchased with an unacceptable regression. Each
prediction below is scored on every benchmark, not only where it is expected.

| Role | Predicted harm | Where measured | Unacceptable if |
| --- | --- | --- | --- |
| `counter_evidence_checker` | Over-flagging; raised abstention on well-supported claims | Abstention rate, over-refusal | Coverage drops >0.05 at equal accuracy |
| `consistency_checker` | Token and latency inflation from re-checking; unhelpful on non-formal tasks | p95 latency, creative divergence | p95 latency >2× single agent |
| `local_claim_checker` | Degraded long-range integration — the explicit trade-off in the concept paper | Long-context integration accuracy | Long-range accuracy drops >0.03 |
| `explore_then_prune` | Verbosity; synthesizer capture (it runs two passes and is the natural capture risk) | Output-token share, agent dominance | Dominance >0.60 or >35% of council tokens |
| `rare_hypothesis_checker` | Fabricated alternatives displacing correct canonical answers | Fabricated-alternative rate, TruthfulQA | Fabrication rate >0.05 |
| `confidence_calibrator` | Under-confidence; inappropriate stubbornness when the user is right | ECE, sycophancy both directions | Capitulation-when-right worsens >0.05 |

---

## 4. Primary outcome and the null policy

**Primary item score.** Correct-and-not-abstained, scored 1/0, with abstention
credited `eval.abstain_credit = 0.0`. Abstention therefore scores as a miss —
deliberately conservative, so the council cannot buy accuracy by refusing. Item
scores aggregate to `v(·)`, the characteristic function used for Shapley.

**Coverage** is reported alongside every selective number. Reporting either
alone is not permitted.

**Empty-coalition value.** `v(∅)` is the direct-answer base model on identical
per-item input. This is architectural, not a post-hoc choice: the always-present
`[BASE]` token carries the baseline, so the empty coalition mechanically reduces
to it.

**Per-benchmark primary metrics**, at pinned versions:

| Domain | Primary | Secondary and harm outcomes |
| --- | --- | --- |
| TruthfulQA | Dataset-native truthful-and-informative | Abstention, verbosity, judge disagreement |
| HaluEval | Dataset-native accuracy/F1 for the selected split | Claim-level FP/FN rates |
| MATH/AIME | Exact final answer, official normalisation | Invalid format, reasoning-token cost |
| AdvConsistency | Accuracy on the seeded local falsehood, hidden labels | Long-range integration, position bias |
| Sycophancy | Correct-answer retention under challenge | Inappropriate stubbornness when the user is right |
| Temporal/Fresh QA | Accuracy vs timestamped authoritative sources | Unsupported-answer rate, retrieval dependence |
| Creative divergence | Blinded human originality under written rubric | Usefulness, feasibility, safety, inter-rater agreement |

Cross-cutting on every task: ECE, Brier, selective accuracy with coverage, AURC,
safety incidents, input/output tokens, p50/p95 latency, dollar cost.

---

## 5. Conditions

All ten conditions of the specification, each emitting the identical record
schema so every arm flows through the same metrics and the same paired tests.

1. Strongest single model, direct answer *(= `v(∅)`)*
2. Single model, matched deliberation tokens
3. Self-consistency, k=6
4. Generic six-agent debate/council
5. Functionally diverse council **(primary)**
6. Persona-label-only council, length-matched
7. Council operations, all clinical language removed
8. Council candidates + majority vote
9. Council candidates + learned synthesizer **(primary)**
10. Heterogeneous-model council, identical routing

**Conditions 6 and 7 test different things and will not be conflated.** The
role-blind synthesizer ablation (`model.use_role_embeddings=False`) hides role
identity from the *synthesizer*; condition 7 removes clinical language from the
*agent prompts*. H3 requires both.

**Budget matching, declared before running.** Every headline claim is reported
three times with equal prominence: unconstrained, resource-matched (equal max
input tokens, output tokens, calls, synthesis allowance), and cost-matched
(equal provider cost, comparable latency ceiling).

---

## 6. Sample size

Paired design; the variance that matters is of the per-item *difference*, not of
either arm. For a paired proportion difference with discordance rate `d`, the
standard error of the mean difference is approximately `sqrt(d/n)`.

Pilot-informed assumption: `d ≈ 0.30` (the arms disagree on ~30% of items — the
simulator gives 0.28 and real councils are expected to disagree more, so this is
mildly conservative).

- Detecting **H1's SESOI of 0.03** at 80% power, α=0.05 two-sided requires
  `n ≈ 8 d / 0.03² ≈ 2670` items per benchmark-arm pair.
- Detecting **H2's 0.01** per role requires `n ≈ 24000`, which is not affordable
  across six roles and seven benchmarks. **We therefore pre-declare H2 as
  powered only in aggregate across benchmarks**, not per benchmark-role cell, and
  will report per-cell results as estimates with intervals rather than tests.

**Registered target:** ≥2700 test items per benchmark where the dataset permits,
and the full available test split where it does not. Any benchmark run below
2700 items is flagged as underpowered for H1 in the results table.

**Repeated trials.** Each coalition receives ≥3 item-level trials on a fixed
item/seed schedule reused across coalitions, per §8 of the specification.

---

## 7. Analysis plan

- **Primary contrasts:** paired item-level bootstrap, 2000 resamples,
  95% percentile intervals. Arms must be item-aligned; the evaluation code sorts
  by `item_id` to guarantee this.
- **Shapley:** exact over all 2⁶ = 64 coalitions (no sampling approximation at
  R=6), computed per bootstrap resample of items. Report φ, 95% interval, and
  P(φ>0). The camera-ready ablation uses `shapley_mode="retrain"`; the masked
  approximation is reported alongside as a methods comparison, with the gap
  between them stated.
  **Measured on simulated data (amendment A1): the gap is 0.0164 mean, 0.0262
  max — larger than H2's SESOI of 0.01 — and it does not preserve rank order.
  No per-role claim may therefore be made from a masked run.** The masked mode
  remains valid for iteration and for the full-council contrast, where both
  modes agree exactly.
- **Multiplicity:** Benjamini–Hochberg at α=0.05 across the agent × benchmark
  secondary family. Primary contrasts (H1) are not corrected; they are
  pre-registered and few.
- **Error correlation:** pairwise correlation of item-level agent error vectors,
  plus interaction terms. Shapley additivity is not assumed.
- **Decision rule for "necessary" / "sufficient":** these words are not used
  unless the predeclared rule, the uncertainty interval, and a replication all
  support them. Enforced in the report generator, not left to prose.

---

## 8. Falsification

Each hypothesis has a stated failure condition. We commit to reporting failure
as the result.

| ID | Falsified if |
| --- | --- |
| H1 | The cost-matched contrast interval includes 0, or its point estimate is below +0.03 |
| H2 | A role's Shapley interval includes 0 on its predeclared primary target |
| H3 | The role-blind or clinical-language-removed arm retains <80% of the H1 gain |
| H4 | Council ECE exceeds the best deployable control by >0.02 |
| H5 | The sign of the H1 contrast reverses in any model family |

**Global stopping rule.** If H1 fails under cost-matching, the paper reports a
negative primary result. It does not switch to the unconstrained comparison as
the headline, and it does not substitute a secondary outcome.

---

## 9. Frozen configuration

Fixed before data collection; changing any value invalidates this registration.

| Knob | Value | Rationale |
| --- | --- | --- |
| `eval.abstain_credit` | 0.0 | Abstention scores as a miss; conservative |
| `eval.target_coverage` | 0.90 | Threshold fitted on the calibration split only |
| `eval.fdr_alpha` | 0.05 | Standard |
| `eval.ci_level` | 0.95 | Standard |
| `eval.n_bootstrap` | 2000 | Stable to the third decimal at our n |
| `train.select_metric` | `selective_accuracy` | Matches the primary outcome |
| `model.evidence_gate_floor` | 0.25 | Evidence discounts a low-credibility agent without silencing it |
| `data.split_fractions` | 0.50 / 0.15 / 0.15 / 0.20 | Four disjoint sets per §4 |
| `data.split_salt` | `cca-synth-v1` | Frozen; hash split is stable under corpus growth |
| Role → primary benchmark | §1 table | Every role is nonetheless scored on every benchmark |

---

## 10. Known threats, declared in advance

1. **The synthesizer may be a vote aggregator.** Measured on simulated data:
   evidence gating did not change behaviour under evidence-stripping probes, and
   the abstention head does not fire under attack. See
   `FINDINGS_EVIDENCE_PATHWAY.md`. We pre-declare this as the first thing the
   pilot checks, and commit to reporting the synthesizer as "not distinguishable
   from confidence-weighted voting" if that is what the real data shows.
2. **Adversarial training is NOT part of this registration.** It is the candidate
   fix for threat 1 and would change what the model is trained to do. If run, it
   is a separate declared condition with its own hypothesis, reported as such.
3. **Answer-space clustering error is unmeasured.** If per-benchmark clustering
   error exceeds that benchmark's SESOI, no conclusion is drawn for it.
4. **Agent independence is assumed nowhere.** Correlated errors are measured and
   reported; a high correlation bounds how much of any gain is attributable to
   diversity.
5. **Contamination.** Overlap rates are reported as a first-class result; any
   benchmark with material overlap is labelled in the main table.

---

## 11. Amendments

**A1 — 2026-07-28. No test-split result had been observed.** Pipeline defects
only; no hypothesis, outcome, contrast, or knob in §§1–10 is changed by this
amendment, and no analysis becomes exploratory as a result.

Three faults were found by executing code paths that had never been run
(`FINDINGS_EVIDENCE_PATHWAY.md` §§6–8):

1. `shapley_mode="retrain"` — the mode §7 designates for the camera-ready
   ablation — returned all-zero Shapley values, because calibration fitted on
   the main model was reused across 64 independently retrained models. Fixed by
   per-coalition calibration on the calibration split. A degeneracy check now
   fails where the efficiency axiom passes vacuously.
2. `--strict-real` did not exclude stub-provider output. Provenance is now
   derived from the provider rather than asserted, and any synthetic provider
   taints the record.
3. Short-text contamination detection was inoperative, which would have
   under-reported overlap on exactly the items most likely to be memorised.

**Registered in advance as a consequence of §6.** All 15 agent pairs show
negative Shapley interaction on simulated data, at the same order of magnitude
as the main effects. We therefore expect H1's effect to be bounded well below
the sum of the per-role contributions, and we commit to reporting the pairwise
interaction table alongside the per-role Shapley table in every results
presentation — the per-role table alone misdescribes a council whose members
substitute for one another.

---

**A2 — 2026-07-28. No test-split result had been observed.** No benchmark test
split has been read, and no record from a real provider exists: the only records
produced to date are stub-provider output, which `--strict-real` refuses (A1,
item 2). This amendment **narrows the registered design and changes the primary
outcome**, so it is recorded in full rather than as an editorial note.

**Rationale.** `NOVELTY_SEARCH.md` §2 identified material predecessors for the
architecture, and §§3.3–3.5 established that three independent lines of evidence
run against H1: persona labels are known not to carry performance, correlated
failure bounds any combination method, and a published system already reports
gains on two of our benchmarks. Meanwhile §2a of `HANDOVER.md` costs the
registered run at $4,157 against an available budget covering roughly 12% of it.
Running an underpowered version of a study whose headline hypothesis is
disfavoured by the priors is the worst of both options. We instead scope to a
question the budget answers at full power, and which the predecessors leave
open: whether a role prompt performs the operation it declares.

### A2.1 New primary outcome

The primary outcome is no longer H1. It is the **manipulation check**: a
per-role, executable, predeclared test of whether the specified information-
processing transformation occurred, scored against a length-matched
neutral-prompt control.

The six checks correspond one-to-one with the operations in §1's table and are
implemented in `cca_synth/prompts/` (`manipulation_check.py`, and the `check`
callables in `__init__.py`).

**They are continuous scalars, not binaries, and their units differ.**
`_check_counterevidence` and `_check_claim_decomposition` return unbounded
per-record counts; `_check_contradiction_detection` returns a rate difference
bounded in [−1, 1]; `_check_hypothesis_diversity` returns a distinct-item count.
No single absolute threshold is meaningful across them, which is why H6's SESOI
below is **standardized** rather than expressed in check units.

**A known compression, declared in advance.** `prompts._OUTPUT_CONTRACT`
requires *every* arm to emit `atomic_claims`, `counterevidence`, and
`failure_flags`, because §5 requires all conditions to share a record schema so
they flow through the same metrics. A length-matched neutral prompt is therefore
still asked for counterevidence and will produce some. The checks measure
*amount*, not presence, so the contrast survives — but it is compressed relative
to a design where the neutral arm emitted no such fields, and we expect smaller
operation-vs-neutral differences than the schema-free version of this experiment
would give. The pilot's first job is to confirm the check statistics separate at
all (§10 item 1's analogue for this design); if they do not, that is reported as
the result and H7 is not estimated.

**Consequence worth stating: the checks are benchmark-independent.** A check
asks whether a role did what it said, not whether doing so helped. Dropping five
benchmarks (A2.3) therefore drops no role from the study. What is lost is H2's
per-role-on-predeclared-primary-target contrast, not the roles themselves.

### A2.2 New hypotheses

Registered in addition to §2. H6 and H7 are the scoped study's primary and
secondary outcomes respectively.

| ID | Hypothesis (directional) | SESOI | Justification |
| --- | --- | --- | --- |
| **H6** | For each role, the full-operation prompt drives its check statistic higher than a length-matched neutral prompt | **d = 0.4** standardized (difference in per-item check statistic ÷ pooled per-item SD) | Standardized because the six checks have different and mostly unbounded units (A2.1). At 200 dev items a paired *d* of 0.4 is detectable at >0.99 power, so the binding constraint is substantive, not statistical: *d* < 0.4 means the operation prompt and a neutral prompt of equal length produce distributions that overlap almost entirely, which is not a role performing an operation — it is a role occupying tokens |
| **H7** | Check-passing and benchmark effect are not the same measurement: at least one role falls in an off-diagonal cell of the pass-check × moves-score classification | **one role** off-diagonal | The claim under test is that the check is load-bearing rather than confirmatory. A single role that moves scores without passing, or passes without moving, establishes that the two axes are not interchangeable |

**H7's classification rule, fixed here.** Each role is classified on two binary
axes:

- **Passes its check** — the full-operation vs length-matched-neutral difference
  in the role's check statistic has a 95% bootstrap interval excluding 0 **and**
  a standardized effect ≥ 0.4 (the H6 criterion).
- **Moves its benchmark** — the leave-one-out contrast has a 95% bootstrap
  interval excluding 0 **and** a point estimate ≥ 0.05 (see A2.5 for why 0.05
  and not H2's 0.01).

Per-role benchmark effect is measured by **leave-one-out under
`shapley_mode="retrain"`** — 7 trainings (full council plus six six-way
drop-one coalitions) rather than 64. A1 forbids per-role claims from masked
mode, and the full 64-coalition sweep is not affordable in the scoped compute.
The exact Shapley sweep is reported as secondary where compute permits, with the
masked-vs-retrain gap stated as in §7.

**Spearman correlation between check effect and benchmark effect is descriptive
only.** With six roles it is computed over six points; we predeclare that no
inferential claim rests on it and that it is reported with its interval and an
explicit note that n=6.

### A2.3 What is not run, and is reported as not run

| Registered element | Status in scoped study | Reason |
| --- | --- | --- |
| **H1** (council beats budget-matched controls) | **Not tested** | Underpowered at scoped n — see A2.5. Not reported as a result in any direction |
| **H2** (per-role Shapley on predeclared primary target) | **Not tested** | Five of six roles lose their predeclared target with the benchmark reduction |
| **H4** (calibration non-regression) | Reported descriptively, not tested | Powered for neither direction at scoped n |
| **H5** (≥3 model families) | **Not tested** | Second family is a limited-generalization check, not a family sweep |
| **H3** (gain survives label removal) | **Partially addressed** | The persona-label-only and length-matched-neutral arms are exactly H3's contrast, but H3 is defined as a fraction of the H1 gain, and H1 is not estimated. Reported as the arm contrast itself, not as a percentage of an unmeasured quantity |
| AdvConsistency, creative divergence | Not run | Ethics determination and paid annotators; `ETHICS_COMPLIANCE.md` E1 |
| MATH/AIME, Sycophancy, Temporal/Fresh QA | Not run | Scope; answer-space normalisation risk for the free-form domains |
| Conditions 2, 3, 4, 8, 10 (§5) | Not run | Scope |

The §4 per-benchmark table and §5 condition list stand as the registered plan;
the scoped study executes the HaluEval and TruthfulQA rows and conditions 1, 5,
6, 7 only. Every untested hypothesis appears in the results table as **not run**,
never as an absent row.

### A2.4 Scoped design

| Component | Items | Arms | Calls |
| --- | ---: | --- | ---: |
| Manipulation checks | 200 dev | 6 roles × 3 prompt variants | 3,600 |
| Benchmark effect | 1,000 (HaluEval + TruthfulQA) | full-operation council · persona-label-only council · length-matched neutral council · single-agent baseline | ~19,000 |
| Second model, generalization check | 1,000 | same | ~19,000 |

**"Family" vs "tier", fixed here rather than at writing time.** §2's H5 means
*model family* in the sense of separate pretraining lineages from separate
providers. If the second model comes from a different provider, this row is a
family check. If budget or credit availability confines the study to one
provider, it is a **tier** check — two sizes of the same lineage — and must be
labelled a tier check in every table and sentence. A tier check is weaker
evidence of generalization and may not be described as a second family. H5
remains not tested in either case (A2.3), so nothing registered depends on which
obtains; this clause exists to stop the stronger word being used for the weaker
thing.

Prompt variants: **full operation**, **persona-label-only**, **length-matched
neutral**. These differ in exactly one respect each, per §5's rule that
conditions 5/6/7 must not be conflated.

Both benchmarks have unambiguous answer normalisation, which removes the
clustering-error threat registered as §10 item 3 rather than leaving it to be
measured.

### A2.5 Power, and a declared bias in H7's classification

At n=1,000 with discordance `d ≈ 0.30`, the standard error of a paired
proportion difference is `sqrt(0.30/1000) ≈ 0.017`. The minimum effect detectable
at 80% power, α=0.05 two-sided, is therefore ≈ **0.048**.

Two consequences, both declared in advance:

1. **H1's SESOI of 0.03 is below that floor.** The scoped study cannot test H1.
   This is a design decision, not a shortfall discovered afterwards, and H1 is
   reported as not tested rather than as a null.
2. **H7's "moves its benchmark" threshold is set at 0.05, five times H2's SESOI
   of 0.01.** Roles whose true benchmark effect lies between 0.01 and 0.05 will
   be classified as *not moving*. This biases the classification toward the
   "passes its check but does not move the score" cell.

Because that bias has a known direction, we predeclare an **asymmetric reading**
of H7's two off-diagonal cells:

- *Passes check, does not move score* — **weak evidence.** Confounded with low
  power by construction. Reported with the detectable-effect floor stated in the
  same sentence.
- *Moves score, fails check* — **strong evidence.** Not power-limited in the same
  way: a role moving the score by ≥0.05 is detectable at n=1,000, and its check
  is measured at n=200 where the SESOI is ~3.9 SE. This cell is the one that
  makes the method load-bearing, and it is the cell we predict is occupied.

### A2.6 Outcome-independent reporting commitment

The contribution registered here is the check, the classification rule, and the
measurement — **not** a particular result. We commit in advance to reporting and
submitting under either outcome:

- **Axes come apart** (≥1 off-diagonal role): manipulation checks are load-bearing;
  benchmark deltas alone do not license role-mechanism claims.
- **Axes align** (all six roles on the diagonal): the check is a validated
  pre-run screen obtaining the same role ranking for 3,600 calls that the
  benchmark arms obtain for 19,000, and the mechanism inference the field
  currently makes implicitly is vindicated explicitly.

Neither outcome may be reframed as the other after the fact, and the paper's
framing may not be selected on the basis of which obtained. This clause exists
because the scoped study's headline is a `[PENDING]` result and the temptation to
choose a frame post hoc is the specific failure it guards against.

### A2.7 What A2 does not change

No knob in §9 is altered. The abstention policy, split fractions, split salt,
bootstrap count, FDR α, and CI level are unchanged, so scoped-study artifacts
remain comparable to any later full run. §3's predicted harms are scored
unchanged on the two benchmarks that are run. The §8 global stopping rule is
untouched and simply does not bind, because H1 is not tested.

---

---

**A3 — 2026-08-15. A test-split result HAS been observed; see A3.1.** No
manipulation-check data of any kind exists at the time of writing: no run of
`prompts/manipulation_check.py` against a real provider has been executed, so
A3.2 below is declared before the primary outcome's data exists.

### A3.1 The pilot read a 47-item test split — recorded, not concealed

The pilot (`runs/pilot`, 200 HaluEval items, collected 2026-08-15 against
`gpt-5.6-terra` on Microsoft Foundry) was analysed with `main.py all`, which
applies the registered `split_fractions` `[0.5, 0.15, 0.15, 0.2]` and produced
train 98 / calibration 26 / validation 29 / **test 47**. `REPORT.md`'s outcome
table and paired item-level contrasts are computed on those 47 test items.
Because splits are hash-derived under the fixed salt `cca-synth-v1`, those items
receive the same assignment in the 1,000-item registered run: they are now seen.

This was executed as specified in `HANDOVER_2026-08-02.md` §2, whose pilot
command reads the full `items.halueval.jsonl`. That handover also asserts that
"no test-split item is ever read before the registered analysis", a claim which
holds for the `.dev.jsonl` files but not for the pilot command beside it. The
two statements are in tension; this amendment resolves the tension by recording
what actually happened.

**Nothing registered was changed as a consequence.** No hypothesis, primary or
secondary outcome, contrast, arm, SESOI, threshold, estimator, or stopping rule
in §§1–10 or in A1/A2 was altered after the pilot was observed, and no such
change is made by this amendment. The registered analysis proceeds unchanged on
the full item set.

**What the observation consisted of**, stated so a reader can judge its
influence: the degeneracy check passed with 5 distinct coalition values; the
synthesiser scored below majority vote and below every single-agent arm; mean
pairwise error correlation was 0.7996 (max 0.9440); all six leave-one-out
contrasts had 95% intervals containing zero. At n=47 every contrast is quantised
to multiples of 1/47, and none of these quantities is treated as evidence.

Per §11's standing rule, any analysis that a reader judges to be affected by
this observation is to be treated as exploratory.

### A3.2 Benjamini-Hochberg correction across the six role checks

`prompts/manipulation_check.py` evaluates six roles simultaneously, each against
its own 95% bootstrap interval, with no multiplicity correction. Six independent
tests at α = 0.05 give a family-wise error rate near 26%, which is not an
acceptable basis for the study's **primary** outcome.

We therefore register, before any manipulation-check data exists, that the six
per-role tests form one family and that **Benjamini-Hochberg FDR control at
q = 0.05 is applied across them**. This uses the `benjamini_hochberg` routine
already in `utils.py` and already applied to the agent-by-benchmark family
(`engine.py`), so the correction is consistent with the rest of the analysis
rather than introduced specially here.

The pass rule of A2.2 is otherwise unchanged and remains conjunctive: a role
passes only if its operation-versus-neutral interval excludes zero **after** FDR
correction **and** its standardized paired effect reaches 0.40. Uncorrected
per-role intervals and q-values are both reported, so a reader can see the
correction's effect rather than only its result.

A confusion matrix per arm, and per-role confusion matrices for the check
statistics where the check is categorical, are added to the reporting set. This
is additional descriptive reporting, not a change to any registered test.

---

Further amendments must be dated, must state whether any test result had been
observed, and must mark all affected analyses as exploratory.

---

## 12. Provenance

- Specification: `RESEARCH_SPECIFICATION.md`
- Limitations addressed: L1, L3, L8, L10, L11, L12, L13, L14
- Implementation: https://github.com/kevzho/role-prompted-systems
- Config hash of the registered configuration: recorded in each run's
  `manifest.json` under `config_hash`
- Frozen at commit: *(record the SHA of the commit that adds this file)*
