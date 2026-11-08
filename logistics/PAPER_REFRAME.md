# Paper reframe: from CCA to manipulation checks

Target: **NeurIPS 2026 workshop** (NeurIPS's suggested contribution date is
2026-08-29; individual workshops set their own and many land earlier — see §9).
The expanded version targets **ICML 2027** (~late Jan 2027).

**ICLR 2027 is dropped.** Its deadline is 2026-09-24, 3.5 weeks after the
workshop date — not enough time to add the domains, model families, or item
counts a main track rejects on, so it would mean submitting the same eight-page
workshop paper into a venue that wants scale. ICML's late-January deadline gives
~5 months to add them, and is the first deadline the ethics-gated components
(`ETHICS_COMPLIANCE.md` E1, 10–16 serial weeks) could plausibly meet.

Written 2026-07-28. Every number marked **[PENDING]** is a placeholder for the
scoped study and must not be written into a draft until measured. The three
attribution findings are real but currently rest on *simulated* records; §5
states exactly what each claim needs. The scoped design is registered as
amendment **A2** in `PREREGISTRATION.md` §11, dated before any test-split read.

---

## 1. Why reframe

`NOVELTY_SEARCH.md` §2 refuted "CCA is the first". `NOVELTY_SEARCH.md` §3.3–3.5
established that the priors run against H1: persona labels are known not to
carry performance, correlated failure bounds any combination method, and
*Council Mode* already reports gains on the same two benchmarks. A paper whose
headline is "the council wins" is betting against four independent lines of
evidence, on a budget that covers 12% of the run needed to test it.

The asset that survives all of that is the *evaluation apparatus*, and
specifically the component that is not standard practice anywhere in the
neighbouring work: a check that a role performed its stated operation before its
benchmark delta is interpreted.

**The reframe costs ~$85 and ~42,000 API calls. The original study costs
$4,157.** That is the whole argument. (Both figures per §6; the earlier ~$50 /
~19,000 estimate omitted the manipulation-check and second-family lines.)

---

## 1a. CCA is not dropped — it becomes the thing under test

The reframe changes CCA's *role in the argument*, not its presence. The six
roles remain the object of study; the manipulation checks exist precisely to
test CCA's central claim.

From the manuscript, §1:

> "An 'optimistic' agent will agree more readily; an OCD-analog agent will
> re-verify k times regardless of valence. **The distinction is mechanistic, not
> cosmetic.**"

That sentence *is* the hypothesis a manipulation check measures. The original
design assumed it and inferred it from downstream benchmark deltas; the reframe
measures it directly and then asks whether it predicts the deltas.

| | Original | Reframe |
| --- | --- | --- |
| The mechanistic claim | Assumed | **Measured** |
| Benchmark delta | Evidence *for* the mechanism | Tested *against* the mechanism |
| Clinical derivation | Load-bearing | Motivation, with the disclaimer |
| Six roles | Present | Present, by functional name |
| Title | Asserts | Asks |

Two consequences worth stating in the draft:

**This addresses H3, the pre-registration's most load-bearing hypothesis —
though not in H3's registered form.** §2 H3 reads "the gain survives removal of
role labels," and the scoped study's three arms — full operation,
persona-label-only, length-matched neutral — are exactly that contrast. This is
not a side issue relabelled as a paper; it is the registered hypothesis that
matters most, run first.

The qualification, per `PREREGISTRATION.md` A2.3: H3 is *defined* as retaining
≥80% of the H1 gain, and the scoped study does not estimate H1. So the paper
reports **the arm contrast itself** — absolute differences between the three
prompt variants — and not a percentage of an unmeasured quantity. That is the
more informative number anyway; the 80% threshold only ever made sense as a
guard on a headline gain the study is no longer claiming.

**It resolves L3 rather than dodging it.** `LIMITATIONS.md` L3 says the IPA
formalism is descriptive rather than operational: `f(C_i, y, x)` is unconstrained,
so any prompt behaviour can be called an IPA after the fact. Tying `f` to a
declared transformation *and its executable check* is what makes it operational.
Keep Definition 1; add the check as the empirical content of `f`.

**The honest risk.** If the checks show the roles do not reliably perform their
operations, that is a negative result about CCA. It is still a publishable
paper — arguably a more useful one — and `PREREGISTRATION.md` §2 already commits
to reporting it. Better to learn it in a four-week workshop paper than after
building the seven-domain study.

---

## 2. Title

**Did the Role Fire? Manipulation Checks for Role-Prompted Multi-Agent LLM
Systems**

Alternatives, in order of preference:

- Verifying Role Operations Before Interpreting Role Effects
- Attribution Without Verification: What Coalition Ablations Do Not Catch

The clinical framing does not appear in the title. It becomes two sentences of
motivation in §2 of the paper, carrying the required disclaimer.

---

## 3. Abstract

> Multi-agent LLM systems assign agents differentiated roles — a critic, a
> verifier, a devil's advocate — and read improvements on downstream benchmarks
> as evidence that the role's information-processing operation produced them. We
> argue this inference is unsupported: a role that does *not* perform its
> operation can still move scores through output length, framing, or ensemble
> variance alone. Experimental psychology closes exactly this gap with a
> *manipulation check* — a direct test that the intended treatment was actually
> delivered before its effect is interpreted. That practice has not carried over
> into multi-agent LLM evaluation, where role effects are routinely reported
> without one.
>
> We introduce **executable manipulation checks**: per-role tests, declared
> before data collection, of whether the specified transformation actually
> occurred, measured against a length-matched neutral-prompt control. We
> implement checks for six functional roles — counter-evidence checking,
> consistency checking, local-claim checking, explore-then-prune, rare-hypothesis
> checking, and confidence calibration — and evaluate three prompt variants per
> role (full operation, persona-label-only, length-matched neutral) on HaluEval
> and TruthfulQA.
>
> **[PENDING]** Of six roles, *N* pass their manipulation check against the
> neutral control. We then classify each role on two axes — passes its check,
> moves its benchmark score — under a rule fixed before data collection.
> *[Branch A, axes come apart:* *M* roles move scores without passing their
> check and *K* pass without moving scores, so a benchmark delta does not
> license a claim about the mechanism that produced it. *Branch B, axes align:*
> the check recovers the same role ordering as the benchmark arms at a fifth of
> the API cost, making it a usable pre-run screen.*]*
>
> We further document two failure modes in coalition-based attribution that pass
> their own correctness tests: (i) a retrain-per-coalition ablation returning
> identically-zero Shapley values while satisfying the efficiency axiom, caused
> by reusing one model's calibration across independently trained coalitions;
> and (ii) a masked approximation whose per-role error (0.026 max, 0.016 mean)
> exceeds the effect size it is used to detect (0.010) and does not preserve
> rank order.
>
> All hypotheses, smallest effect sizes of interest, and predicted per-role
> regressions were pre-registered before data collection. Code, prompts, checks,
> execution manifests, and raw structured records are released.

Word count ≈ 300 with one branch resolved. Trim the final paragraph first if the
venue caps at 250.

**Only one branch is written into the draft** — whichever the data gives. Both
are stated here, and in `PREREGISTRATION.md` §11 A2.6, so that the framing is
fixed before the result is known rather than chosen once it is. Note the
asymmetry A2.5 registers: the *moves score without passing check* cell is strong
evidence, while *passes check without moving score* is confounded with the
study's detectable-effect floor of 0.048 and must be reported with that floor in
the same sentence.

The third attribution finding — pairwise interactions matching main-effect
magnitude — is deliberately **not** in the abstract. See §6a.

---

## 4. Contributions

State these as a numbered list at the end of the introduction.

1. **Manipulation checks for role-prompted agents.** We carry a standard
   instrument from experimental psychology into multi-agent LLM evaluation,
   where it is not current practice: a direct test that a role prompt performed
   its declared information-processing operation, run before any benchmark
   result is interpreted. Each check is executable, is specified against a
   length-matched neutral control, and is fixed in a pre-registration. We
   provide checks for six roles.

   *Framing note.* This is a **method-transfer** claim, not a priority claim.
   `NOVELTY_SEARCH.md` §0 is explicit that a first-pass search cannot establish
   *no predecessor exists* — that is precisely the error that killed the
   architecture's "first" claim, and restaking the paper on a fresh negative
   existential would repeat it. "Not standard practice" is supportable from the
   search we have and survives a reviewer who knows one counterexample.

2. **A measured relationship between passing the check and moving the score.**
   **[PENDING]** We classify six roles on both axes under a rule fixed before
   data collection (`PREREGISTRATION.md` A2.2). If the axes come apart, the
   check is load-bearing rather than confirmatory: a role can move scores
   without doing its job, or do its job without moving scores. If they align,
   the check is a validated pre-run screen recovering the same role ordering for
   3,600 calls that the benchmark arms need 19,000 to obtain. **The contribution
   is the instrument and the rule, not either outcome** — A2.6 commits to
   reporting whichever obtains, and forbids selecting the framing afterwards.

3. **Two self-certifying failure modes in coalition attribution.** Both were
   found by executing a code path that had never been run, both are silent, and
   both are certified by the pipeline's own correctness test. We give the cause,
   the fix, and a degeneracy check that catches the class. A third finding —
   interaction terms matching main-effect magnitude — is reported in §6 with its
   confound stated, and is not claimed as a contribution (§6a).

4. **A pre-registered, release-complete evaluation harness.** Hypotheses and
   SESOIs fixed in advance; predicted per-role *harms* alongside benefits;
   provenance enforced structurally so fabricated records cannot reach a results
   table; exclusions, retries, and failed calls published as a first-class table.
   Released with prompts, manifests, and raw records.

Contributions 1, 3, and 4 exist today. Contribution 2 is the scoped study.

---

## 5. Evidence ledger — what each claim needs

Do not write a claim into the draft before its row is satisfied.

| Claim | Status | Needs |
| --- | --- | --- |
| Verifying role operations is **not standard practice** | Supported by `NOVELTY_SEARCH.md`, **first-pass search only** — and this weaker claim is what a first-pass search *can* support | Still reproduce §0's backlog: native ACL Anthology / OpenReview / Semantic Scholar queries with screened counts. ~1 day. Needed to say "we found none in *N* screened papers" rather than "not standard" unquantified |
| ~~No prior work verifies role operations~~ | **Withdrawn** | Unsupportable by any first-pass search. Do not restate in any form |
| Manipulation checks are implementable | **Done** — `cca_synth/prompts/`, tested | — |
| Roles pass/fail their checks (contribution 2) | **[PENDING]** | The scoped study, §6 |
| Retrain mode can return all zeros | **Measured**, simulated records | Re-confirm on real records in the scoped study, or state "observed in our pipeline" and give the mechanism |
| Masked-vs-retrain gap exceeds SESOI | **Measured**, simulated (0.0164 mean / 0.0262 max) | Same. The *mechanism* generalizes; the *magnitude* is ours |
| Interactions match main-effect magnitude | **Measured**, simulated (all 15 pairs) — **confounded**, see §6a | Real records, plus the simulator's induced correlation reported as a parameter. Demoted from the abstract until then |
| Provenance can be enforced structurally | **Done**, 4 tests | — |

**The honest framing for the attribution findings:** they are *existence proofs
plus mechanisms*, demonstrated in our pipeline. That is a legitimate methods
contribution — "this can happen, here is why, here is the check that catches
it" — and it does not require claiming a measured rate in the wild. Write them
that way and findings (i) and (ii) are unassailable. Finding (iii) is not, for
the reason in §6a.

---

## 6. The scoped study

| Component | Calls | Cost (Sonnet 5, batched + cached) |
| --- | ---: | ---: |
| Manipulation checks: 6 roles × 3 variants × 200 dev items | 3,600 | $7 |
| Benchmark effect: HaluEval + TruthfulQA, 1,000 items, 4 arms | ~19,000 | $39 |
| Second model family (limited generalization) | ~19,000 | $39 |
| **Total** | **~42,000** | **~$85** |

Arms: full-operation council · persona-label-only council · length-matched
neutral council · single-agent baseline. That set tests contribution 2 *and*
gives a cost-matched comparison, because six Haiku calls cost about what one
Opus call costs.

Two benchmarks, not seven. Both have unambiguous answer normalisation, which
sidesteps the clustering risk `normalize.py` documents. AdvConsistency and
creative divergence are out of scope — they need the ethics determination and
paid annotators, and are reported as future work.

**One compute line item the API table does not show.** Per-role benchmark effect
is measured by leave-one-out under `shapley_mode="retrain"` — 7 trainings, not
64 — because A1 forbids per-role claims from masked mode and the full sweep is
not affordable in scope. This is GPU/CPU time, not API spend, and it is the only
part of the scoped study that touches the H10 hardware blocker. Registered in
`PREREGISTRATION.md` A2.2.

**The study is underpowered for H1 by design.** At n=1,000 the detectable effect
is ≈0.048 against H1's SESOI of 0.03. H1 is therefore reported as *not tested*,
never as a null result. A2.5 records this in advance, along with the direction of
the bias it induces in contribution 2's classification.

---

## 6a. Why the interaction finding is not in the abstract

`cca_synth/README.md` states plainly that the simulator "deliberately induces
correlated common-mode failure" so the pipeline is not validated on
unrealistically independent agents. That was the right modelling choice. It also
means the finding *all 15 pairs are substitutable, with |interaction| at or above
main-effect magnitude* is partly downstream of a constant we chose, and a
reviewer will say so in one line.

Findings (i) and (ii) do not have this problem, and the distinction is worth
being precise about:

| Finding | Depends on | Vulnerability |
| --- | --- | --- |
| (i) Retrain mode returns all zeros while efficiency passes | Nothing about the data — `0 == 0 − 0` is arithmetic, and the calibration-reuse mechanism is an engineering fact | **None.** Bulletproof as an existence proof |
| (ii) Masked-vs-retrain gap exceeds the SESOI, rank order not preserved | A property of the *estimator*, not the generating process. The magnitude is ours; the mechanism is general | Low. Report the magnitude as observed-in-our-pipeline |
| (iii) Interactions match main effects | The simulator's induced error-correlation parameter | **High — partly circular.** Cannot carry abstract weight on simulated records |

Disposition: keep (iii) in §6 of the paper, report the induced correlation
parameter alongside it, and state that real records are needed before the
magnitude means anything. Promote it to the abstract only if the scoped study
reproduces it on real records — in which case it becomes the strongest of the
three, because it would then be a measurement rather than an existence proof.

The `PREREGISTRATION.md` A1 commitment to publishing the pairwise interaction
table alongside every per-role Shapley table is unaffected. That commitment is
about *reporting discipline* and stands regardless of what the magnitude turns
out to be.

---

## 7. Paper skeleton (8 pages)

| § | Content | Pages |
| --- | --- | --- |
| 1 | Introduction + contributions | 1 |
| 2 | Related work: role prompting, multi-agent debate, coalition attribution. Cite *Theater of Mind*, *Council Mode*, the co-failure ceiling, the persona literature. Two sentences on clinical inspiration, with the disclaimer | 1.25 |
| 3 | Manipulation checks: definition, the neutral-control requirement, the six checks | 1.5 |
| 4 | Experimental setup: pre-registration, arms, benchmarks, budget matching | 1 |
| 5 | Results: the pass-check × moves-score classification **[PENDING]**, with the 0.048 detectable-effect floor stated in-line | 1.5 |
| 6 | Attribution failure modes: (i) and (ii) with mechanisms and the degeneracy check; (iii) with its simulator confound stated (§6a) | 1.25 |
| 7 | Limitations — draw from `LIMITATIONS.md`; include `NOVELTY_SEARCH.md` §0 on the search's own limits; state H1 as not tested rather than null | 0.5 |

---

## 8. Claims the paper does not make

State this explicitly in §7. It is the cheapest credibility you will ever buy.

- Not that CCA outperforms any baseline. The scoped study is not powered for it,
  and the pre-registration's H1 is not tested here.
- Not that clinical constructs map onto LLM operations. Inspiration only,
  with the disclaimer.
- Not that this is the first work in any category — `NOVELTY_SEARCH.md` §0 caps
  what a first-pass search can support. **This includes manipulation checks.**
  The claim is that verifying role operations is not standard practice, not that
  nobody has done it.
- Not a measured prevalence of the failure modes in published work. They are
  existence proofs with mechanisms.
- Not that interaction terms generally match main-effect magnitude in multi-agent
  systems. On simulated records that finding is confounded with a modelling
  constant we chose (§6a).
- Not a null result for H1. H1 is **not tested**; the scoped study's detectable
  effect is 0.048 against H1's SESOI of 0.03.

---

## 9. Four-week schedule

| Week | Dates | Work |
| --- | --- | --- |
| 1 | Jul 28 – Aug 3 | ~~Amend `PREREGISTRATION.md`~~ — **done, A2, 2026-07-28, before any test-split read**. Materialise + hash HaluEval and TruthfulQA. Run the pilot (~$6). **Fix the target workshop and its real deadline** (see below). Redeem API credits |
| 2 | Aug 4 – 10 | Run the manipulation-check study (~$7). This is contribution 2 and the highest-risk result — get it early |
| 3 | Aug 11 – 17 | Benchmark arms (~$39). Leave-one-out retrain for per-role effects (§6). Second model family if week 2 held (~$39). Analysis via `failure_analysis.py` |
| 4 | Aug 18 – 29 | Write. §§3, 6, 7 can be drafted in week 1 — they need no results |

Sections 1, 2, 3, 6, 6a, and 7 do not depend on any experiment. Draft them while
the runs are queued.

**The schedule's binding constraint is a date nobody has yet.** NeurIPS's
suggested contribution deadline is 2026-08-29 and the mandatory notification date
is 2026-09-29, but the accepted-workshop list was not published as of
2026-07-28, and individual workshops routinely set deadlines earlier than the
suggested date. This schedule has **zero slack against Aug 29**, so an earlier
workshop deadline breaks it rather than compressing it.

Treat this as a standing watch item, not a week-1 task. When the list appears,
target venues where a methods or negative-results paper is in scope —
evaluation and benchmarking, multi-agent systems, or a "what went wrong" track —
rather than the largest general workshop. Fit matters more than prestige for a
paper whose headline is an instrument rather than a win.

If the real deadline lands before ~Aug 22, drop the second model family first
(§6, ~$39 and a week of wall-clock). It is a limited-generalization check, and
`PREREGISTRATION.md` A2.3 already reports H5 as not tested, so cutting it costs
no registered claim.
