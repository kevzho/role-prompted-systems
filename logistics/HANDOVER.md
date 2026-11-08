# Handover — 2026-07-28

State of the CCA study, what changed overnight, what is blocked, and an honest
assessment of the ICLR timeline.

---

## 0. Disk: diagnosed, worked around, still needs your attention

**`/System/Volumes/Data` is 100% full — 184 GiB used of 228 GiB, with 21 MiB
available.** The ~44 GiB gap between "used" and "capacity" is purgeable space
that macOS has not reclaimed.

What this is **not**: the local snapshots are `com.apple.os.update-*` (staged OS
update snapshots), not Time Machine backups. And `<TMP>` is
only 24 KB, so the tooling is not the cause. There are ~233 MB of stale,
closed `*.bin` files sitting directly in `/private/tmp` from 23:40 on Jul 27,
owned by your account — not mine to remove, so I left them.

**The work was unblocked without deleting anything**, two ways:

1. `TMPDIR` is pointed at the project volume, so Python's temp files land on
   `<EXTERNAL_VOLUME>` instead of the full volume.
2. Eleven pre-existing tests hardcoded `/tmp/cca_t_*.jsonl` paths and are now
   fixed to use a per-run scratch directory under the repo. That was a genuine
   defect regardless of disk: two concurrent suite runs shared those paths and
   silently overwrote each other's fixtures.

**The suite runs and passes.** To reproduce:

```bash
cd <PROJECT>/cca_synth
export TMPDIR=<PROJECT>/.tmp && mkdir -p "$TMPDIR"
python3 -m pytest tests -q
```

You should still reclaim space on the internal volume — a full data volume will
bite you elsewhere. Start with the stale `/private/tmp/*.bin` files and
`~/Library/Caches`.

---

## 1. What was completed

### Fully done and verified

| Prompt | Deliverable |
| --- | --- |
| **P16** Bibliography | `references.bib` (37 entries) + `BIBLIOGRAPHY.md`. Every `REFERENCE_CORRECTIONS.md` fix applied; two citations per role; per-entry verification status |
| **P17** Novelty search | `NOVELTY_SEARCH.md`. 10 query families, related-work matrix, verdict |
| **P20** Ethics compliance | `ETHICS_COMPLIANCE.md`. All 37 requirements worked as a checklist with exact status |
| **P8** | `ADVCONSISTENCY_DATACARD.md` + `cca_synth/advconsistency/` — construction spec **and** the working toolkit: balanced cell planning, minimal-pair construction, blinded annotation packets, agreement, adjudication, ambiguity-pilot go/no-go, and a freeze that raises on any §6 violation. Only the expert labelling itself is human-gated |
| **P1** amendment | `PREREGISTRATION.md` §11 A1 — dated, recorded *before* any test-split read, covering the three pipeline defects and registering the substitutability expectation |
| **P18** audit | `MANUSCRIPT_EDITS.md` — the PDF text was extractable after all (`pypdf`), so the rewrite is specified quote-by-quote: **8 blockers, 7 major, 6 minor**, each tied to the limitation requiring it. Applying them needs the LaTeX source; deciding them does not |

### Built and tested

All of the following ship with passing tests.

| Prompt | Deliverable |
| --- | --- |
| **P11** Code gaps | Training resume (optimiser/scheduler/scaler/EMA/step/RNG); `coalition_metric` wired up; exclusion ledger; DDP validated on a real two-rank Gloo group. See `cca_synth/P11_VALIDATION.md` |
| **P12** Statistics | Shapley interaction indices; repeated-trial schedule; **computed** necessary/sufficient verdicts; exclusions table in `REPORT.md`; degeneracy check on the coalition value function |
| **P7** Benchmarks | `cca_synth/benchmarks/` — 7 adapters, pinned versions, licences, harm metrics |
| **P9** Contamination | `cca_synth/contamination.py` — dedup, split-leakage, post-cutoff partition, hashed splits |
| **P10** Judging | `cca_synth/judging/` — blinding, order randomisation, rubrics, model-judge audit gate |
| **P19** Release | `cca_synth/release.py` — §10 checklist, environment lockfile, hashed bundle |
| **P20** (code half) | `cca_synth/orchestrator/safety.py` — kill switch (3 independent triggers), single-agent fallback, dissent preservation, degradation reporting. 15 tests |
| **P15** (machinery) | `cca_synth/failure_analysis.py` — `PREREGISTRATION.md` §3's six "unacceptable if" thresholds encoded, so a role's pass/fail is *computed*; off-target harm detection (L12); and the §8 stopping rule enforced, so a negative primary result cannot be quietly replaced by the unconstrained comparison. The prose still needs P14's numbers |
| Tests | ~150 new tests across `test_statistics`, `test_benchmarks`, `test_distributed`, `test_release`, `test_safety`, `test_advconsistency`, `test_failure_analysis`, `test_orchestrator`. Suite: **266 passed, 1 skipped** — up from 117 at session start |

### Three defects found by running code that had never been run

All three were silent. None crashed, none failed a test, and two of them
produced output that looked entirely reasonable. That is the argument for
executing every path before the real run rather than after.

**`shapley_mode="retrain"` returned all zeros — and the self-test certified it.**
This is the one with the most direct consequence for the paper.
`PREREGISTRATION.md` §7 names `retrain` as the mode for the **camera-ready
ablation number**. It had never been run. Running it produced one distinct
coalition value across all 64, v(A) = v(∅) = 0, and every role's φ exactly
0.0000 with CI [0, 0] — while the efficiency check passed, because `0 == 0 − 0`.
Cause: the abstention threshold and temperature were fitted once on the main
model and reused across 64 independently retrained models with different score
scales, so everything abstained. Fixed by calibrating each coalition's model
individually; a new **degeneracy check** now fails loudly where the efficiency
axiom cannot see a problem. Full write-up: `FINDINGS_EVIDENCE_PATHWAY.md` §8.

Had the registered plan been executed without this fix, the headline ablation
would have reported that no role contributes anything — a symmetric,
efficiency-checked table of zeros that reads like a finding and is an artifact.

With the fix, the corrected comparison delivers the methods number P11 asked
for: **masked vs retrain differ by 0.0164 on average and 0.0262 at worst, while
H2's per-role SESOI is 0.01.** The approximation error is larger than the effect
being sought, and it does not preserve rank order — masked ranks
`counter_evidence_checker` second, retrain ranks it last. Two practical
consequences: `retrain` is mandatory for any per-role claim (§7 already required
it, now with a reason), and no "role X contributes most" sentence may be written
from a masked run.

### Two further defects

**`--strict-real` did not exclude simulated data.** A `StubProvider` run emitted
records stamped `provenance: "real"`, and `build_datasets` inferred "real" from
the mere existence of a records file. So a fully fabricated run would have
produced a `REPORT.md` stamped as empirical, with no banner — contradicting what
`README.md` promises. Fixed in three places and pinned by four tests; see
`FINDINGS_EVIDENCE_PATHWAY.md` §7. Every claim the `/logistics` package makes
rests on that stamp meaning something.

**Contamination detection silently failed on short text.** `shingles()` collapsed
any text under 13 tokens to a single whole-text shingle, so a 12-token question
and its near-twin scored a Jaccard similarity of 0 and were reported as
unrelated. Benchmark questions are routinely under 13 tokens, so this was the
common case — it would have under-reported overlap on exactly the items most
likely to be memorised. Now shingled at two widths.

### One result worth pre-registering against

**All 15 agent pairs are substitutable** — negative Shapley interaction, every
interval excluding zero, mean |interaction| 0.096 against per-role Shapley
values of 0.036–0.084. The interactions are the same order as the main effects
and in places larger.

Every pair contributes *less together than separately*. That is limitation L7
seen through the value function rather than the error vectors, and it matches
the co-failure ceiling reported across 67 frontier models (arXiv:2606.27288).
It bounds H1 before the real run: the council's gain is capped well below the
sum of its parts, which makes the budget-matched comparison harder to win.

Simulated data, so treat it as a property of the simulator until the pilot says
otherwise — but the *magnitude* and the uniformity across all 15 pairs are not
things the simulator was built to produce. Registered as amendment A1 in
`PREREGISTRATION.md` §11, before any test-split result was observed.

### The three literature findings that matter most

**1. "CCA is the first" is dead.** `NOVELTY_SEARCH.md` §2 found *Theater of Mind*
(arXiv:2604.08206), a cognitive architecture coupling a central hub with "a
heterogeneous swarm of **functionally constrained agents**" — the same
architectural move, from Global Workspace Theory instead of psychopathology.
Plus *WiseMind* (npj Digital Medicine 2026), where a clinical theory already
generates a multi-agent role decomposition. The "categorical, not a matter of
degree" framing goes too: the neighbours differ by degree.

**2. There is a direct head-to-head comparator on your exact benchmarks.**
*Council Mode* (arXiv:2604.02923, already your ref [10]) reports −35.9% relative
hallucination on a 1,200-item HaluEval subset and +7.8 points TruthfulQA versus
the best single model. You must report against it or explain why not.

**3. The persona literature predicts your H3 result.** Zheng et al.
(arXiv:2311.10054) evaluated 162 personas and found no improvement;
arXiv:2603.18507 found expert personas *hurt* MMLU (68.0% vs 71.6%). So "roles
help" is not your finding, and H3 — gains survive label removal — becomes the
load-bearing hypothesis. Say so before running, not after.

Also worth citing prominently: **arXiv:2606.27288**, a co-failure ceiling across
67 frontier models. That is limitation L7 measured at scale, and it sets the
prior for H1. Conversely **arXiv:2606.05976** ("models correct others but not
themselves") is the strongest *supporting* citation for the council premise and
is currently uncited — it belongs in the introduction in place of weaker
clinical analogy.

---

## 2. What still needs a human

Ordered by lead time.

| # | Task | Why only you can do it | Lead time |
| --- | --- | --- | --- |
| H1 | **Free disk space** | Machine access | minutes |
| H2 | **Institutional ethics determination** (`ETHICS_COMPLIANCE.md` E1) | Gates *all* human-subject work: AdvConsistency annotation and creative-divergence rating | **2–6 weeks** |
| H3 | **Compensated lived-experience + clinical terminology review** (E5) | Recruitment and payment; cannot be substituted by careful writing | **3–5 weeks** |
| H4 | **Recruit AdvConsistency expert annotators** (P8 §3) | Recruitment, payment, blocked by H2 | **4–8 weeks** |
| H5 | **Recruit creative-divergence human raters** (P10) | Same, blocked by H2 | **3–6 weeks** |
| H6 | **Name data owner + accountable system owner** (E14/E33) | One sentence; blocks E15, E20, E21 | 1 day |
| H7 | **Provider/region/retention disclosure** (E23) | Must be settled *before the first real API call* | 2–3 days |
| H8 | **Verify 19 bibliography entries** (`BIBLIOGRAPHY.md` §3) | Author lists not retrieved; ref [14] shows this has already gone wrong once | ~90 min |
| H9 | **Approve API spend** for the pilot and full run | Money — Anthropic-priced estimates and a recommended allocation of the $500 credit in §2a. Note the credit cannot fund RQ5's second provider | your call |
| H10 | **CUDA host with ≥2 GPUs** (`P11_VALIDATION.md` §4) | Hardware; not reachable on Apple Silicon | 1 day once available |
| H11 | **Materialise + hash the benchmark datasets** | Every `DatasetPin.sha256` is empty; `verify_pin` refuses until filled | 1 day |
| H12 | **Manuscript source** | Only `disorders_ml_ai.pdf` exists — no LaTeX. The edits are fully specified in `MANUSCRIPT_EDITS.md`; what is missing is a file to apply them to. If none survives, that document doubles as the rebuild spec | 1 week if rebuilding |

**H7 deserves a flag.** The `OPENROUTER_API_KEY` in your environment means
requests are *brokered*. The ethics addendum requires stating which providers and
regions receive data and whether they may train on it — and with a broker that
question is one level deeper than the key suggests. Enumerate the upstream
provider per model.

---

## 2a. What the API spend actually looks like

I did **not** spend anything. Costed against **current Anthropic list pricing**,
because a $500 YC Startup School Anthropic credit is the available budget.

Assumptions: ~600 input (≈360 cacheable role system prompt + 240 item) and ~350
output tokens per call; ~38 calls per item across the ten §3 conditions
(conditions 8 and 9 reuse condition 5's records, so they add none). Prices are
list; the effective column applies prompt caching on the stable system prompt
**and** the Batch API's 50% discount.

**The structural fact: cost is ~85% output tokens.** A §7 record is ~350 output
against ~240 uncached input. Input caching is a minor lever; output length is
the major one.

| Model | List $/MTok | Effective per call (cached + batched) |
| --- | --- | --- |
| Haiku 4.5 | $1 / $5 | $0.0010 |
| Sonnet 5 | $2 / $10 *(intro, through 2026-08-31)* | $0.0020 |
| Sonnet 5 | $3 / $15 (after) | $0.0030 |
| Opus 5 | $5 / $25 | $0.0051 |

| Run | Haiku 4.5 | Sonnet 5 (intro) | Opus 5 |
| --- | --- | --- | --- |
| **P13 pilot** — 200 items, ~1,400 calls, unbatched (latency wanted) | **$3** | **$6** | **$14** |
| **P14 full** — 5 domains × 2,700 items, 513k calls, per model | **$520** | **$1,039** | **$2,598** |
| **P14 full**, all three Anthropic models | — | — | **$4,157** |

**A $500 credit therefore covers roughly 12% of the registered full run.**

### Two findings that change how to spend it

**The Anthropic credit cannot satisfy RQ5 no matter how it is spent.**
Specification §4 requires at least three model families from **at least two
providers**. Haiku, Sonnet, and Opus are three models from *one* provider. The
credit funds the within-Anthropic arms; the cross-provider family needs a second
budget (the OpenRouter key) regardless. Do not burn credit trying to buy RQ5.

**The cost-matched contrast falls out of the pricing almost exactly.** Six Haiku
calls cost ≈ $0.012; one Opus 5 call ≈ $0.010. So a **six-Haiku council costs
about what one Opus call costs** — which *is* `PREREGISTRATION.md` §5's
cost-matched regime, arriving for free. It is also the more interesting
question: does a council of cheap models beat one expensive model at equal
spend? A six-Opus council against a one-Opus control is both the expensive
experiment and the less surprising one.

### Recommended allocation of the $500

1. **Pilot on Sonnet 5 — ~$6.** §10.1 names three things it should check first.
2. **Primary arm: Haiku 4.5 council, full 2,700-item power across 5 domains —
   ~$520** (trim to ~2,600 items/domain to land under $500). Keeping H1 properly
   powered matters more than model count, since RQ5 needs another provider
   anyway.
3. **Cost-matched control: Opus 5 single-agent** on the same items — cheap,
   because it is 1 call per item, not 38.
4. **Cross-family replication: OpenRouter**, on separate budget.

Two levers if more room is needed. **Batch API is 50% off** and this study is
entirely offline — measure p50/p95 latency on a synchronous subsample instead of
the whole run. And because cost is output-dominated, **capping
`data.max_claims`** in the §7 record from 8 to 4 cuts output ~30%, a larger
saving than any caching change.

The pilot command, ready to run once you approve:

```bash
cd <PROJECT>/cca_synth
export TMPDIR=<PROJECT>/.tmp && mkdir -p "$TMPDIR"
# 1. orchestrator -> records JSONL   2. records -> full pipeline
python3 main.py all --records <orchestrator-output>.jsonl --strict-real \
  --run-name pilot-halueval
```

`--strict-real` will now correctly **refuse** anything the stub provider
produced (see §1), so a pilot that silently ran on fakes is no longer possible.

## 3. ICLR timeline — the honest version

**ICLR 2027: abstract Sept 19, 2026; paper Sept 24, 2026.** That is **7.5 weeks**
to abstract and **8.3 weeks** to full paper.

### The full protocol does not fit

`PREREGISTRATION.md` §6 registers ≥2700 test items per benchmark for H1 power.
The full P14 run is 10 conditions × 7 domains × ≥3 model families × 3 budget
regimes, plus a 64-coalition retrain-mode ablation and the §9 robustness battery.

The compute is affordable. **The humans are not.** Ethics determination (2–6
weeks) must complete *before* annotator recruitment (4–8 weeks) which must
complete before AdvConsistency exists at all — and only then can it be run,
analysed, and written up. That is 10–16 weeks of strictly serial human process
against 8 weeks of calendar. AdvConsistency and creative divergence cannot be in
an ICLR 2027 submission.

### What does fit: a five-domain confirmatory study

Drop the two human-judged domains. Run the full ten-condition protocol on
**HaluEval, MATH/AIME, TruthfulQA, Sycophancy, and Temporal QA**. TruthfulQA
still needs judging, but its judge can be a model judge *provided* the P10 audit
gate passes with ~100 human-audited items — which is one afternoon of your own
time, not an annotator recruitment.

This is a legitimate paper: five domains, ten arms, three model families, three
budget regimes, exact Shapley with intervals, interaction terms, contamination
reported as a first-class result, and a pre-registration that predates every
number. The two dropped domains become declared future work — and the
pre-registration's `advconsistency` and `creative_divergence` rows are reported
as **not run**, which is honest and costs nothing.

### Week-by-week

| Week | Dates | Work |
| --- | --- | --- |
| 0 | Jul 28–Aug 2 | **H1 disk.** Run the suite; fix what the ~50 new tests find. Finish retrain-mode validation (`P11_VALIDATION.md` §5). **Start H2 ethics and H3 terminology review today** — they run in the background all the way through. H6, H7. |
| 1 | Aug 3–9 | H11 materialise + hash the five datasets. Wire adapters into the orchestrator. H8 bibliography verification. Amend `PREREGISTRATION.md` §11 to scope to five domains — **dated, before any test result is observed.** |
| 2 | Aug 10–16 | **P13 pilot**: 200 real HaluEval items end to end. `PILOT_FINDINGS.md`. Expect the simulator's assumptions to be wrong — that is the output. Especially: real pairwise error correlation vs the 0.10 assumption, and whether abstention ever fires. |
| 3 | Aug 17–23 | Fix what the pilot breaks. P9 contamination pass on all five. Scale to 2700 items on one benchmark, one family. TruthfulQA judge audit (100 human-labelled items — yours). |
| 4–5 | Aug 24–Sep 6 | **P14 full run.** Ten conditions × five domains × three families × three budget regimes. Masked Shapley throughout; retrain mode for the camera-ready number on the primary contrast only. §9 robustness battery on real data. |
| 6 | Sep 7–13 | **P15 failure analysis.** Every agent on every benchmark with BH q-values. Regressions per role. Negative results written as findings. P19 release artifact. |
| 7 | Sep 14–19 | **P18 manuscript.** Requires H12 — if there is no LaTeX source, rebuild from scratch this week. **Abstract deadline Sep 19.** |
| 8 | Sep 20–24 | Final figures, tables, appendices, artifact links. **Paper deadline Sep 24.** |

### Risk assessment

| Risk | Probability | Mitigation |
| --- | --- | --- |
| No LaTeX source for the manuscript (H12) | **high** | Budget a full week. Decide in week 0, not week 7 |
| Pilot invalidates simulator assumptions badly enough to need model changes | **medium-high** | `FINDINGS_EVIDENCE_PATHWAY.md` §3 already flags that abstention never fires. Adversarial training is *explicitly excluded* from the registration (§10.2) — if you add it, it is a new declared condition, not a fix |
| H1 fails under cost-matching | **medium** | This is a result, not a failure. §8 stopping rule already commits you to reporting it. RouteLLM-class baselines make it a real possibility |
| Ethics determination slips past week 4 | medium | Does not affect the five-domain plan; only affects whether H3's terminology review lands before submission |
| Retrain-mode Shapley too slow at full scale | medium | Use masked for all arms, retrain for the primary contrast only. Report the gap |

**Recommendation:** commit to the five-domain scope in week 0 and amend the
pre-registration before any test-split read. If you would rather have the full
seven-domain study, target **ICML 2027 (deadline ~late Jan 2027)** — that gives
the human components the 16 weeks they actually need, and lets ICLR 2027 pass.

---

## 4. Do this in the next 24 hours

1. **Email your ethics board (H2) and start terminology-review recruitment
   (H3).** These are the only two items with multi-week lead times. Every other
   task on this list can be done in an afternoon; these cannot, and they gate
   the human half of the study. Do them first even though they feel least like
   progress.
2. **Decide five-domain vs seven-domain**, and amend `PREREGISTRATION.md` §11
   accordingly — dated, and **before** any test-split result is observed.
   Amending afterwards makes the study exploratory rather than confirmatory,
   which is the one mistake that cannot be undone later.
3. **Approve the pilot spend** (~$6 on Sonnet 5, §2a) and run P13. The harness is already
   proven: it was rehearsed end to end on 200 items against the stub provider,
   and `--strict-real` correctly *refused* the output because the stub is
   synthetic. Swapping in a real provider is the only remaining difference. The
   pre-registration names three things the pilot should check first; expect the
   simulator to be wrong about at least one.
4. Name the data owner (H6) — one line in `ETHICS_COMPLIANCE.md` — and settle
   provider/region disclosure (H7) before the pilot's first call.
5. Decide the venue: ICLR 2027 scoped to five domains, or ICML 2027 with the
   full seven.

Free disk space on the internal volume when convenient (§0). It is no longer
blocking, but a data volume at 21 MiB free will bite something else.

**Nothing has been committed.** `git status` in `cca_synth/` shows 13 modified
files (+1,213 / −50) and 11 new paths — `benchmarks/`, `judging/`,
`contamination.py`, `release.py`, `orchestrator/safety.py`, five test modules,
and `P11_VALIDATION.md` — plus the new documents in `logistics/`. I do not
commit or push without being asked.

The suite is green, so a single commit is defensible. But the three defects in
§1 are each worth their own commit: a fix that is findable in the history is
worth more than a tidy diff, and all three are the kind of thing a reviewer may
ask you to demonstrate.
