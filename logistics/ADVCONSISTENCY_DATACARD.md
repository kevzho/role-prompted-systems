# AdvConsistency — data card

**Status: NOT BUILT.** This card is written in advance so that construction is
executed against a fixed specification rather than documented after the fact.
Every section marked **[PENDING]** must be completed with real numbers before
`advconsistency` may appear in a results table. The adapter
(`cca_synth/benchmarks/advconsistency.py`) refuses to load while its pin reads
`UNBUILT`, so this cannot be forgotten.

Written 2026-07-28 against `RESEARCH_SPECIFICATION.md` §6.

---

## 1. The phenomenon, defined independently of any role

§6's first requirement is that the phenomenon be defined without reference to
the local-claim-checker role. Defining it as "what that role detects" would make
the benchmark circular and the role's measured advantage tautological.

> **A seeded local contradiction** is a single clause in a passage that
> contradicts a fact stated elsewhere in the same passage, where:
>
> 1. the contradiction is **decidable from the passage alone** by a reader who
>    has read it in full — no outside knowledge is required;
> 2. the contradicting clause is **locally plausible** — a reader shown only the
>    surrounding sentence would have no reason to doubt it;
> 3. exactly **one** clause is altered between the members of a pair, so the
>    contradiction has a single identifiable cause.

This definition mentions no agent, no prompt, and no model. It is reproduced
verbatim in `advconsistency.PHENOMENON_DEFINITION` so the code and the card
cannot drift.

**What is deliberately excluded.** Contradictions requiring world knowledge;
contradictions arising from ambiguous pronoun reference; passages where two
statements are in tension but both could be true; numeric inconsistencies that
require arithmetic beyond one step. Each was excluded because expert agreement
on them was expected to be low, and a benchmark whose labels annotators dispute
measures the annotation process rather than the model.

---

## 2. Construction protocol

### 2.1 Paired passages

Each item is one member of a **minimal pair**. Both members share a passage
identifier (`pair_id`); they differ in exactly one clause. One member contains
the seeded contradiction; the other is consistent. Both are evaluated, and the
per-pair difference is the unit of analysis, which removes passage difficulty as
a confound.

### 2.2 Positive controls

§6 requires passages where **global context legitimately resolves an apparent
local ambiguity**. Without them, a system that answers "contradiction" to
everything scores perfectly on the primary outcome. These carry
`is_positive_control: true`, and flagging one is a false alarm scored under
`over_flagging_rate`.

**Target: 25% of items are positive controls.** Chosen so that a
flag-everything strategy scores at most 0.75, well below any threshold that
would be reported as a result.

### 2.3 Balancing

Four factors are balanced by construction, not checked afterwards:

| Factor | Levels | Rationale |
| --- | --- | --- |
| Answer position | contradiction in first / middle / last third | Position bias is a named §5 harm outcome |
| Topic | 6 domains, equal counts | Prevents a topic-specific effect masquerading as a general one |
| Passage length | short (150–250w) / medium (250–450w) / long (450–800w) | Long-range integration is the predicted *harm* for `local_claim_checker` |
| Contradiction distance | adjacent sentence / same paragraph / across paragraphs | The distance axis is what separates local checking from global integration |

`cca_synth/advconsistency/balance.py` verifies the realised design and refuses
to freeze a split whose cells are unbalanced beyond a declared tolerance
(default 20% relative deviation per cell).

**The toolkit exists and is tested.** `cca_synth/advconsistency/` implements
everything around the expert judgement, so expert time is spent only on the one
thing experts are for:

| Step | Function | Gate it enforces |
| --- | --- | --- |
| Plan a balanced design *first* | `assign_cells` | Any prefix stays balanced, so an interrupted run is still usable |
| Build minimal pairs by clause substitution | `make_pair` | Refuses unless the clause occurs exactly once |
| Verify the realised design | `balance_report`, `control_report`, `length_report` | §2.3 factors, §2.2 control rate, honest length buckets |
| Allocate blinded annotation packets | `build_annotation_packets` | **No annotator ever sees both members of a pair**; ids hidden; ≥2 independent ratings per item |
| Measure agreement | `annotator_agreement` | Krippendorff's α; "ambiguous" excluded, not coerced |
| Resolve disagreements | `adjudicate` | Refers rather than guesses; discards counted |
| Decide whether to scale | `ambiguity_pilot_report` | §4's α ≥ 0.80 and discard ≤ 20% go/no-go |
| Freeze | `freeze_split` | **Raises** on unbalanced factors, missing controls, split pairs, or any item lacking an adjudicated expert label |

`freeze_split` reports every failure at once rather than stopping at the first,
so one revision pass fixes everything. It also publishes the
**label-vs-intent disagreement rate** — how often experts overruled the
constructor — because a high rate means the templates produce ambiguous items.

### 2.4 Long-range subset

Items where the contradicting fact sits more than one paragraph from the
contradicted fact carry `requires_long_range: true`. This subset is what
measures the trade-off `PREREGISTRATION.md` §3 predicts for
`local_claim_checker`: decomposing into atomic claims should *help* local
detection and *hurt* long-range integration. If both move the same way, the
predicted mechanism is not what is happening.

---

## 3. Annotation — **[PENDING]**

### 3.1 Who

**Required: domain experts, per §6.** Non-expert crowdworkers are not
sufficient: the passages are constructed to be locally plausible, which is
precisely the property that defeats a fast annotator.

- **Target: 3 independent expert annotators per item**, plus one adjudicator.
- Recruitment source: **[PENDING]**
- Expertise verified by: **[PENDING]**

### 3.2 Pay

**[PENDING — must be recorded before annotation begins.]**

`ETHICS_AND_DATA_GOVERNANCE.md` requires pay to be documented. Record: hourly
rate, currency, the estimated per-item time from the pilot, whether pay is
guaranteed hourly or per-item, and the platform fee if any. Pay must not be
below the local living wage for the annotator's jurisdiction.

### 3.3 Instructions

Annotators receive the §1 definition verbatim, the rubric in
`cca_synth/judging/rubrics.py` (`ADVCONSISTENCY_RUBRIC`), and worked examples of
each excluded category from §1. They are **not** told which passages are
positive controls, nor the pair structure — showing both members of a pair to
one annotator would make the task trivial.

Annotators may mark an item **"ambiguous"**. This is not a failure state: the
ambiguity pilot (§4) exists to remove such items, and forcing a binary judgement
would manufacture agreement.

### 3.4 Agreement — **[PENDING]**

| Statistic | Target | Realised |
| --- | --- | --- |
| Krippendorff's α (contradiction present) | ≥ 0.80 | **[PENDING]** |
| Krippendorff's α (clause identification) | ≥ 0.70 | **[PENDING]** |
| Items marked ambiguous by ≥1 annotator | reported | **[PENDING]** |
| Items requiring adjudication | reported | **[PENDING]** |

α is computed by `benchmarks.metrics.krippendorff_alpha`. **If α falls below
target, the benchmark is not released.** An α of 0.6 on a 2-class task means the
labels themselves are uncertain at roughly the size of the effect being measured.

### 3.5 Adjudication

Items where the three annotators do not agree unanimously go to a fourth
adjudicator who sees the annotations but not the annotator identities. The
adjudicator may mark an item **"discard"**, and discarded items are reported
with a count and reason distribution — not silently dropped.

---

## 4. Ambiguity pilot — **[PENDING]**

§6 requires piloting for ambiguity **before** freezing. Protocol:

1. Construct ~150 candidate items across all balance cells.
2. Annotate with the full expert protocol.
3. Compute per-cell agreement. Any cell below α = 0.70 has its construction
   template revised, not its items relabelled.
4. Discard items marked ambiguous by ≥1 annotator.
5. Report the pilot discard rate. **A discard rate above 20% means the
   construction templates are producing ambiguous items and the phenomenon
   definition needs narrowing before scaling.**

Pilot results: **[PENDING]**

---

## 5. Splits and the hidden test set — **[PENDING]**

| Split | Purpose | Public? | n |
| --- | --- | --- | --- |
| development | template iteration, prompt debugging | yes | **[PENDING]** |
| calibration | threshold and temperature fitting only | yes | **[PENDING]** |
| validation | model selection | yes | **[PENDING]** |
| **test (hidden)** | final reported numbers | **no** | **[PENDING]** |

**Both members of a pair always land in the same split.** Splitting a pair
across the boundary leaks the answer: an annotator or a model that has seen the
consistent member has seen almost all of the contradicting one.

The hidden test split is **not released**. Published instead: its SHA-256 over
sorted item ids (`contamination.hash_release_split`), its balance table, and
baseline results. §6 permits either a permanently non-public test set or a
periodically refreshed one; this card commits to **non-public with annual
refresh**, and the refresh date is recorded here when it happens.

Split hash: **[PENDING]**

---

## 6. Deduplication — **[PENDING]**

§6 requires dedup against public corpora **and model development prompts**.
Run `contamination.overlap_against_corpus` against, at minimum:

- [ ] the source texts the passages were adapted from
- [ ] any public NLI / contradiction-detection corpus (SNLI, MNLI, ANLI, DocNLI)
- [ ] this project's own development prompts and the `prompts/` directory
- [ ] `contamination.split_leakage_report` across all four splits

Overlap results: **[PENDING]**

**Honest limit, stated here rather than discovered later:** no model's actual
pretraining corpus is available to diff against. Every overlap rate is a lower
bound. Because the passages are *constructed* rather than sourced, the exposure
risk is lower than for a scraped benchmark — but "constructed" is not "unseen",
since the source material the passages adapt may itself be in pretraining data.

---

## 7. Licences and personal data

- **Constructed passages:** CC-BY-4.0.
- **Adapted source material:** licence recorded per item in
  `source_licence`. Any item whose source licence prohibits redistribution is
  excluded from the public splits.
- **Personal data: none.** Passages contain no identifiable individuals. Where a
  passage needs a person, a clearly fictional name from a fixed list is used.
  This is checked mechanically before release, not asserted.
- **Annotator data:** annotator identifiers are pseudonymous and are not
  published. Per-annotator agreement statistics are published in aggregate only.

---

## 8. Baseline results — **[PENDING]**

§6 requires publishing baselines with the data card, so that a later system's
number has something to be compared against.

| System | Accuracy | Positive-control accuracy | Long-range accuracy | Position bias |
| --- | --- | --- | --- | --- |
| Human expert (held-out annotator) | **[PENDING]** | | | |
| Always-"contradiction" | 0.75 by construction | 0.00 | — | 0.00 |
| Strongest single model, direct | **[PENDING]** | | | |
| Six-agent council | **[PENDING]** | | | |

The always-"contradiction" row is filled in already because it follows from the
25% positive-control design. It is the number any reported result must clear.

---

## 9. What this benchmark cannot support

Stated in advance, so it is not negotiated after seeing results.

1. **It is not a general reasoning benchmark.** It measures one narrow, precisely
   defined phenomenon.
2. **It cannot validate the `local_claim_checker` role's clinical inspiration.**
   A role that scores well here has performed a text operation; nothing about
   weak central coherence follows.
3. **Constructed passages are not naturally occurring text.** Performance here
   does not transfer automatically to real documents, whose contradictions are
   rarely single-clause and rarely decidable from the passage alone.
4. **The 25% positive-control rate is a design choice, not a base rate.** Nothing
   here estimates how often real documents contradict themselves.

---

## 10. Checklist before release

- [ ] Phenomenon definition frozen and matching `PHENOMENON_DEFINITION` (§1)
- [ ] Paired passages, one clause differing (§2.1)
- [ ] Positive controls present at ≥20% (§2.2)
- [ ] Balance verified across all four factors (§2.3)
- [ ] Expert annotators recruited, expertise documented (§3.1)
- [ ] Pay documented (§3.2)
- [ ] Annotator instructions published verbatim (§3.3)
- [ ] Krippendorff's α reported and above target (§3.4)
- [ ] Adjudication procedure run and discards reported (§3.5)
- [ ] Ambiguity pilot complete, discard rate below 20% (§4)
- [ ] Hidden test split frozen, hash published (§5)
- [ ] Dedup run against all four corpora (§6)
- [ ] Licences recorded per item; no personal data (§7)
- [ ] Baselines published (§8)
- [ ] Adapter pin updated from `UNBUILT` to the release revision + sha256
- [ ] Institutional ethics determination obtained for the annotation component
      (`ETHICS_COMPLIANCE.md` item E1)

---

## 11. What is already done, and what is left

**Done (2026-07-28):** this card, and the complete `cca_synth/advconsistency/`
toolkit with 30 passing tests. Every mechanical step is implemented and gated.

**Left, and it is entirely human:**

1. **Ethics determination** (E1) — gates everything below.
2. **Recruit and pay domain experts** (§3.1, §3.2).
3. **Author the passages.** `assign_cells` gives a balanced design to fill;
   somebody has to write ~150 pilot passages and their minimal-pair partners.
   This is the largest single block of human work in the benchmark.
4. **Run the ambiguity pilot** (§4) and act on its go/no-go.
5. **Annotate at scale**, adjudicate, freeze.

The toolkit will refuse to freeze a corpus that fails any §6 requirement, so the
failure mode to worry about is not producing an invalid benchmark — it is
running out of calendar. Item 3 is why `HANDOVER.md` §3 recommends dropping
AdvConsistency from an ICLR 2027 submission.
