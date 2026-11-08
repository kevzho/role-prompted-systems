# DRAFT — Amendment A4 (not yet registered)

**Status: DRAFT for author review, written 2026-09-29.** This text is not in
force until it is appended to `PREREGISTRATION.md` §11 with its registration
date, and the commit SHA that adds it is recorded. No A4 data may be collected
before then. All [DECIDE] items were settled by 2026-10-23. Remaining before registration:
the author's 18 rewordings (logistics/A4_paraphrases.md, checked with
cca_synth/prompts/check_paraphrases.py) and their hashes recorded in A4.3.

Target (revised 2026-10-23): ICSE 2027 AGENT workshop (paper 2026-11-27,
notification 2026-12-11, published by IEEE), and in parallel the AAAI-27
Workshop on Multi-Agent Collaboration (WMAC 2027; due 2026-11-20, non-archival)
-- the parallel submission only after AGENT's organizers confirm in writing that
concurrent submission to a non-archival workshop is allowed, and only if WMAC's
own policy permits it; both venues are told of the other and of the CJSJ
preliminary version. This is a separate paper from the CJSJ submission: its
contribution is the validity of the checks (A4.1-A4.7); the CJSJ results
appear only as cited motivation.

Motivation: the single review of the NeurIPS 2026 workshop
submission (rating 6, confidence 4) raised two concerns — **construct validity**
of the checks ("one of six passes" should not be read as "one of six implements
its operation") and **scope** (six specific roles from one council design).
A4 adds instruments aimed at those two concerns and nothing else.

---

**A4 — [REGISTRATION DATE]. Manipulation-check results HAVE been observed.**
All H6 results under A2/A3 are known to the author (one of six roles passes on
`gpt-5.6-terra`, two of six on `grok-4.3`; per-role *d* in
`paper/sections/99_appendix.tex`, Table `tab:raw`). Benchmark-arm and
leave-one-out results are also known. Every A4 design choice below was made
with that knowledge, and this amendment says so rather than presenting A4 as
blind.

What that does and does not compromise:

- **New data (A4.1–A4.5, A4.7) is confirmatory** for the questions A4 asks,
  because no A4 condition, wording, or model has been run. The author knows
  which roles passed before; A4 fixes in advance how every new result is
  scored, so that knowledge cannot steer the scoring.
- **Re-analysis of existing records (A4.6) is exploratory**, even though the
  statistic has never been computed, because the records it uses were
  collected and inspected under A2/A3.
- **No A2/A3 result is re-scored.** H6's original verdicts stand as reported.
  A4 can qualify how they are interpreted (A4.8); it cannot change them.

### A4.1 Positive and negative behavioural controls for every check

The reviewer's objection is that a failed check might mean an insensitive
instrument rather than an absent operation. A check that cannot detect the
operation when it is certainly present explains its own failures; A4 tests
that directly.

Two new prompt variants per role, both padded to within 10% of the operation
prompt's length and sharing `_OUTPUT_CONTRACT`:

- **`exemplar`** (positive control): the operation prompt plus one worked
  example in which the operation is visibly performed on an unrelated item.
  Example items are drawn from outside every benchmark split used here.
- **`anti`** (negative control): an instruction to *avoid* the operation
  (for example, "do not look for evidence against your answer"), otherwise
  identical to the operation prompt.

**Registered criterion (H8, instrument sensitivity).** A role's check is
*sensitive* if both hold, each with the A3.2 machinery (paired bootstrap,
2,000 resamples, BH across the six roles, q = 0.05):

- `exemplar` − `neutral` ≥ **d = 0.40** with interval excluding 0, **and**
- `neutral` − `anti` ≥ **d = 0.20** with interval excluding 0.

The asymmetric thresholds are deliberate: a neutral prompt already produces
some of each behaviour under the shared contract (A2.1's declared
compression), so there is less room to push it down than up.

**Consequence, fixed now.** A role whose check is *not* sensitive has its H6
verdict reported as **uninterpretable**, not as a failure — on both models, in
the text, the tables, and the abstract. This is the direct answer to the
review, and it applies whether the change helps or hurts the paper's headline.

A deterministic unit layer precedes the behavioural one: synthetic records in
which the operation is present or absent by construction, asserted in
`tests/` to move each check in the right direction. This tests the check
function, not the model, and is reported as a software test, not a result.

### A4.2 Blinded human validation (convergent validity)

**Sample.** 50 records per role (300 total), stratified equally across
`operation`, `neutral`, `persona`, `exemplar`, `anti` (10 each), drawn by
fixed seed from the `gpt-5.6-terra` dev-set records.

**Blinding.** Records are exported through `judging/blinding.py` with prompt
variant, role name, and check value removed and order randomised. The key is
stored separately and not opened until annotation is complete.

**Rating.** For each record, the annotator reads that role's one-sentence
operation definition and answers: *Was the operation performed?* (yes / no),
plus a 1–3 degree rating. The rubric is written and committed before the
export is generated.

**Annotator.** The author, as sole annotator (decided 2026-10-01). A second
independent annotator would strengthen this considerably, but
`ETHICS_COMPLIANCE.md` E1 requires an ethics determination before any other
person annotates; sole-author annotation is stated as a limitation. Intra-rater reliability: a random 20% of records re-annotated at
least seven days later under fresh randomisation, reported as Cohen's κ.

**Registered criterion (H9, convergent validity).** For each role, the AUROC
of the check value against the human yes/no label, with a 95% bootstrap
interval. Convergent validity is *supported* for a role if AUROC ≥ **0.70**
and its interval excludes 0.50. Reported for all six roles regardless of
outcome. A role failing H9 is reported with the same "uninterpretable" status
as a role failing H8.

### A4.3 Wording robustness (paraphrases)

**Design.** Three additional wordings of each role's operation prompt (four in
total, counting the registered one), each preserving the procedural steps,
length-matched within 10% to the original, and passing the existing clinical-
language and persona-vocabulary tests in `tests/test_prompts.py`.

**Authorship and lock.** (Decided 2026-10-23.) The author writes all 18
paraphrases before any A4 collection, and their SHA-256 hashes are recorded in
this amendment at registration. No paraphrase may be edited, replaced, or
dropped after any A4 data exists. Generating them with a model under test is
not permitted.

**Registered criterion (H10, wording robustness).** H6's rule is applied to
each wording separately. A role is **robust-pass** if it passes on ≥ 3 of 4
wordings, **robust-fail** if it fails on ≥ 3 of 4, and **wording-dependent**
otherwise. All three categories are reported per role per model, alongside
the per-wording *d* and interval. The original single-wording verdict is
reported next to it, so a reader can see whether the wording mattered.

### A4.4 A role-name condition

The registered `persona` variant is a generic description ("You are a careful,
skeptical checker."), not a role name. It does not test the practice most
multi-agent frameworks follow, which is naming a role. A4 adds:

- **`rolename`**: a one-line role name in the form "You are the {name}.",
  padded to length like `persona`, containing no procedural vocabulary.

Names (decided 2026-10-23), chosen from roles that appear in published
multi-agent frameworks, fixed at registration:

| Role | Proposed name |
| --- | --- |
| counter_evidence_checker | devil's advocate |
| consistency_checker | verifier |
| local_claim_checker | fact-checker |
| explore_then_prune | brainstormer |
| rare_hypothesis_checker | contrarian |
| confidence_calibrator | skeptic |

Reported descriptively against `neutral` and against `operation` with the same
statistics as H6. No pass/fail hypothesis is registered for `rolename`; its
purpose is to show whether a bare name moves the check at all.

### A4.4a A known cue in the control prompts, declared

Found 2026-09-29 while verifying the persona prompts against the run hashes
in `runs/checks/manipulation_report.json` (all six match
`prompts/__init__.py`). `_pad_to_length` pads the `neutral` and `persona`
variants with repeated copies of:

> "Take the time you need. Be accurate and specific. Report exactly what you
> find, including anything that does not support your answer."

The clause "including anything that does not support your answer" is a mild
instruction toward counter-evidence reporting, present in **both control
prompts** and absent from no control. It therefore biases the controls
*toward* the counter-evidence operation.

**Direction of the bias, stated.** It raises the neutral baseline for the
counter-evidence check, so it works *against* counter-evidence passing. The
A2/A3 counter-evidence pass (d = 1.03 and 0.52) cleared a harder baseline
than intended; the bias cannot have produced it. For the other five checks
the clause is not obviously related, but its effect is unmeasured.

**What A4 does about it.** All A4 conditions keep the registered padding
unchanged, so A4 results stay comparable to A2/A3. In addition, a
**padding-free neutral** (`neutral_plain`: the neutral body padded with
content-free filler — the fixed string "This line is padding and carries no
instruction." repeated) is run on the 200 dev items on both existing models,
and the H6 contrast is reported against both neutrals. If a role's verdict
differs between them, that is reported as padding-sensitivity, and the
padding-free contrast is treated as the cleaner estimate. The filler string
is fixed here and may not be changed after data exists.

### A4.5 A second scorable benchmark

**Change.** TruthfulQA's MC1 split (single correct answer), pinned and hashed
like the existing datasets, replaces the generation split for accuracy
analysis.

**This overrides a recorded design decision.** `benchmarks/truthfulqa.py`
rejected MC1 because "a council that deliberates over free text and then picks
from four options is not the system the paper describes." That objection is
weaker than it reads: every accuracy result in the paper already comes from
HaluEval, which is also a selection task. The generation split, meanwhile,
yielded zero scorable items (A2 results). A4 adopts MC1 and reports the change
of task format as a limitation (decided 2026-10-01).

**Implementation note.** MC1 items carry more options than the configured
`max_options = 6`. The default is to raise `max_options` to cover the largest
item, rather than subsample distractors, which would change item difficulty.
The value used is recorded in the config hash.

Arms: the three councils and the single-call baseline, as registered in A2,
on all 817 MC1 items, `gpt-5.6-terra` only. The contrasts, correction, and
thresholds are those of A2.

### A4.6 Cross-role matrix (discriminant validity) — exploratory on existing data

Every record already passes through all six checks. A4 computes the full 6×6
matrix: for each operation prompt *i*, the paired *d* on each check *j*
against the matching neutral records.

- **Diagonal** entries reproduce H6.
- **Off-diagonal** entries show whether a prompt moves checks other than its
  own. Discriminant validity for role *i* is *supported* if its diagonal
  entry exceeds the largest absolute off-diagonal entry in its row by at least
  **0.20**.

**Status.** Exploratory for the A2/A3 records on both existing models, because
those records have been observed. **Confirmatory** for the third model (A4.7)
and for A4's new wordings, whose records do not yet exist.

### A4.7 A third, open-weight model family

Llama 3.3 70B Instruct via Microsoft Foundry (decided 2026-10-01), subject to
availability on the existing resource; if unavailable, the nearest open-weight
instruct model of comparable size, chosen **before** any data is collected and
recorded here with its exact version. Decoding is unchanged: temperature 0,
1,024 output tokens, seed 1337. An open-weight model is chosen so the
manipulation-check results can be reproduced without a commercial API.

Run on the third model: all check conditions (`operation`, `neutral`,
`persona`, `rolename`, `exemplar`, `anti`, and the three new wordings) on the
same 200 dev items. Benchmark arms are **not** run on it for the AAAI version.

### A4.8 How A4 changes interpretation, fixed in advance

- H6 verdicts from A2/A3 are reported unchanged.
- Each is then **qualified** by H8 (sensitive?), H9 (validated?), and H10
  (robust to wording?). The paper's headline count of passing roles is stated
  **only** over roles whose checks are both sensitive and validated; the count
  of uninterpretable roles is stated in the same sentence.
- If fewer than three roles have interpretable checks, the paper reports that
  the instrument, not the roles, is the principal finding, and reframes
  accordingly. This branch is fixed now so it cannot be chosen after the fact.

### A4.9 Outcome-independent reporting commitment

A2.6 applies to A4 unchanged. In particular, if A4.1 or A4.2 show that some
checks are flawed, that is reported as a result, the affected roles are marked
uninterpretable, and the checks are **not** revised and re-run within this
submission. A revised check is a new instrument and requires its own
amendment.

### A4.10 Budget

At A2/A3 observed costs (≈ $7 per 3,600 check calls; ≈ $39 for three
benchmark arms on 1,500 items):

| Component | Calls (approx.) | Cost (approx.) |
| --- | --- | --- |
| New check conditions, 2 existing models (6 roles × 6 new prompts × 200 items × 2) | 14,400 | $30–60 |
| Third model, all 9 check conditions | 10,800 | $20–40 |
| MC1 benchmark arms (3 councils × 6 agents + single call, 817 items) | 15,500 | $30–50 |
| Padding-free neutral (A4.4a), 2 existing models | 2,400 | ≈ $5 |
| **Total** | **≈ 43,100** | **≈ $85–155** |

Funded from the Microsoft for Startups credit (`RUN_PLAN_2026-08-14.md`).

### A4.11 What A4 does not change

No knob in §9 is altered: split fractions, split salt, bootstrap count,
FDR α, CI level, and abstention policy are unchanged. A2's hypotheses, A3's
correction, and every reported number stand. A4 adds conditions, a model, a
benchmark split, and three instrument-validation criteria (H8–H10); it removes
nothing.

---

## Not in A4 — deferred to the ACL Rolling Review amendment (A5)

Recorded here so the scope is explicit, not so it is registered:

- A mixed-effects model with random effects for item and wording, reporting
  the variance share attributable to wording versus role.
- A full dose-response ladder (A4.1's `anti` → `neutral` → `persona` →
  `rolename` → `operation` → `exemplar` already provides a partial one; A5
  would pre-register a monotonicity test over it).
- Common roles taken directly from published frameworks (critic, verifier,
  debate), with checks written for them.
- The cost-matched single-agent baseline registered in A2 and never collected.
- Enough items to test whether passing a check predicts an accuracy gain.
- Optionally, a representation-level check on the open-weight model.

## Related work to add (found 2026-09-29; re-run the full novelty search before submission)

- Bean et al., *Measuring What Matters* (arXiv:2511.04703) — construct validity in LLM benchmarks; A4.1–A4.2 follow its recommendations.
- Desai et al., *What AI Benchmarks Actually Measure* (arXiv:2609.08812) — convergent/discriminant validity for AI evaluation; framework for A4.6.
- *Persona Following Is Not Selective Control* (arXiv:2609.35036, posted 2026-09-28) — cross-attribute influence of persona prompts with a neutral-condition diagnostic; closest recent method, must be cited and distinguished.
- Sclar et al. 2023; Mizrahi et al. 2024 (*State of What Art?*) — prompt sensitivity and multi-prompt evaluation; A4.3 follows them.
- *Adversarial Stress Testing of Role-Playing Language Agents* (arXiv:2608.03166) — role-fidelity metrics; related, measures staying in character rather than performing an operation.
