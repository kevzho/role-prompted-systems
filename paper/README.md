# Full manuscript

*Did the Role Fire? Manipulation Checks for Role-Prompted Multi-Agent LLM Systems.*
Kevin Zhou, Sparsh Roy, Maxim Kudryashov.

**Status.** This is the version submitted to the NeurIPS 2026 workshop on
Evaluation of Interactive Agents (not accepted; one review). Two changes have
been made for public release:
- the author block is restored and the style is set to `preprint`;
- the anonymous mirror link is replaced with this repository.

Figure 2 is now generated from this repository's data (see Build). The text
itself is unchanged. **Several of its claims are superseded**; see the checklist
below. The shorter, corrected versions are the extended abstract
(`../extended_abstract/`) and the poster (`../poster/`).

## Build

```bash
cd ../cca_synth && python3 make_paper_figures.py   # writes ../paper/figures/checks_panel.pdf
cd ../paper && make                                # latexmk -pdf; output main.pdf
```

The build needs a TeX Live installation with `latexmk`. Figure 1 and the
classification table are drawn in TikZ and `tabular` inside the document. A
built `main.pdf` and the generated figure are included so the paper can be read
without building it.

## Revision checklist

Each item gives its source: **[A]** the AES review, **[E]** the EIA review,
**[K]** a known error found in the artifact audit, **[X]** our exploratory audit
(`cca_synth/analysis_cross_role.py`).

### Claims the audit overturns
- [X] **"Explore-then-prune reflects a genuine reversal"** (§5.1). The prompt
  records only the hypotheses that survive pruning, and the check counts
  recorded claims as first-pass breadth, so the drop is consistent with a
  mismatch between prompt and check. It is not evidence of less exploration.
- [X][A] **"Only one in six roles accomplishes its stated operation"** (§5.1,
  abstract). No role has a validated pass. The terra pass is an unguarded raw
  count. On grok, local-claim is 2 × claim count by construction.
- [X] **"The overall finding replicates"** (§5.3). The two sign changes across
  families (local-claim, rare-hypothesis) involve checks that track claim count
  (r = 0.79–1.00).
- [K] **"Bare role-name prompts show major reductions in refusal rates"**
  (abstract). `analysis_refusal.py` now reproduces it: refusals happen only on
  unscored TruthfulQA items, and persona is lower by 1.63 points vs. operation
  (q = 0.006) and 0.97 vs. neutral (q = 0.033). That is real, but "major"
  overstates it. The appendix also needs three numbers corrected: the operation
  arm's rate is 6.17% (printed "6.1%"); operation − neutral is +0.67 points
  [−0.57, +1.87], q = 0.28 (printed +0.63 [−0.57, +1.80], q = 0.31); persona −
  neutral is −0.97 points, q = 0.033 (printed −1.00, q = 0.035).

### What the AES reviewer asked for
- [A] An appendix with what each check computes, the five guards and what
  happened to each (`guard_terms`, `guard_stress`), and all 18 prompts
  (`cca_synth/prompts/__init__.py`).
- [A] Report local-claim as **redundant** with claim count, and separate
  structural entanglement between checks from cross-effects of prompts
  (`claim_count`, `matrix`).
- [A] Add the signed, direction-aware margin as a sensitivity analysis
  (`signed_discriminant`).
- [A] Add persona vs. neutral (`persona_vs_neutral`).
- [A] Report the setup: HaluEval QA yes/no task; temperature 0; seed 1337; one
  sample per item; d is paired **d_z**; councils use majority vote and run on
  `gpt-5.6-terra`. Give council intervals (`runs/single_vs_council.json`).
- [A] For each guard, say which reading applies: **defeatable**, or **not
  tested by this data**. The calibration guard is defeatable on terra only
  (ECE 0.144 → 0.112); no constant shift helps on grok.
- [A] Narrow "fundamental measurement problem" to the actual scope.
- [A] Add related work on judge-based persona adherence (verify citations
  first), and a convergent-validity plan.

### What the EIA reviewer asked for
- [E] Construct validity. The audit above answers it directly.
- [E] Scope: six roles from one council design. State this in the abstract
  and introduction.

### Known errors (artifact audit)
- [K] Compute paragraph (§4.3): about 35,700 calls issued, 35,208 usable;
  ≈17.66M input tokens; $0.001792 per call; 2.44 s mean latency. Steps and wall
  time differ per run.
- [K] Guard tally: five checks had guards, and **none survived**. "Six
  pre-registered guards" is wrong.
- [K] §5.3: **three of four** sign-change differences survive FDR, not four.
- [K] The units list omits `_check_alternative_generation` (a signed
  difference) and `_check_calibration` (negative ECE in [−1, 0]).
- [K] Model version: manifests carry `model_snapshot`, not a version field.
- [K] State the tie rule in the main text: under ties-excluded, two of the three
  council contrasts lose significance.
- [K] The "1,128 pilot calls" and the "1,134 / 1,500" and "366" join counts
  can't be verified from this repository. Cite them as collection-time
  observations, or drop them.
- [K] Say explicitly that the synthesiser ties majority vote, as the
  registration said it would report. **But the synthesiser numbers do not
  reproduce on retraining** (accuracy 0.783 → 0.793, ECE 0.261 → 0.132; see
  `REPRODUCE.md`). Report them over several seeds, with spread, or drop the
  synthesiser comparison. In particular, "worst ECE of any estimator" does not
  hold on retraining.
- [K] Two neutral prompts serve the six roles (`e470bafac566be18` for four,
  `ba6a153f4675917f` for consistency and local-claim).
- [K] Typos: "tranfer", "diagonstic", "d = −0.18).but".
