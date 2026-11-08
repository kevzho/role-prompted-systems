# Bibliography: corrections, per-role citations, and verification backlog

Companion to `references.bib`. Applies every correction in
`REFERENCE_CORRECTIONS.md` and every citation requirement raised by
`NOVELTY_SEARCH.md`.

Written 2026-07-28; verification backlog (§3) cleared 2026-08-02.
The manuscript exists only as `disorders_ml_ai.pdf` with no
LaTeX source in this repository, so the corrections could not be applied to the
manuscript directly. `references.bib` is a drop-in replacement bibliography;
this document is the change list to apply alongside it.

---

## 1. Change list against the PDF's bibliography

| Ref | Change | Reason |
| --- | --- | --- |
| [9] | Venue corrected: NeurIPS 2024 → **First Workshop on LLM Persona Modeling, NeurIPS 2025** | Wrong venue and year |
| [9] | Restate the 95.6% GPI reduction as **specific to that paper's corporate-decision simulation** | Currently generalised beyond its setting |
| [10] | Full author list and record added (`wu2026council`, arXiv:2604.02923) | Was incomplete |
| [11] | Full author list and record added (`bause2026mixtures`, arXiv:2605.25929) | Was incomplete |
| [12] | Full record added (`huang2025valuediversity`); cite **benefit and trade-off together** | Was incomplete; benefit cited without the diminishing-returns and instability findings |
| [14] | Attribution corrected: "M. Jepma et al." → **Addicott et al. 2021**, *Neuropsychopharmacology* 46(3):614–621 | Misattributed |
| [14] | Restate as 26 adults with ADHD vs 23 controls in a six-armed bandit task; **not** a universal shifted explore/exploit ratio | Overgeneralised |
| [14] | **Either remove the introductory ADHD mapping or explain why no ADHD-derived role appears in the council** | The council has six roles, none ADHD-derived |
| [18] | Replaced: Dunning → **Clance & Imes 1978** | Dunning–Kruger does not establish impostor phenomenon |
| [18] | Stop equating impostor phenomenon with Dunning–Kruger; prefer "impostor phenomenon" over "impostor syndrome" | Two distinct constructs |
| [20] | Use the actual title: *Emergence of psychopathological computations in large language models* (Lee et al. 2025) | Was described only as "synthetic psychopathology", losing scope and caveats |
| — | **17 new entries added** for the novelty search | See §4 |

---

## 2. Two citations per role

`REFERENCE_CORRECTIONS.md` requires, for each role, (1) a current review or
computational account of the human construct and (2) an AI/ML source that
directly motivates the **implemented operation**.

The second column is the one that was missing entirely. Every link is
**inspiration**, and the claim-language column states the strongest sentence the
pair supports — nothing narrower may be written until the manipulation check
(P2) and benchmark evidence (P13/P14) exist.

| Role | Human construct | AI/ML source motivating the operation | Strongest supportable sentence |
| --- | --- | --- | --- |
| `counter_evidence_checker` | `kapur2003psychosis` (aberrant salience) | `selfcorrection2026illusion` — models catch errors in others' output that they miss in their own | "Requiring an explicit counterclaim per material claim is inspired by accounts of heightened threat-evidence weighting, and is motivated by evidence that models evaluate external claims more reliably than their own." |
| `consistency_checker` | `adams2016computational` (computational psychiatry; checking behaviour) | `stechly2024selfverification`, `huang2024selfcorrect` — self-critique fails without a sound external verifier | "Fixed independent invariant checks are inspired by accounts of repeated verification, and are motivated by evidence that unstructured self-critique degrades reasoning while structured external checks improve it." |
| `local_claim_checker` | `frith2003autism` (weak central coherence) | `cascade2026hallucination` — error propagation across agents | "Atomic-claim decomposition before global reconciliation is inspired by weak-central-coherence accounts, and is motivated by evidence that unchecked local errors propagate through multi-agent systems." |
| `explore_then_prune` | `gruber2011bipolar` (positive emotion and reward disturbance) | `crossmodel2026disagreement` — cross-model disagreement as an epistemic-uncertainty signal | "Generate-then-score is inspired by accounts of reward-driven exploration, and is motivated by evidence that disagreement across candidates carries usable uncertainty information." |
| `rare_hypothesis_checker` | `kapur2003psychosis` (salience of the non-canonical) | `deliberative2026illusion` — stance homogenisation suppresses minority positions | "Requiring one evidence-gated non-canonical alternative is inspired by accounts of altered salience, and is motivated by evidence that deliberation homogenises stances and loses minority hypotheses." |
| `confidence_calibrator` | `clance1978impostor` (impostor phenomenon) | `parrot2025`, `syceval2025` — sycophancy taxonomies measuring both capitulation and stubbornness | "Listing disconfirming conditions before emitting a probability is inspired by accounts of difficulty internalising achievement, and is motivated by evidence that models capitulate under user pressure in measurable, taxonomised ways." |

**Note on `addicott2021adhd`.** It is retained in `references.bib` because the
manuscript's introduction currently cites it, but **no role derives from it**.
Per the change list, either the introductory mapping is removed or the paper
explains the exclusion. It cannot stay as an unexplained mapping.

**Required disclaimer**, per `ETHICS_AND_DATA_GOVERNANCE.md`, wherever this
table or any part of it appears:

> These roles are functional AI processing constraints. They are not diagnostic
> models, simulations of lived experience, or claims that mental health
> conditions confer a single cognitive mechanism.

---

## 3. Verification backlog — CLEARED 2026-08-02

`references.bib` marks every entry with its verification level. The 22 entries
that were `UNVERIFIED` (4) or `verified-search` (18) on 2026-07-28 were opened
against the arXiv abstract page or the publisher record on **2026-08-02** and
are now **`verified-full`**. No entry failed verification: every arXiv ID
resolved, and every title matched the claimed subject.

Submitting with a fabricated or mis-attributed author list is exactly the
failure `REFERENCE_CORRECTIONS.md` was written to catch, and reference [14]
shows it had already happened once.

### 3.1 Corrections made during verification

Author lists were filled in verbatim for all 18 AI/ML entries; volume, issue,
pages, DOI, and edition were filled in for the 4 clinical sources. Beyond
those, the following substantive corrections were required:

| Entry | Correction |
| --- | --- |
| `persona2024doubleedged` | **Title was the superseded v1 title.** Current record (v2, 19 Oct 2024): *Persona is a Double-edged Sword: **Mitigating the Negative Impact of** Role-playing Prompts in Zero-shot Reasoning Tasks* — not "Enhancing the Zero-shot Reasoning by Ensembling…". Any sentence citing it as evidence that role-play ensembling *improves* reasoning must be re-read against v2. |
| `ong2024routellm` | Venue added: **ICLR 2025**, not a 2024 preprint. Published title reads "**from** Preference Data"; the arXiv preprint reads "with". |
| `huang2024selfcorrect` | Now cited as **ICLR 2024** (was a 2023 arXiv preprint entry). |
| `syceval2025` | Now cited as **AIES 2025** (was an arXiv preprint entry). |
| `zheng2023helpfulassistant` | Now cited as **Findings of EMNLP 2024** (was a 2023 arXiv preprint entry). Key retains `2023`. |
| `sumers2024coala` | Now cited as **TMLR 2024** (v3 is the TMLR camera-ready); year corrected 2023 → 2024. |
| `parrot2025` | Title was truncated; the record carries the subtitle *— A Sycophancy Robustness Benchmark for LLMs*. |
| `wisemind2026` | Full record: *npj Digital Medicine* **9**, article 575 (2026), doi 10.1038/s41746-026-02559-9, ten authors. The arXiv preprint arXiv:2502.20689 has a different title and is **not** this record. |
| `frith2003autism` | Publisher is **Blackwell Publishing, Oxford**, 2nd edn 2003, ISBN 0-631-22901-9; 1st edn was Basil Blackwell 1989. |

### 3.2 Authorship weight — three sources are single-author preprints

Verification surfaced a fact the novelty argument depends on. Three of the
entries recruited to constrain this paper's claims are **unrefereed
single-author preprints**:

- `theaterofmind2026` — Wenlong Shang, sole author, arXiv cs.MA, no venue.
  This is the entry §4 relies on to refute "CCA is the first".
- `cofailure2026ceiling` — Josef Chen, sole author, arXiv cs.AI, no venue.
  This is the entry §4 relies on to set the empirical ceiling on *any*
  combination method, including CCA.
- `steering2026unreliability` — Joschka Braun, sole author; the record states
  it is a **Master's thesis** (University of Tübingen, 89 pages). Portions
  appeared at the ICLR 2025 Workshop on Foundation Models in the Wild
  (arXiv:2505.22637), which is the citable refereed fragment.

None of the three is disqualified, but none carries peer review, and the
manuscript should say so where it leans on them. `selfcorrection2026illusion`,
`cascade2026hallucination`, `deliberative2026illusion`,
`crossmodel2026disagreement`, `expertpersonas2026`, and
`steering2025whatcanwesteer` are multi-author but likewise carry no venue.

### 3.3 Numeric claims re-checked against the source

- `zheng2023helpfulassistant` — "162 roles, no improvement over a no-persona
  control" **confirmed** (162 roles, 6 interpersonal relationship types, 8
  expertise domains). The paper *also* reports that oracle per-question
  best-persona aggregation improves accuracy while automatic persona selection
  is no better than random. Cite both halves; the first alone overstates it.
- `expertpersonas2026` — "MMLU 68.0% with expert personas vs 71.6% baseline"
  **confirmed** (Figure 1b). Scope correction: 68.0% is the *best* persona
  variant ("the minimum persona suffering the least"), not a typical one; all
  variants damaged accuracy.

### 3.4 Still outstanding

Nothing in this backlog. The `verified-human` entries (§1) were checked on
2026-07-26 and are unchanged — including `wu2026council`'s reported HaluEval
and TruthfulQA figures, which were **not** re-opened in this pass and remain
on the 2026-07-26 human check.

---

## 4. Entries added because of the novelty search

These are not optional. Each removes a specific overclaim identified in
`NOVELTY_SEARCH.md` §5.

| Entry | Removes |
| --- | --- |
| `theaterofmind2026` | "CCA is the first" — functionally constrained specialised agents under a central hub already exist |
| `wisemind2026` | The implication that clinical theory has not previously produced a multi-agent role decomposition |
| `sumers2024coala` | The implication that cognitive-architecture language agents are a new programme |
| `cofailure2026ceiling` | An unstated optimistic prior for H1; belongs in the introduction beside the measured error correlation |
| `zheng2023helpfulassistant`, `expertpersonas2026`, `persona2024doubleedged` | The implication that "roles help" is this paper's finding; they establish the prior for H3 |
| `selfcorrection2026illusion` | Reliance on clinical analogy where a direct AI result supports the council premise better |
| `ong2024routellm` | The implication that a six-call council is the natural cost baseline |
| `steering2025whatcanwesteer`, `steering2026unreliability` | Unqualified V2 claims about activation-level intervention (L4) |
| `wu2026council` (already [10]) | The absence of a head-to-head comparator on the same benchmarks |

---

## 5. Claim-language rules carried into the manuscript

From `LIMITATIONS.md`. A citation licenses a sentence; it does not license a
stronger one.

- Every clinical citation supports **"this operation is inspired by"** and
  nothing beyond it.
- No citation in this bibliography supports **"we demonstrate"**, **"CCA
  reduces"**, **"empirically validating each IPA"**, or **"CCA is the first"**.
- `yu2025personality`'s 95.6% figure may be quoted **only** with its simulation
  scope attached.
- `huang2025valuediversity` must be cited with both its benefit and its
  diminishing-returns/instability finding, never the benefit alone.
- `addicott2021adhd` supports a narrow association in a six-armed bandit task
  with n=26/23, and nothing about LLM exploration.
