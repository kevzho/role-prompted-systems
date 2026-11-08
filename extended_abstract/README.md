# Extended abstract: AES Fall 2026 (NYC)

*Did the Role Fire? Auditing the Manipulation Checks Behind Role-Prompted
Multi-Agent Systems.* Kevin Zhou, Sparsh Roy, Maxim Kudryashov.
Agent Evaluation Science, Fall 2026. Submitted 2026-10-25 and accepted as a poster
(Session 1, Poster 23). The venue is non-archival.

- `main.tex`: the source **exactly as submitted**. Its comments still point to
  pre-release paths; the current sources are listed below.
- `Did_the_Role_Fire_AES2026_submitted.pdf`: the submitted PDF.

## Build

```bash
cp <AES-Latex-Template>/assets/aes-mark.pdf assets/   # not redistributed; see below
latexmk -xelatex main.tex
```

The build needs XeLaTeX. The style (`aes.sty`, `aes-author-year.bst`) comes from the
[AES LaTeX template](https://github.com/agent-evalscience/AES-Latex-Template),
which is MIT-licensed (`AES-TEMPLATE-LICENSE`). Open Sans is under the OFL
(`fonts/OFL.txt`), and the bibliography style under the LPPL (`licenses/`).

The template's notices say the AES mark carries no public licence, so
`assets/aes-mark.pdf` is not in this repository and `aes.sty` needs it to build.
Copy it from the template.

## Where every number comes from

All paths are relative to `cca_synth/`.

| Section | Numbers | Source |
|---|---|---|
| 2 Method | 200 items, 3,600 calls per family, no exclusions | `runs/checks{,-grok}/records.*.manifest.json` |
| 3 Verdicts | d = +1.03 / +0.52 / +0.69; 0.61 → 1.285 | `runs/checks{,-grok}/manipulation_report.json` |
| 3 Verdicts | 0.846 / 0.846 / 0.839 vs. 0.861 | `runs/single_vs_council.json` |
| 3 Verdicts | leave-one-out intervals contain zero | `runs/arms-cca_council/ablation.json` |
| 4 Audit | guard outcomes; 1,191 / 1,200 | `runs/cross_role{,_grok}.json` → `guard_terms`, `guard_stress` |
| 4 Audit, Table 1 | margins, "1 of 12", r with claim count | `runs/cross_role{,_grok}.json` → `discriminant`, `claim_count` |

`analysis_cross_role.py` regenerates `runs/cross_role*.json`, and
`tests/test_cross_role.py` pins every Table 1 value. Section 4 is **exploratory**.

## Corrections since submission

The submitted text is kept as the record. The points below are corrected in the
poster (`../poster/`) and are planned for the full manuscript.

| Submitted text | Correction |
|---|---|
| Two guards "were vulnerable, while three of them offered little difference in observed outputs" | Two are **defeatable**. Three **never varied** because the behaviour they guard against barely occurred: models self-reported a source on 99.9% / 99.2% of claims, only 1.7% / 0.5% of claims were "uncertain", and 1 / 37 supported claims had no source. These guards are untested by this data, not shown invalid. |
| The local-claim guard "was constant in 1,191/1,200 records" | Constant at exactly 2, so on this data the local-claim check **is 2 × claim count**: redundant by construction, not just "correlated". |
| The discriminant criterion as "increased that check more … than it increased any other check" | Own-check d minus the **largest absolute** d on any other check, at least 0.20. |
| "a key reversal effect is explained by a prompt–check misalignment" | The evidence is **consistent with** the misalignment: the prompt records only surviving hypotheses, and the check counts recorded claims as breadth. |
| "This is a fundamental measurement problem" | Too broad for six roles, one council design, one benchmark and two models. |
| Positive control as "told to … exhibit" the operation | The operation prompt **plus a worked example** of the operation. The negative control is an instruction to avoid it. |
| "Cohen's d" / "d" | Paired **d_z**: the mean of the paired differences divided by their SD. |
| Typesetting | Straight quotes, periods before `\citep`, `$0.79-1.00$` printed with a minus sign, "1000". |

**Added after the AES review:** a signed (direction-aware) margin, persona vs.
neutral on each role's own check, and council intervals. These are in
`runs/cross_role*.json` → `signed_discriminant` and `persona_vs_neutral`, and in
`runs/single_vs_council.json`.
