# AES Fall 2026 poster (48 × 36 in, landscape)

"Did the Role Fire? Auditing the Manipulation Checks Behind Role-Prompted
Multi-Agent Systems." Agent Evaluation Science, NYC, Session 1, Poster 23.

## Build

From the bundle root, with the environment from `REPRODUCE.md` active:

```bash
cd cca_synth
python3 analysis_cross_role.py --records runs/checks/check_records.jsonl --out runs/cross_role.json
python3 analysis_cross_role.py --records runs/checks-grok/check_records.jsonl --out runs/cross_role_grok.json
cd ..
python3 poster/make_poster_figures.py
cd poster && latexmk -xelatex poster.tex
```

Requirements:
- **XeLaTeX** with `fontspec`, `tcolorbox`, `tikz`, `tabularx` and `enumitem` (TeX Live or MacTeX).
- **Open Sans,** shipped in `fonts/` under the SIL Open Font License (`fonts/OFL.txt`).

The two `analysis_cross_role.py` commands are only needed if you want to
regenerate `runs/cross_role*.json`. The shipped copies are what the figures
read.

## Where every number comes from

| Poster section | Source |
|---|---|
| 2 Method, 4 Verdicts | `cca_synth/runs/checks{,-grok}/manipulation_report.json` |
| 3 The six checks (guard outcomes) | `cca_synth/runs/cross_role{,_grok}.json` → `guard_terms` (never-tested guards) and `guard_stress` (defeatable guards) |
| 4 Bare role names | `cross_role{,_grok}.json` → `persona_vs_neutral` |
| 5 Claim counters | `cross_role{,_grok}.json` → `claim_count`; the scatter re-scores `runs/checks/check_records.jsonl` |
| 5 Matrix, "1 of 12" | `cross_role{,_grok}.json` → `matrix`, `discriminant` |
| 6 Direction matters | `cross_role{,_grok}.json` → `signed_discriminant` |
| 6 Councils | `cca_synth/runs/single_vs_council.json` |

Sections 3, 5 and 6 are **exploratory**: they were computed on records already
observed under the registered analysis. The A4.6 margin (0.20) was fixed in
`logistics/A4_DRAFT.frozen-2026-10-25T141552Z.md` before the matrix was first
computed. The signed margin was added after the AES review.

## The AES mark

`assets/aes-mark.pdf` is not redistributed. The AES template's notices state that
the mark is provided for AES paper identity and carries no public licence. To print
the poster with it, copy `assets/aes-mark.pdf` from the AES LaTeX template
(https://github.com/agent-evalscience/AES-Latex-Template) into `poster/assets/`.
Without it, the poster builds the same way, minus the logo.
