# Reproducibility artifact

Everything reported in the paper is reproduced by the commands below **from the
records shipped here**. No API keys, no network access, and no model weights are
required. Collection (re-querying the models) does need credentials and is *not*
what these commands do.

## Layout

```
cca_synth/            code, prompts, checks, analyses
  prompts/            the six roles: operation / persona / neutral, and the six check functions
  benchmarks/         dataset adapters and the pinned-revision materialiser
  orchestrator/       collection (needs credentials; not required to reproduce)
  runs/               the collected records and the analysis outputs
  data/               the benchmark corpora, raw and materialised, with SHA-256 pins
logistics/            specification, pre-registration, ethics review, data card,
                      and the frozen draft amendment A4 (with its SHA-256)
extended_abstract/    the AES Fall 2026 extended abstract as submitted (source + PDF), with
                      the source of every number and corrections since submission
poster/               the AES Fall 2026 poster: figure script, LaTeX source, fonts, PDF
ENVIRONMENT.json      interpreter and 350 package versions, captured via importlib.metadata
LICENSE               MIT, with third-party attribution for the two corpora
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r cca_synth/requirements.txt
cd cca_synth
```

Verified on Python 3.13.9. Exact package versions are in `ENVIRONMENT.json`.

`python3 -m pytest -q` runs the full suite offline: 347 passed, 1 skipped.

## Reproducing the paper

Each command was run against this bundle with `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_KEY` and `OPENAI_API_KEY` unset.

| command | reproduces |
|---|---|
| `python3 -m prompts.manipulation_check --records runs/checks/check_records.jsonl` | the six per-role checks, §5.1 |
| `python3 analysis_check_robustness.py --records runs/checks/check_records.jsonl` | threshold sweep, verdict stability, component decomposition (Appendix J) |
| `python3 analysis_check_robustness.py --records runs/checks-grok/check_records.jsonl` | the same three on the second lineage |
| `python3 analysis_factor_decomposition.py --records runs/checks/check_records.jsonl` | Figure 2(b) |
| `python3 analysis_cross_family.py` | Figure 2(a), the cross-family contrast |
| `python3 analysis_cross_arm.py --out runs/cross_arm.json` | cross-arm accuracy contrasts |
| `python3 analysis_single_vs_council.py` | council against a single call |
| `python3 analysis_confidence.py` | verbalised confidence against correctness |
| `python3 make_paper_figures.py --out-dir figures/` | Figure 2 as it appears in the paper |
| `python3 analysis_cross_role.py --records runs/checks/check_records.jsonl --out runs/cross_role.json` | **exploratory** cross-role audit: the 6×6 matrix, the A4.6 discriminant margin and its signed variant, correlations with claim count, persona vs. neutral, guard-term values (AES extended abstract §4, poster §§3, 5–6) |
| `python3 analysis_cross_role.py --records runs/checks-grok/check_records.jsonl --out runs/cross_role_grok.json` | the same on the second lineage |
| `python3 analysis_refusal.py --out runs/refusal.json` | **exploratory** refusal rates by council arm (manuscript appendix "refusal") |
| `scripts/analyse_arms.sh` | the full per-arm pipeline (synthesiser, evaluation, leave-one-out attribution) that wrote `runs/arms-*/`; offline, minutes per arm |

**`runs/checks-grok/manipulation_report.json` carries two extra blocks.** It was
written by the collection pipeline (`orchestrator/manipulation.py`), which adds
the collection manifests (`collection`) and the variant join counts (`join`)
to the report. The check runner above reproduces every other field exactly.
The terra report was written by the runner alone and has neither block.

**`manipulation_check` exits non-zero on purpose.** The check runner is a gate:
it returns a failing status unless every role passes, so that a collection run
cannot quietly proceed past a failed check. On the shipped records it prints
`1/6 roles passed` and exits 1. That is the reported result, not an error.

## What reproduces exactly, and what does not

Checked on 2026-11-08 in a fresh Python 3.13.9 environment (Apple M-series, MPS):

- **Exact:**
  - every manipulation-check statistic (all 12 d, q-values and verdicts);
  - the whole cross-role audit and the refusal analysis;
  - every training-free estimator: majority vote, confidence-weighted vote,
    the direct answer, the six single agents, and the oracle;
  - the council-vs-single-call and cross-arm contrasts.
- **Same conclusion, different decimals:** leave-one-out attribution. Five of
  the six roles' intervals are identical. Explore-then-prune's moves from
  [−0.011, +0.023] to [−0.029, +0.028]. Every interval still contains zero.
- **Not reproduced:** the trained synthesiser (`cca_synthesizer` in
  `runs/arms-*/evaluation.json`). Retraining with `scripts/analyse_arms.sh`
  gives accuracy 0.793 and ECE 0.132 on the 184 test items, against 0.783 and
  0.261 shipped. The shipped evaluation was written on 2026-08-23, three days
  after training, and the weights are not shipped, so the exact model cannot be
  recovered. Training on MPS with mixed precision is not bitwise deterministic
  despite `"deterministic": true`. Treat synthesiser numbers as single-run
  estimates. No poster or extended-abstract claim depends on them.

## Collection scripts

`cca_synth/scripts/collect_checks.sh` and `cca_synth/scripts/collect_arms.sh`
record exactly how the shipped records were collected. Running them needs Azure
credentials and costs money, and nothing in this file requires it.

## The cross-role audit is exploratory

`analysis_cross_role.py` re-scores the shipped check records and makes no
calls. Everything it reports was computed **after** those records had been
collected and inspected, so none of it is confirmatory, and the registered
verdicts (1/6 and 2/6) are unchanged by it.

- **Fixed in advance:** the A4.6 margin (0.20) was fixed in draft amendment A4
  before the matrix was first computed. The draft is shipped as
  `logistics/A4_DRAFT.frozen-2026-10-25T141552Z.md`; verify it with
  `cd logistics && shasum -a 256 -c A4_DRAFT.FREEZE.sha256`. The timestamp is
  self-recorded, and A4 itself is not registered.
- **Added after review:** the signed margin was added in response to the AES
  review, as a sensitivity analysis.
- **Tests:** `tests/test_cross_role.py` pins every reported number to these
  records. One example: the matrix diagonal must equal the registered effects
  in `manipulation_report.json`.

## The poster

```bash
python3 ../poster/make_poster_figures.py      # from cca_synth/
cd ../poster && latexmk -xelatex poster.tex
```

See `poster/README.md` for what each section reads and for the AES mark, which
is not redistributed.

## Release checklist

```bash
python3 -c "import json,release; print(json.dumps(release.check_release('runs/arms-cca_council','.','../logistics'),indent=1))"
```

Reports 11/11 items present. `assemble_release()` in the same module rebuilds a
hashed bundle and refuses to pass on a run whose provenance is `simulated`.

## What is deliberately not shipped

- **Trained synthesiser weights** (`*.pt`, ~38 MB each). Nothing in the paper's
  analysis path reads them; the synthesiser results are in
  `runs/arms-*/evaluation.json`.
- **Superseded runs** — `_stale_2026-08-16/`, `rehearsal*`, `p11*`, `smoke`,
  pilots. Retained locally, omitted here as they support no reported result.
- **Model API access.** `gpt-5.6-terra` and `grok-4.3` were served through
  Microsoft Foundry; the deployment name and serving route are recorded in each
  run manifest and are asserted rather than derived, as §4.3 states.

The benchmark data *is* shipped, raw and materialised, under `data/`. If you
would rather rebuild it, `python3 -m benchmarks.materialize` re-downloads the
pinned revisions and re-verifies them against the SHA-256 in
`data/materialize_manifest.json`; the pins are `207f479c...` (HaluEval QA) and
`741b8276...` (TruthfulQA generation).

## Provenance

Records carry a `provenance` field. A provider that does not declare itself
produces records marked `simulated`, any synthetic provider taints the whole
record, and the analysis path refuses to run on simulated data. Every run
reproduced above is `real`.

Two edits were made to this bundle for double-blind review, and neither touches
a reported quantity. Absolute filesystem paths in `runs/*/run.log` and in the
`path` field of `runs/*/evaluation.json` were replaced with the literal
`<BUNDLE>`; no other field of any record was altered. The investigator's name,
contact address and repository URL were withheld from `logistics/`.

For the public release (November 2026), the names were restored in `LICENSE`
and `logistics/ETHICS_COMPLIANCE.md`, and the repository URL in
`logistics/PREREGISTRATION.md` §12. These replace placeholders only; no other
text of the registration was changed. The internal working notes shipped in
`logistics/` (`HANDOVER*.md`, `FINDINGS_EVIDENCE_PATHWAY.md`, `NOVELTY_SEARCH*.md`
and others) are included because shipped files cite them. Machine paths in them
are replaced with `<PROJECT>`, `<EXTERNAL_VOLUME>` and `<TMP>`.
