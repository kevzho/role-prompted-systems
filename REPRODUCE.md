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
logistics/            specification, pre-registration, ethics review, data card
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

`python3 -m pytest -q` runs the full suite offline: 337 passed, 1 skipped.

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
| `python3 analysis_cross_arm.py` | cross-arm accuracy contrasts |
| `python3 analysis_single_vs_council.py` | council against a single call |
| `python3 analysis_confidence.py` | verbalised confidence against correctness |
| `python3 make_paper_figures.py --out-dir figures/` | Figure 2 as it appears in the paper |

**`manipulation_check` exits non-zero on purpose.** The check runner is a gate:
it returns a failing status unless every role passes, so that a collection run
cannot quietly proceed past a failed check. On the shipped records it prints
`1/6 roles passed` and exits 1. That is the reported result, not an error.

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
contact address and repository URL are withheld from `logistics/` and marked
as such in place.
