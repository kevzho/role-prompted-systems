# `cca_synth` — a calibrated synthesizer over structured candidate records

A PyTorch implementation of the one component of the Cognitive Constraint
Architecture (CCA) proposal that is a *learning* problem: the **synthesizer**
of `RESEARCH_SPECIFICATION.md` §7, together with the calibration, Shapley
ablation, statistics, and governance apparatus of §8–§10.

> These roles are functional AI processing constraints. They are not diagnostic
> models, simulations of lived experience, or claims that mental health
> conditions confer a single cognitive mechanism.

## Current scope: the study being run is narrower than the study registered

`PREREGISTRATION.md` §§1–10 describe a seven-domain, ten-condition study. That
remains the plan of record. **Amendment A2 (§11, 2026-07-28) narrows what is
actually being executed**, and this repository is configured for the narrowed
version — see `configs/README.md` and `logistics/PAPER_REFRAME.md`.

| | Registered plan | Being run |
| --- | --- | --- |
| Primary outcome | H1: the council beats budget-matched controls | The **manipulation check** — did each role perform its declared operation, against a length-matched neutral control |
| Benchmarks | 7 | 2 (HaluEval, TruthfulQA) |
| Conditions | 10 | 4 (conditions 1, 5, 6, 7) |
| Per-role attribution | 64-coalition exact Shapley | **Leave-one-out**, 7 trainings (`engine.leave_one_out_analysis`) |
| Config | `configs/default.json` | `configs/scoped.json` |

H1, H2, and H5 are **not tested** under the scoped design and are reported as
*not run* rather than as null results — at n=1,000 the detectable effect is
≈0.048 against H1's SESOI of 0.03 (A2.5). The five out-of-scope benchmark
adapters in `benchmarks/` are kept, tested, and unused; AdvConsistency and
creative divergence additionally need an ethics determination and paid
annotators before they can run at all.

The rest of this document describes the full apparatus, which is unchanged and
still exercised by the test suite.

## What this repository is, and is not

The CCA concept paper describes a six-agent prompt-level council. Running that
council is an API-orchestration task, not a GPU task. What *is* a GPU task —
and what the specification leaves underspecified — is the operator that turns
six structured candidate records into one answer, a calibrated confidence, an
abstention decision, and an auditable claim-level decision record.

This repository implements that operator and everything needed to evaluate it
honestly:

- a permutation-invariant, coalition-masked synthesizer (`model.py`);
- post-hoc temperature and selective-prediction calibration fitted **only** on a
  held-out calibration split;
- the full `2**6 = 64` coalition sweep with exact Shapley values, item-level
  bootstrap intervals, and `P(φ > 0)`;
- the FDR-corrected agent × benchmark family that limitation L12 requires;
- pairwise error-correlation measurement (limitation L7);
- the §9 robustness battery, including injection delivered through another
  agent's output;
- an audit trail that structurally cannot contain chain-of-thought.

**It contains no empirical evidence about CCA.** `LIMITATIONS.md` L1 is explicit
that the source paper reports no completed experiments, and this repository
fabricates none. It ships a labelled *simulator* so the pipeline is executable
and testable today; every artifact produced from it is stamped
`"provenance": "simulated"`, every report carries a banner, and `--strict-real`
refuses to run at all on synthetic data.

That last guarantee is enforced in three places, because it was once true in
none of them (see `logistics/FINDINGS_EVIDENCE_PATHWAY.md` §7):

- `Provider.synthetic` defaults to **`True`**, so a provider that forgets to
  declare itself has its output stamped *simulated*. The fail-safe direction is
  deliberate — the cost is a refused run, whereas the opposite default silently
  admits fabricated records to a results table.
- `Council.provenance` is derived from its providers, and **any** synthetic
  provider taints the whole record. Five-sixths real is not real.
- `dataset.build_datasets` reads those stamps. A corpus is real only if every
  record says so; a missing stamp, a `"simulated"` stamp, or a mixture all yield
  `"simulated"` with a warning naming what was seen. Provenance is never
  inferred from "this was loaded from a file".

## Install and run

```bash
pip install -r requirements.txt

# End-to-end on simulated records: train → calibrate/evaluate → ablate → report
python main.py all --config configs/default.json --run-name demo

# The scoped study's primary outcome (A2.1): 6 roles x 3 variants x N items
python -m orchestrator.manipulation --items dev_items.jsonl \
  --out-dir runs/checks --provider anthropic --model <snapshot> \
  --limit 200 --strict-real

# The scoped study (PREREGISTRATION.md A2), on real records
python main.py all --records records.jsonl --config configs/scoped.json \
  --strict-real --run-name scoped-pilot

# Structural invariant tests (permutation invariance, Shapley axioms,
# BH correctness, audit-trail governance, simulator sanity)
python -m pytest tests -q

# Multi-GPU
torchrun --nproc_per_node=4 main.py train --config configs/default.json
```

Artifacts land in `runs/<run_id>/`: `manifest.json` (the §4 execution manifest),
`config.json`, `evaluation.json`, `ablation.json`, `audit_records.jsonl`, and
`REPORT.md`.

### Reproducing the arm study end to end

The four collected arms and every number in the paper's results section come
from this sequence. Collection needs real provider credentials; everything
after it is deterministic and runs offline from the records.

```bash
# 1. Collect the three council arms and the single-agent baseline.
#    The baseline is condition 1 and supplies v(empty).
for ARM in cca_council persona_only_council neutral_council; do
  python -m orchestrator.collect --items data/items.arms1500.jsonl \
    --out runs/arms/records.$ARM.jsonl --condition $ARM \
    --provider azure-openai --model <deployment> --max-workers 8 \
    --allow-large-paid-run
done
python -m orchestrator.collect --items data/items.arms1500.jsonl \
  --out runs/arms/records.direct_answer.jsonl --condition direct_answer \
  --provider azure-openai --model <deployment> --max-workers 8 \
  --allow-large-paid-run

# 2. Join the baseline into every council arm. Without this join
#    base_answer_option stays -1 and v(empty) silently degrades to
#    "no answer at all" rather than the registered null policy.
for ARM in cca_council persona_only_council neutral_council; do
  cp runs/arms/records.$ARM.jsonl runs/arms/records.$ARM.pre-merge.jsonl
  python -m orchestrator.conditions merge-base \
    --council runs/arms/records.$ARM.pre-merge.jsonl \
    --direct runs/arms/records.direct_answer.jsonl \
    --out runs/arms/records.$ARM.jsonl
done

# 3. Train, calibrate, evaluate, and ablate each arm (7 trainings per arm).
for ARM in cca_council persona_only_council neutral_council; do
  python main.py all --config runs/arms-$ARM/config.json \
    --records runs/arms/records.$ARM.jsonl --run-name arms-$ARM --strict-real
done

# 4. The three cross-arm analyses.
python analysis_cross_arm.py        --runs-dir runs --out runs/cross_arm.json
python analysis_single_vs_council.py --runs-dir runs --out runs/single_vs_council.json
python analysis_confidence.py        --runs-dir runs --out runs/confidence_analysis.json
```

Step 2 is not optional and is not idempotent — merge from the `.pre-merge`
copy, never from an already-merged file. Step 4's three scripts are independent
of each other and read only `runs/`.

`analysis_confidence.py` is **exploratory** and was not pre-registered; its
artifact carries that label and the paper reports it as an observation.

### Using real records

```bash
python main.py all --records path/to/records.jsonl --strict-real
```

Each JSONL line follows the §7 schema:

```json
{
  "item_id": "tqa-0001",
  "benchmark": "truthfulqa",
  "n_options": 4,
  "gold_option": 2,
  "base_answer_option": 0,
  "candidates": [{
    "role": "counter_evidence_checker",
    "present": true,
    "answer_option": 2,
    "abstain": false,
    "confidence": 0.71,
    "counterevidence": ["..."],
    "failure_flags": ["unsupported_claim"],
    "atomic_claims": [{
      "claim": "...",
      "evidence_status": "supported",
      "source_or_check": "named_check",
      "confidence": 0.83,
      "veridical": true
    }],
    "output_tokens": 412, "latency_ms": 1830, "cost_usd": 0.0021, "retries": 0
  }]
}
```

Missing role slots are materialised as `present=false` and reported as
exclusions rather than silently imputed. `veridical` is optional; where it is
absent the claim-adjudication loss is masked out for that claim.

## Design decisions that answer specific requirements

| Requirement in `/logistics` | How it is enforced |
| --- | --- |
| §7.1 synthesizer must not see condition names or clinical labels | `model.use_role_embeddings=False` deletes the only path role identity takes into the network. A test asserts it is then exactly label-invariant, and a control test asserts the role-aware model is *not* — so the ablation is not vacuous. |
| §7.3 agent confidence is a feature, not ground truth | An explicit Platt-scaling `ConfidenceRecalibrator` with a content-conditioned correction, trained against whether each agent was actually right. |
| §7.4 record the rule for every accepted/rejected claim | A closed 5-rule vocabulary and a discrete rule head; the declared policy lives in `engine.rule_targets_from_records` and is unit-tested. |
| §8 `v(∅)` must be predeclared | The always-unmasked `[BASE]` token carries the direct-answer baseline, so the empty coalition is *architecturally* the base model's answer. |
| §8 64 coalitions with intervals and `P(φ>0)` | Coalition-dropout training makes one model valid for every coalition; Shapley is exact (not sampled) at `R=6`, and bootstrap resamples *items* rather than re-running the model. Efficiency axiom verified to ~1e-16, **plus a degeneracy check** — efficiency alone passes vacuously on a collapsed value function (`0 == 0 − 0`), which once let an all-zero ablation certify itself. |
| §8 additivity must not be assumed | Exact Shapley–Owen pairwise interaction indices over all 64 coalitions, with bootstrap intervals and a complementary/substitutable/additive verdict per pair. The per-role table is never presented alone. |
| §8 "necessary"/"sufficient" require a rule, an interval, and a replication | `necessity_sufficiency_verdicts` *computes* the permitted wording from all three; `REPORT.md` renders it verbatim. The words cannot enter the artifact by prose. |
| §9 order and role-label permutations | No positional encoding over the agent axis; invariance is a theorem, checked numerically in both the test suite and the robustness report. |
| §9 synthesizer capture by one verbose agent | A hinge penalty on peak single-agent pooling mass, applied during training rather than only measured afterwards. |
| Ethics addendum: no chain-of-thought logging | `AuditRecordWriter` raises on any key outside the allowlist, any key matching the hidden-reasoning/credential denylist, and any *value* that is a clinical identifier. |
| L7 agent diversity may be superficial | Pairwise error correlation is measured and reported; the simulator deliberately induces correlated common-mode failure so the pipeline is not validated on unrealistically independent agents. |
| L10/L18 cost and latency | `cost_report` gives per-agent token share, latency, and dollars per item — the denominator any budget-matched comparison needs. |

## Honest limitations of this implementation

- **Leave-one-out is not a Shapley value.** The scoped study's per-role
  estimator (`engine.leave_one_out_analysis`, registered in A2.2) does not
  satisfy efficiency, charges every interaction to whichever role is dropped,
  and says nothing about the 57 coalitions it never evaluates. Under the
  substitutable pairs A1 recorded, that biases each contrast toward zero — so a
  role clearing the threshold has cleared a conservative bar, but a role failing
  it has not been shown to contribute nothing. The function reports all of this
  in its `caveats`, and reports no efficiency check, because none applies.
- `shapley_mode="masked"` reuses one coalition-dropout-trained synthesizer for
  all 64 coalitions. This approximates, and does not equal, retraining per
  coalition. `shapley_mode="retrain"` does the exact thing at 64× the cost, and
  the mode is recorded in every artifact. Each retrained coalition is calibrated
  **individually** on the calibration split; sharing a threshold across
  independently trained models collapses every coalition value to
  `abstain_credit` (see `logistics/FINDINGS_EVIDENCE_PATHWAY.md` §8).
- The synthesizer is deliberately **text-free**. It consumes structured records
  only. Answer options must be normalised into a shared option space upstream;
  optional precomputed option embeddings are supported but not required.
- The simulator's constants are modelling assumptions, stated in
  `dataset.py` and grid-searched against explicit realism targets (solo accuracy
  well above chance, error correlation ≈ 0.10, consensus as weak positive
  evidence). They are not measurements of any real system.
- Evaluation is replicated across ranks rather than sharded, because a
  `DistributedSampler` pads its final shard and would corrupt the paired
  item-level statistics.

## Claim language

Per `LIMITATIONS.md`, results from this repository support *we propose*, *we
hypothesize*, and *this operation is inspired by*. They do not support *we
demonstrate*, *CCA reduces*, or *CCA is the first*. Clinical identifiers appear
nowhere in the code, the tensors, or the audit trail; `config.CLINICAL_TRACEABILITY`
exists solely so readers of the original concept paper can locate the
corresponding section, and the audit writer actively rejects those strings as
values.
