# P11 validation record

What was fixed, what was tested, and — importantly — what could not be tested
and why. Written 2026-07-28.

> **Suite status: 266 passed, 1 skipped.** The single skip is the CUDA
> multi-GPU test, which declines to run on a host without two CUDA devices
> rather than passing vacuously (see §4).
>
> Reproduce with:
>
> ```bash
> export TMPDIR=<BUNDLE>/.tmp && mkdir -p "$TMPDIR"
> python3 -m pytest tests -q
> ```
>
> `TMPDIR` is redirected because the machine's data volume is full; see
> `logistics/HANDOVER.md` §0.

## 1. Training resume

**Was:** `Trainer.save_checkpoint` persisted optimizer, scheduler, and scaler
state, but `Trainer.load_best` restored weights only. There was no resume path
at all — `load_best` exists to put the selected model back before evaluation,
which is a different operation.

**Now:**

- `Trainer.resume_from(path, strict=True)` restores model, optimizer, scheduler,
  GradScaler, EMA shadow weights, `global_step`, `best_metric`, `best_epoch`,
  the early-stopping counter, the per-epoch history, and the coalition-sampling
  RNG state.
- `train()` starts from `self.start_epoch` rather than 0.
- `last.pt` is now written **every epoch**, not only at the end, and always
  carries optimizer state regardless of `train.save_optimizer_state`. A
  checkpoint that cannot resume is not doing the one job it has.
- Exception: the per-epoch write is skipped when `fixed_coalition` is set. The
  retrain sweep's 64 inner trainings share one `run_id`, overwrite each other,
  and are never resumed, so a full 39 MB optimiser checkpoint per epoch there
  costs roughly 7.5 GB of I/O across the sweep and buys nothing.
- Resuming across a changed config hash **raises** by default. Silently
  resuming into a different configuration produces a model matching neither.
- CLI: `--resume` (bare, uses `last.pt`) or `--resume PATH`, plus
  `--resume-ignore-config`.

**Tests:** `tests/test_statistics.py::test_resume_*` (4 tests). **Passing.**

One test had to be corrected rather than the code: the first draft resumed a
2-epoch checkpoint into a 4-epoch trainer and expected it to work. It does not,
and should not — `train.epochs` enters the config hash, and the LR schedule is a
function of `total_steps`, so a checkpoint written under a 1-epoch cosine curve
does not belong on a 3-epoch one without the operator saying so. The guard fired
correctly; the test's premise was wrong. Extending a schedule now requires
`--resume-ignore-config`, which is the intended ergonomics.

## 2. Dead `EvalConfig.coalition_metric`

**Was:** declared and validated but never read; scoring was hardwired to
`eval.abstain_credit`.

**Now:** wired up rather than deleted. `config.COALITION_METRICS` defines three
item-level characteristic functions, all oriented higher-is-better:

| Metric | What it isolates |
| --- | --- |
| `selective_accuracy` | The pre-registered default; abstention scores `abstain_credit` |
| `accuracy` | Ignores the abstention policy — separates answering well from declining to answer |
| `negative_brier` | A proper scoring rule; no hard decision at all |

`utils.coalition_item_scores` dispatches; `EvalResult.item_scores` takes the
metric; every call site in `engine.coalition_sweep` and `main.py` passes
`cfg.eval.coalition_metric`. Invalid values now fail at config validation.

Note: `coalition_metric` is **not** frozen in `PREREGISTRATION.md` §9 — only
`abstain_credit` is. Changing it is therefore a secondary/exploratory analysis,
not a registered one, and must be labelled as such.

**Tests:** 4 tests. **Passing**, including one asserting the three metrics are
genuinely different functions — a registry whose entries all compute the same
thing would pass a smoke test and mean nothing.

## 3. Dropped `CandidateRecord.retries`

**Was:** parsed by `dataset.py` and dropped.

**Now:** partially already fixed in an earlier session (retries reach the cost
tensor and `engine.cost_report`). This session completed it:

- `dataset.exclusion_ledger` walks the parsed corpus and reports missing outputs
  **by role and by declared failure flag**, retries by role, abstentions by
  role, and — the useful new number — `unexplained`: absent records carrying no
  `failure_flag`, i.e. calls the orchestrator dropped without recording why.
- Retries are counted from the parsed records, not recovered by inverting the
  log-compressed cost tensor, so the figure is exact.
- `REPORT.md` now carries a first-class **"Exclusions, failed calls, and
  retries"** section with a per-role table. This also satisfies P12's
  exclusions requirement.

**Verified before the disk filled:** ran end-to-end on 400 simulated items;
2400 expected calls, 75 missing (3.12%), broken down by role and flag, 123
retries. The section renders correctly.

**Tests:** 2 tests. **Passing.**

## 4. Untested distributed path — **partially validated, honestly**

**Was:** DDP written but only ever exercised single-process on MPS.

**Now:** `tests/test_distributed.py`:

| Path | Coverage |
| --- | --- |
| Two-rank Gloo process group over CPU | **PASSING.** Real `mp.spawn` run: gradient sync verified (parameter checksums identical across ranks to 1e-6), `reduce_scalar`, `all_gather_object`, evaluation **replicated not sharded** (identical item order on both ranks), only rank 0 is the main process |
| Autocast decision logic | **PASSING.** Every device × dtype × enabled combination, including the pre-Ampere bf16→fp16 fallback, via monkeypatch |
| float16 mask-sentinel overflow | **PASSING.** Numerically pinned for float32/float16/bfloat16 under `T = MIN_TEMPERATURE` — the CUDA-only bug recorded in `FINDINGS_EVIDENCE_PATHWAY.md` §4 |
| **NCCL, real bf16 autocast, GradScaler overflow, TF32** | **NOT COVERED — hardware-gated** |

**The gradient-sync result is the substantive one.** DDP had never actually been
run multi-process; it is now confirmed that two ranks hold identical parameters
after training, and that the evaluation split is replicated rather than sharded.
The latter is the property the paired item-level statistics depend on — a
`DistributedSampler` on the eval loaders would pad the final shard and silently
misalign the bootstrap.

One test bug was found and fixed on the way: the item-ordering assertion used the
builtin `hash()`, which is salted per process (`PYTHONHASHSEED`). Across two
spawned ranks it can never agree, so the assertion was testing the salt rather
than the ordering. It now uses a SHA-256 digest.

`test_cuda_multi_gpu_smoke` **skips** on this host rather than passing
vacuously, with a message naming the command to run on a CUDA box:

```
torchrun --nproc_per_node=2 -m pytest tests/test_distributed.py
```

**This item is not closed.** P11 asked for multi-GPU CUDA validation including
bf16 autocast, GradScaler under float16, and TF32. None of that is reachable on
an Apple Silicon machine. It requires a CUDA host with ≥2 GPUs and is listed in
the handover as a hardware-gated task. Claiming it validated would be exactly
the kind of overstatement the rest of this project exists to avoid.

## 5. Unvalidated `shapley_mode="retrain"` — **it was broken**

**Was:** implemented, never run end to end.

**Executed, and it failed.** The first run produced **all-zero Shapley values**:
one distinct coalition value across all 64, v(A) = v(∅) = 0, every φ exactly
0.0000 with CI [0, 0]. The masked run on the identical config gave 30 distinct
coalition values and v(A) = 0.3314.

**Cause.** The abstention threshold and logit temperature were fitted once on
the main coalition-dropout model and reused for all 64 independently retrained
models, which have entirely different selection-score scales. The inherited
threshold sat above every score, everything abstained, and every coalition value
collapsed to `abstain_credit = 0.0`.

The efficiency check passed *vacuously* — `0 == 0 − 0` — which is why nothing
raised. An identity check cannot detect a degenerate solution.

**Fixed** by calibrating each coalition's model individually inside the loop
(own temperature, own threshold, both fitted on the calibration split only).
Full write-up in `logistics/FINDINGS_EVIDENCE_PATHWAY.md` §8.

**Why this was the most valuable thing in P11.** `PREREGISTRATION.md` §7 names
`retrain` as the mode for the **camera-ready ablation number**. Running the
registered plan without this fix would have produced a headline table stating
that no role contributes anything — symmetric, efficiency-checked, and entirely
an artifact.

Wall clock: ~6 minutes for 64 trainings at these settings (3 epochs, 800
simulated items) on Apple Silicon CPU/MPS, plus per-coalition calibration.

### The masked-vs-retrain gap — and it is larger than H2's effect size

With the fix in place, both modes ran on identical configuration (3 epochs, 800
simulated items, 200 bootstrap resamples), differing only in
`eval.shapley_mode`. This isolates the masking approximation.

| role | masked | retrain | gap |
| --- | ---: | ---: | ---: |
| `counter_evidence_checker` | +0.0618 | +0.0356 | **−0.0262** |
| `consistency_checker` | +0.0381 | +0.0510 | +0.0129 |
| `local_claim_checker` | +0.0566 | +0.0711 | +0.0145 |
| `explore_then_prune` | +0.0836 | +0.0607 | −0.0230 |
| `rare_hypothesis_checker` | +0.0556 | +0.0560 | +0.0004 |
| `confidence_calibrator` | +0.0356 | +0.0570 | +0.0214 |

**mean |gap| = 0.0164 · max |gap| = 0.0262**

Two consequences, both of which belong in the methods section:

1. **The gap exceeds H2's SESOI.** `PREREGISTRATION.md` §2 sets the smallest
   effect size of interest for a per-role contribution at **+0.01**. The masking
   approximation moves individual Shapley values by 0.016 on average and 0.026
   at worst — larger than the effect H2 is trying to detect. **The masked mode
   therefore cannot support per-role conclusions at the registered effect
   size.** It remains fine for iteration and for the full-council contrast.

2. **The approximation does not preserve rank order.** Masked ranks
   `explore_then_prune` first and `counter_evidence_checker` second; retrain
   ranks `local_claim_checker` first and `counter_evidence_checker` **last**.
   Any statement of the form "role X contributes most" is an artifact of the
   mode unless it is computed under `retrain`.

This vindicates §7's choice of `retrain` for the camera-ready number, and gives
that choice a quantitative justification it previously lacked.

Both modes agree exactly on the endpoints — v(∅) = 0.0000 and v(A) = 0.3314 —
so the disagreement is entirely in how the *interior* coalitions are valued,
which is precisely where the masking approximation operates. Retrain resolves 36
distinct coalition values against masked's 30.

*Simulated data.* This bounds the approximation error in this pipeline; it is
not evidence that the same magnitude holds on real records. The pilot should
re-measure it.

*Note:* `runs/p11retrain2/ablation.json` predates the degeneracy check by a few
minutes and so carries no `degeneracy_check` key. Any run from here on does; the
check was verified rendering in `runs/degencheck/REPORT.md`
(`27 distinct coalition values — PASS`).

Regenerate the comparison with:

```bash
python3 -c "
import json
r=json.load(open('runs/p11retrain2/ablation.json'))['shapley']
m=json.load(open('runs/p11masked/ablation.json'))['shapley']
for a,b in zip(m['roles'], r['roles']):
    print('%-26s masked %+.4f  retrain %+.4f  gap %+.4f'
          % (a['role'], a['shapley'], b['shapley'], b['shapley']-a['shapley']))
"
```

Note this gap is a property of *simulated* data. It bounds how much the masking
approximation distorts Shapley values in this pipeline; it is not evidence that
the same distortion size holds on real records, and the pilot should re-measure
it before the camera-ready number is taken from the masked mode.

## Summary

| Item | Status |
| --- | --- |
| 1. Training resume | **Done** — implemented, 4 tests passing |
| 2. Dead `coalition_metric` | **Done** — wired up, 4 tests passing |
| 3. Dropped `retries` | **Done** — exclusion ledger in `REPORT.md`, verified on an 800-item run |
| 4. DDP validation | **Gloo/CPU done** (gradient sync confirmed across two ranks); **CUDA multi-GPU open — hardware-gated** |
| 5. Retrain-mode Shapley | **Executed** — see §5 above for the masked-vs-retrain gap |

Suite: **266 passed, 1 skipped**, the skip being the CUDA test declining to run
rather than passing vacuously.

## P13 harness: dress rehearsal passed

The pilot harness was exercised end to end at full pilot scale (200 items)
against the stub provider, to prove it works before any money is spent:

```
records written: 200   provenance: simulated
$ python3 main.py all --records runs/rehearsal_records.jsonl --strict-real
ValueError: eval.strict_real=True but data.provenance is 'simulated';
refusing to produce a report that could be mistaken for an empirical result
```

Both halves are correct. The orchestrator emitted 200 schema-valid records and
stamped them `simulated` because a `StubProvider` produced them; the pipeline
then refused to build a results table from them.

**Before the §7 provenance fix this same sequence produced a `REPORT.md`
stamped `provenance: real`.** The rehearsal is therefore both a harness check
and a regression test for the most serious defect found this session.

Swapping `StubProvider` for a real provider is the only remaining difference
between this and P13. That swap needs authorised spend and the E23 provider
disclosure, neither of which is an engineering task.

## Bugs found on the way

Three, all now fixed and covered:

1. **`contamination.shingles` collapsed short text to a single shingle.** A
   12-token question and its 13-token near-twin each became one whole-text
   shingle; those differ, so their Jaccard similarity was 0 and the pair was
   reported as unrelated. Benchmark questions are routinely under 13 tokens, so
   this was the common case, not an edge case — it would have silently
   under-reported contamination on exactly the items most likely to be
   memorised. Shingles are now emitted at two widths and unioned.
2. **Eleven tests hardcoded `/tmp/cca_t_*.jsonl`.** Independent of any disk
   problem, two concurrent suite runs shared those paths and silently overwrote
   each other's fixtures. Now a per-run scratch directory under the repository.
3. **The DDP item-ordering assertion used the builtin `hash()`**, which is
   salted per process. Across two spawned ranks it can never agree, so the
   assertion was testing `PYTHONHASHSEED` rather than the ordering. Now SHA-256.
