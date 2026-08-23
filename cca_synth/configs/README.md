# Configurations

`config_from_dict` raises on unknown keys, so these files carry no comment
fields. What each one is for lives here instead.

| File | Registered as | Use |
| --- | --- | --- |
| `default.json` | `PREREGISTRATION.md` §9, frozen configuration | The full seven-domain, ten-condition study. Still the plan of record; not the study currently being run |
| `scoped.json` | `PREREGISTRATION.md` §11 amendment **A2** (2026-07-28) | The scoped study actually being executed |

## What `scoped.json` encodes

A2 narrows the design and changes the primary outcome. The knobs that differ
from `default.json` are the executable half of that amendment.

| Knob | `default` | `scoped` | Why |
| --- | --- | --- | --- |
| `eval.shapley_mode` | `masked` | `retrain` | A1 forbids per-role claims from masked mode: the gap is 0.0164 mean / 0.0262 max against H2's SESOI of 0.010, and it does not preserve rank order |
| `eval.attribution_estimator` | `shapley` | `leave_one_out` | A2.2 registers leave-one-out as the scoped study's per-role estimator: 7 trainings instead of 64 under `retrain`. A different estimand, not an approximation — the artifact's caveats say which Shapley properties do not hold |
| `eval.strict_real` | `false` | `false` — **pass `--strict-real` on the CLI instead** | See below |
| `eval.loo_moves_threshold` | `0.05` | `0.05` | A2.2's "moves its benchmark" criterion. Explicit here because it is a registered decision rule |
| `data.max_claims` | `12` | `4` | Cost is ~85% output tokens; capping claims cuts output ~30%, a larger saving than any caching change |

### `strict_real` is a run-time flag, not a config field

Setting `"strict_real": true` in a config file does not work, and the failure is
the guardrail behaving correctly. `Config.validate()` runs before any records
are loaded, at which point `data.provenance` is still the default `"simulated"`,
so the run aborts with *"refusing to produce a report that could be mistaken for
an empirical result"*.

The fix is **not** to also set `data.provenance: "real"` in the file. That
string is supposed to be *derived* from the providers that produced the records
(`Provider.synthetic` → `Council.provenance` → `dataset.build_datasets`), and
asserting it by hand in a config is precisely the fabrication path the three-way
guarantee in the top-level `README.md` exists to close.

Pass the flag on the command line, where provenance has been established from
the records:

```bash
python main.py all --records records.jsonl --config configs/scoped.json \
    --strict-real --run-name scoped-pilot
```

Everything in `PREREGISTRATION.md` §9 is unchanged between the two files —
`abstain_credit`, `target_coverage`, `fdr_alpha`, `ci_level`, `n_bootstrap`,
`split_fractions`, `split_salt`, `select_metric`, `evidence_gate_floor`. A2.7
requires that, so scoped-study artifacts stay comparable to any later full run.

## What `scoped.json` does not do

- **It cannot test H1.** A2.5: at n=1,000 the detectable effect is ≈0.048 and
  H1's SESOI is 0.03. H1 is reported as *not tested*, never as a null.
- **It does not select benchmarks or arms.** Those are orchestrator-side
  (`--condition`, and the item files you pass). This config governs the
  synthesizer and the analysis.
- **It does switch the estimator.** `eval.attribution_estimator:
  "leave_one_out"` routes `main.py ablate` (and `all`) through
  `engine.leave_one_out_sweep` / `engine.leave_one_out_analysis` — 7 trainings
  under `retrain`, not 64 — which is what A2.2 registers. The resulting
  `ablation.json` carries `estimator: "leave_one_out"`, no Shapley table, no
  efficiency check (none applies to this estimand), and no interaction or
  necessity/sufficiency sections, because those need coalitions the sweep never
  evaluates. The agent-by-benchmark FDR family is still computed: its contrast
  reads only the seven evaluated coalitions.
