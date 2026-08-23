# Council orchestrator

Benchmark items in, structured §7 candidate records out. This is the component
everything else in the repository waits on: the synthesizer consumes records,
and until now nothing produced them.

## Use

```bash
# Collect (offline stub — no credentials, no spend)
python -m orchestrator.collect --items items.jsonl --out records.jsonl \
    --provider stub --model stub-provider-2026-07-28

# Collect against a real provider
python -m orchestrator.collect --items items.jsonl --out records.jsonl \
    --provider anthropic --model <pinned-snapshot-id>

# Heterogeneous council (condition 10)
python -m orchestrator.collect --items items.jsonl --out records.jsonl \
    --condition heterogeneous_council \
    --provider anthropic --model <snapshot-a> \
    --role-provider consistency_checker=openai:<snapshot-b>

# Then train on it
python main.py all --records records.jsonl --strict-real
```

Item input, one JSON object per line:

```json
{"item_id": "tqa-0001", "benchmark": "truthfulqa",
 "question": "...", "options": ["a", "b"], "gold": 0, "context": ""}
```

Omit `options` for free-form benchmarks; the shared option space is then built by
clustering the agents' own answers via `normalize.cluster_answers`. That
clustering has its own error rate — see `normalize.clustering_risk_report`, which
blocks conclusions on benchmarks where it exceeds the SESOI.

## Five properties enforced structurally

Each corresponds to a requirement that is easy to satisfy by accident and easy to
break silently, so each is a test rather than a convention.

**Agent isolation.** §4 requires isolation until the declared communication
round. Each agent's user content is built from the *item alone*; no agent's
output is reachable from another's call, because the call sites share no mutable
context. `test_agents_are_isolated` plants a unique marker in every response and
asserts it never appears in any input.

**Failures are recorded, never imputed.** Refusal, truncation, timeout, and
unparseable JSON all produce `present=false` with the matching failure flag.
Nothing is retried into existence and no missing agent is filled with a neutral
guess — §8 requires publishing failed calls, and a silently imputed agent would
corrupt both the Shapley ablation and the error-correlation measurement.

**Untrusted inter-agent text.** Item content is wrapped in an explicit `<data>`
boundary, every prompt declares that supplied text is data and never
instruction, and injection-like content is **flagged rather than removed** so the
§9 robustness analysis can still see it.

**Data minimisation.** `redact()` strips emails, phone numbers, national ID
numbers, card numbers, and credentials before dispatch — the addendum's point is
that authorising one agent to see data does not authorise six. The manifest
records prompt *hashes*, never prompt text, and provider errors log only the
exception type, because messages can echo prompt content.

**Snapshot pinning.** §4 forbids a moving alias as the sole identifier.
Providers reject `*-latest` and friends at construction, so a run cannot be
recorded against an identifier that will mean something else next month.

## Schema validation rejects rather than repairs

`validate_record` implements `prompts.output_schema` directly. The important
consequence: an unexpected top-level key is a **rejection**. If a model
volunteers a `reasoning` or `thinking` field, the record is marked invalid rather
than having the field stripped — stripping would silently normalise a prompt
that had asked for chain-of-thought, which is precisely what
`ETHICS_AND_DATA_GOVERNANCE.md` forbids.

Out-of-range answer indices are clamped to `-1` and flagged; confidences are
clamped; claims with a status outside the closed vocabulary are dropped.

## Providers

| Provider | Use |
| --- | --- |
| `StubProvider` | Deterministic, offline. Injects controlled rates of refusal, truncation, and malformed JSON so failure handling is exercised rather than assumed. **Not a model simulator** — its answers carry no signal, so anything trained on them measures plumbing, not behaviour |
| `AnthropicProvider` | Messages API; SDK imported lazily. Correct as written — `max_tokens` and `temperature` are both valid, and `seed` is discarded because the API has none |
| `OpenAIProvider` | Chat completions; SDK imported lazily; passes `seed`. Recovers from two deterministic parameter rejections — see below |

Prices are recorded per provider for the L18 cost report. Add a provider by
subclassing `Provider` and registering it in `_PROVIDERS`.

**Neither real provider implements prompt caching or the Batch API.** Realised
cost is therefore list price, roughly double the "effective per call" column in
`logistics/HANDOVER.md` §2a, which assumes both. At the scoped study's ~42,000
calls that is ~$170 rather than ~$85 — immaterial against either available
credit, but not immaterial at the full run's 513,000 calls.

**`OpenAIProvider` handles two parameter incompatibilities in-band.** Newer
models reject `max_tokens` in favour of `max_completion_tokens`, and reasoning
models reject any non-default `temperature`. Both are detected from the error
message and retried once with the parameter adjusted, *without* consuming the
retry budget — these are deterministic rejections, so retrying them unchanged
would exhaust the budget and report `error` for every call in the run. Each flag
flips at most once, so a genuinely broken request still terminates. The
alternative, a hard-coded model-name allowlist, goes stale silently.

Prices on both providers are list prices for the default pinned snapshot and
must be updated when the snapshot changes; `estimate_cost` has no way to know
they are wrong.

## Manipulation checks (the scoped study's primary outcome)

`PREREGISTRATION.md` A2.1 makes the manipulation check the primary outcome.
`python -m orchestrator.manipulation` is its measurement path:

```bash
python -m orchestrator.manipulation --items dev_items.jsonl \
    --out-dir runs/checks --provider anthropic --model <snapshot> \
    --limit 200 --strict-real
```

It runs the council three times over the *same* items with the *same* seed,
changing only `Council.variant` (`operation` / `neutral` / `persona`), joins the
three streams by item and role, and scores each role's check. Cost is
`3 × 6 × n_items` — A2.4's 3,600 calls at 200 items. Everything the council
enforces is inherited rather than reimplemented; the shared seed matters here in
a way it does not for a single run, because the three passes are a paired
comparison and varying the seed would confound the operation effect with
sampling noise.

Three properties that are easy to get wrong and are therefore tested:

**The pass rule is conjunctive and standardized.** A2.2: the 95% interval must
exclude zero **and** the standardized effect must reach 0.40. Standardized
because the six checks do not share a unit — two return unbounded per-record
counts, one returns a rate difference in `[-1, 1]`, one returns negative ECE. An
absolute threshold across them is not interpretable, and the earlier absolute
`MIN_EFFECT = 0.10` was replaced for that reason. It is not a cosmetic change:
on a stub run it flips `local_claim_checker` from pass to fail.

**Absent records are excluded, not scored as zero.** Every check returns near
zero for an empty record, so scoring a refusal would count it as "the operation
did not happen" — and asymmetrically, because refusal rates differ by variant.
Exclusion is pairwise *per role*, so one role's refusal does not discard the
other five roles' observations for that item, and the count is reported.

**Outcomes are derived, so the calibration check is scoreable.**
`prompts._check_calibration` needs a `correct` key and returns `0.0` without
one, which made `confidence_calibrator` unpassable no matter what the model did.
The join derives `correct` from the item's gold label; unlabelled items
(`gold_option == -1`) omit it and are masked, as everywhere else.

One planning consequence: the pairwise-complete requirement **compounds across
three variants**. At a 5% per-variant absent rate an item survives all three with
probability ≈0.86, so collect ~15% more items than the analysis needs.

## Conditions

`--condition` selects which §3 condition to run. Conditions 5, 6, and 7 differ
*only* by prompt variant, which is the entire point — they must differ in exactly
one respect or the comparison is confounded. `CONDITION_VARIANTS` keeps that
mapping explicit so a later edit cannot blur it.

Still to build (P5 in `REMAINING_PROMPTS.md`): matched-deliberation single agent,
self-consistency k=6, and generic debate. Those need multi-turn or multi-sample
control flow rather than a different prompt.

## What the manifest carries

Model snapshots per role, heterogeneity flag, prompt version and per-variant
hashes, sampling parameters, seed, call counts, and exclusion rates for refusal,
truncation, malformed output, error, and retries. Written next to the records as
`<out>.manifest.json`, and merged into the run manifest by `utils.build_manifest`.
