#!/bin/bash
# Collect the manipulation-check records (A2: 6 roles x 3 variants x 200 items,
# 3,600 calls per model) and score them. NEEDS CREDENTIALS AND COSTS MONEY.
# Not needed to reproduce anything: the records this wrote are shipped in
# runs/checks/ and runs/checks-grok/. The sampling settings below are the ones
# every records.*.manifest.json reports (temperature 0, seed 1337, 1,024 tokens,
# 4 workers). The provider is recorded as azure-openai in the grok manifests;
# the terra check manifests predate the provider field, so for terra the route
# is asserted here, not derived (both models were served from the same
# Microsoft Foundry resource).
#
# Usage: AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_KEY=... \
#        scripts/collect_checks.sh <model-deployment> <out-dir>
#   e.g. scripts/collect_checks.sh gpt-5.6-terra runs/checks
#        scripts/collect_checks.sh grok-4.3      runs/checks-grok
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; OUT="$2"
python3 -m orchestrator.manipulation \
  --items data/items.halueval.dev.jsonl --limit 200 \
  --out-dir "$OUT" \
  --provider azure-openai --azure-endpoint "$AZURE_OPENAI_ENDPOINT" \
  --model "$MODEL" \
  --temperature 0 --seed 1337 --max-tokens 1024 \
  --max-workers 4 --resume
