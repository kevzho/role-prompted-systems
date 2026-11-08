#!/bin/bash
# Collect the benchmark arms on gpt-5.6-terra. NEEDS CREDENTIALS AND COSTS MONEY.
# Not needed to reproduce anything: the records are shipped in runs/arms/.
# Kept as the exact record of what was run. Collection happened in two stages:
#   1. 500 HaluEval + 500 TruthfulQA items per arm;
#   2. HaluEval extended to 1,000 with --resume. TruthfulQA's generation split
#      has no gold answer index, so more of it cannot add a scorable item.
# Each arm's manifest records n_items = 1,500.
#
# Usage: AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_KEY=... TERRA_DEPLOYMENT=... \
#        scripts/collect_arms.sh
set -euo pipefail
cd "$(dirname "$0")/.."
for cond in cca_council persona_only_council neutral_council direct_answer; do
  echo "=== $cond stage 1 started $(date '+%F %T') ==="
  python3 -m orchestrator.collect \
    --items data/items.halueval.jsonl --items data/items.truthfulqa.jsonl \
    --limit-per-source 500 \
    --out runs/arms/records.$cond.jsonl \
    --provider azure-openai --azure-endpoint "$AZURE_OPENAI_ENDPOINT" \
    --model "$TERRA_DEPLOYMENT" --condition $cond \
    --max-workers 4 --resume
  echo "=== $cond stage 2 started $(date '+%F %T') ==="
  python3 -m orchestrator.collect \
    --items data/items.halueval.jsonl --limit 1000 \
    --out runs/arms/records.$cond.jsonl \
    --provider azure-openai --azure-endpoint "$AZURE_OPENAI_ENDPOINT" \
    --model "$TERRA_DEPLOYMENT" --condition $cond \
    --max-workers 4 --resume
done
