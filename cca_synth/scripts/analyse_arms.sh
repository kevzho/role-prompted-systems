#!/bin/bash
# Re-run the full analysis pipeline (synthesiser training, evaluation, leave-one-out
# attribution, robustness probes) on each shipped council arm. Offline; no
# credentials. Writes runs/arms-<arm>/ (evaluation.json, ablation.json, ...).
set -euo pipefail
cd "$(dirname "$0")/.."
for cond in cca_council persona_only_council neutral_council; do
  echo "=== analysing $cond $(date '+%F %T') ==="
  python3 main.py all \
    --records runs/arms/records.$cond.jsonl \
    --config configs/scoped.json --strict-real \
    --run-name arms-$cond
done
