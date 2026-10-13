#!/usr/bin/env bash
# Two more seeds on the HEADLINE arm only. The single-seed result (0.7010) straddles the
# pre-registered 0.69 boundary by 0.011, which is smaller than plausible run-to-run noise,
# so the verdict cannot be called from one run. ~2.8h each.
set -u
cd "$(dirname "$0")/.."
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
for S in 1 2; do
  echo "[$(date +%H:%M:%S)] masked seed $S"
  .venv/bin/python -m ft.train --arm masked --epochs 3 --bs 16 --seeds $S
  echo "[$(date +%H:%M:%S)] seed $S exit=$?"
done
echo "[$(date +%H:%M:%S)] done"
