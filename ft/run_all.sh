#!/usr/bin/env bash
# Both arms, sequentially -- they contend for the one GPU, so never in parallel.
# Headline first, capability control second. ~3.2h + ~1h.
set -u
cd "$(dirname "$0")/.."
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
stamp() { date "+[%H:%M:%S]"; }

echo "$(stamp) ARM 1/2: masked  -- THE HEADLINE (facts and pleadings only)"
.venv/bin/python -m ft.train --arm masked --epochs 3 --bs 16 --seeds 573
echo "$(stamp) arm 1 exit=$?"

echo "$(stamp) ARM 2/2: last512 -- CAPABILITY CONTROL (leaky; never a reported result)"
.venv/bin/python -m ft.train --arm last512 --epochs 3 --bs 16 --seeds 573
echo "$(stamp) arm 2 exit=$?"

echo "$(stamp) done"
