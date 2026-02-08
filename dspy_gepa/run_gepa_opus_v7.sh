#!/bin/bash
cd /pscratch/sd/s/siddart2/dspy_gepa
source .venv/bin/activate
export WANDB_API_KEY=${WANDB_API_KEY:?Set WANDB_API_KEY env var}
unset NCCL_SOCKET_IFNAME

# GEPA Run 7: Fixed PrimeRLAdapter field-based signature detection
# Key fix: GEPA's build_program() calls signature.with_instructions() which creates
# a "StringSignature" (losing __name__="SokobanSolver"). PrimeRLAdapter was checking
# __name__ to decide raw vs ChatAdapter format. After with_instructions(), it fell
# through to ChatAdapter, causing 13.5% accuracy instead of ~34%.
# Fix: Check field names {question} -> {answer} instead of __name__.
# Also includes: cache clearing, dataset fix, adapter global config from earlier runs.

# CRITICAL: Clear DSPy disk cache BEFORE starting Python.
echo "Clearing DSPy disk cache..."
rm -rf "$HOME/.dspy_cache"
mkdir -p "$HOME/.dspy_cache"
echo "  Cache cleared: $HOME/.dspy_cache"

python -u sokoban_gepa.py \
    --reflection-model claude-opus \
    --train-size 500 --val-size 200 --test-size 500 \
    --max-metric-calls 10000 \
    --reflection-minibatch-size 5 \
    --num-threads 32 \
    --temperature 1.0 \
    --max-tokens 8192 \
    > /pscratch/sd/s/siddart2/dspy_gepa/gepa_opus_run7.log 2>&1
