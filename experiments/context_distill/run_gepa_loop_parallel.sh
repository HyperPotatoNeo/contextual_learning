#!/usr/bin/env bash
# =============================================================================
# GEPA Loop: Launch 2 seeds in parallel on 8 GPUs (4 per seed)
#
# GPU allocation:
#   Seed 42: inference=[0], teacher=[1], trainer=[2,3], GEPA vLLM on 0,1,2,3 (port 8000)
#   Seed 43: inference=[4], teacher=[5], trainer=[6,7], GEPA vLLM on 4,5,6,7 (port 8001)
#
# Usage:
#   bash experiments/context_distill/run_gepa_loop_parallel.sh
#   bash experiments/context_distill/run_gepa_loop_parallel.sh --dry-run
#   bash experiments/context_distill/run_gepa_loop_parallel.sh --start-iter 2
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Source .env if present (for WANDB_API_KEY, ANTHROPIC_API_KEY, etc.)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Pass through any CLI args (--dry-run, --start-iter N)
EXTRA_ARGS="$@"

echo "============================================================"
echo "GEPA Loop: Launching 2 seeds in parallel"
echo "  Seed 42: GPUs 0-3, RL ports 8000+, GEPA port 8100"
echo "  Seed 43: GPUs 4-7, RL ports 9000+, GEPA port 9100"
echo "  Extra args: ${EXTRA_ARGS:-none}"
echo "============================================================"

# Ensure NCCL doesn't bind to wrong interface
unset NCCL_SOCKET_IFNAME

# Create log directory
mkdir -p outputs/gepa_loop_s42 outputs/gepa_loop_s43

# Launch seed 42 on GPUs 0-3
# RL ports: 8000 range (inference=8000, teacher=8001 auto)
# GEPA vLLM port: 8100
GEPA_SEED=42 \
OUTPUT_DIR=outputs/gepa_loop_s42 \
BASE_CONFIG=experiments/context_distill/rl_gepa_loop_s42.toml \
INFERENCE_GPU_IDS='[0]' \
TEACHER_GPU_IDS='[1]' \
TRAINER_GPU_IDS='[2,3]' \
INFERENCE_PORT=8000 \
GEPA_GPU_IDS=0,1,2,3 \
GEPA_TP=4 \
VLLM_PORT=8100 \
    bash experiments/context_distill/run_gepa_loop.sh $EXTRA_ARGS \
    > >(tee outputs/gepa_loop_s42/run.log) 2>&1 &
PID_S42=$!
echo "Seed 42 launched (PID $PID_S42), logging to outputs/gepa_loop_s42/run.log"

# Stagger launch to avoid vLLM distributed init port collisions
sleep 10

# Launch seed 43 on GPUs 4-7
# RL ports: 9000 range (inference=9000, teacher=9001 auto)
# GEPA vLLM port: 9100
GEPA_SEED=43 \
OUTPUT_DIR=outputs/gepa_loop_s43 \
BASE_CONFIG=experiments/context_distill/rl_gepa_loop_s43.toml \
INFERENCE_GPU_IDS='[4]' \
TEACHER_GPU_IDS='[5]' \
TRAINER_GPU_IDS='[6,7]' \
INFERENCE_PORT=9000 \
GEPA_GPU_IDS=4,5,6,7 \
GEPA_TP=4 \
VLLM_PORT=9100 \
    bash experiments/context_distill/run_gepa_loop.sh $EXTRA_ARGS \
    > >(tee outputs/gepa_loop_s43/run.log) 2>&1 &
PID_S43=$!
echo "Seed 43 launched (PID $PID_S43), logging to outputs/gepa_loop_s43/run.log"

echo ""
echo "Both seeds running. Monitor with:"
echo "  tail -f outputs/gepa_loop_s42/run.log"
echo "  tail -f outputs/gepa_loop_s43/run.log"
echo ""
echo "Waiting for both to complete..."

# Wait for both and capture exit codes
EXIT_42=0
EXIT_43=0
wait $PID_S42 || EXIT_42=$?
wait $PID_S43 || EXIT_43=$?

echo ""
echo "============================================================"
echo "PARALLEL GEPA LOOP COMPLETE"
echo "  Seed 42: exit code $EXIT_42"
echo "  Seed 43: exit code $EXIT_43"
echo "============================================================"

# Exit with error if either failed
if [ $EXIT_42 -ne 0 ] || [ $EXIT_43 -ne 0 ]; then
    echo "ERROR: One or more seeds failed"
    exit 1
fi
