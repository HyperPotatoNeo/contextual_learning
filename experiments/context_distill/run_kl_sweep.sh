#!/bin/bash
# Parallel KL sweep experiments: 4 experiments × 2 seeds = 8 runs
# 2 runs in parallel (4 batches), each on 4 GPUs:
#   Run on GPUs 0-3: inference [0,1] (dp=2), teacher [2], trainer [3]
#   Run on GPUs 4-7: inference [4,5] (dp=2), teacher [6], trainer [7]
#
# Usage: bash experiments/context_distill/run_kl_sweep.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

# Load environment variables (WANDB_API_KEY etc.)
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

GPU_ARGS_LO="--inference-gpu-ids '[0,1]' --teacher-gpu-ids '[2]' --trainer-gpu-ids '[3]'"
GPU_ARGS_HI="--inference-gpu-ids '[4,5]' --teacher-gpu-ids '[6]' --trainer-gpu-ids '[7]'"

DIR="experiments/context_distill"

run_experiment() {
    local name="$1"
    local config="$2"
    local gpu_args="$3"
    echo ""
    echo "============================================================"
    echo "  Starting experiment: $name"
    echo "  Config: $config"
    echo "  Time: $(date)"
    echo "============================================================"
    echo ""
    eval uv run rl @ "$config" $gpu_args
    echo ""
    echo "  Finished: $name at $(date)"
    echo ""
}

# Batch 1: A1-s42 + A1-s43
echo "============================================================"
echo "  Batch 1: A1-s42 (GPUs 0-3) + A1-s43 (GPUs 4-7)"
echo "  Time: $(date)"
echo "============================================================"
run_experiment "A1-s42" "$DIR/rl_task1.0_kl0.0_s42.toml" "$GPU_ARGS_LO" &
PID1=$!
run_experiment "A1-s43" "$DIR/rl_task1.0_kl0.0_s43.toml" "$GPU_ARGS_HI" &
PID2=$!
wait $PID1 $PID2
echo "  Batch 1 complete at $(date)"

# Batch 2: A2-s42 + A2-s43
echo "============================================================"
echo "  Batch 2: A2-s42 (GPUs 0-3) + A2-s43 (GPUs 4-7)"
echo "  Time: $(date)"
echo "============================================================"
run_experiment "A2-s42" "$DIR/rl_task1.0_kl0.01_s42.toml" "$GPU_ARGS_LO" &
PID3=$!
run_experiment "A2-s43" "$DIR/rl_task1.0_kl0.01_s43.toml" "$GPU_ARGS_HI" &
PID4=$!
wait $PID3 $PID4
echo "  Batch 2 complete at $(date)"

# Batch 3: A3-s42 + A3-s43
echo "============================================================"
echo "  Batch 3: A3-s42 (GPUs 0-3) + A3-s43 (GPUs 4-7)"
echo "  Time: $(date)"
echo "============================================================"
run_experiment "A3-s42" "$DIR/rl_task1.0_kl0.001_s42.toml" "$GPU_ARGS_LO" &
PID5=$!
run_experiment "A3-s43" "$DIR/rl_task1.0_kl0.001_s43.toml" "$GPU_ARGS_HI" &
PID6=$!
wait $PID5 $PID6
echo "  Batch 3 complete at $(date)"

# Batch 4: B1-s42 + B1-s43
echo "============================================================"
echo "  Batch 4: B1-s42 (GPUs 0-3) + B1-s43 (GPUs 4-7)"
echo "  Time: $(date)"
echo "============================================================"
run_experiment "B1-s42" "$DIR/rl_task0.0_kl1.0_s42.toml" "$GPU_ARGS_LO" &
PID7=$!
run_experiment "B1-s43" "$DIR/rl_task0.0_kl1.0_s43.toml" "$GPU_ARGS_HI" &
PID8=$!
wait $PID7 $PID8
echo "  Batch 4 complete at $(date)"

echo ""
echo "============================================================"
echo "  All experiments complete at $(date)"
echo "============================================================"
