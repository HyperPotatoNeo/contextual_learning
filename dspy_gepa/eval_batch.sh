#!/bin/bash
# Batch eval comparison: standard vs GEPA teacher prompt
# Submit: sbatch -A m4881 -C "gpu&hbm80g" --qos=interactive --time 1:00:00 --gpus-per-node 4 -J eval_compare -o /pscratch/sd/s/siddart2/dspy_gepa/eval_compare_%j.log -e /pscratch/sd/s/siddart2/dspy_gepa/eval_compare_%j.log eval_batch.sh

set -e

cd /global/homes/s/siddart2
export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd $SCRATCH

echo "============================================"
echo "  Eval Comparison: Standard vs GEPA Teacher"
echo "  Job: $SLURM_JOB_ID on $(hostname)"
echo "  Started: $(date)"
echo "============================================"

podman-hpc run --rm -it \
  --user "$(id -u):$(id -g)" \
  --replace \
  --name skyrl_eval \
  --group-add keep-groups \
  --userns keep-id \
  --gpu \
  --nccl \
  --shm-size=8g \
  -e SCRATCH -e HOME \
  -v "$SCRATCH":"$SCRATCH" \
  -v "$HOME":"$HOME" \
  -w "$SCRATCH/dspy_gepa" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash /pscratch/sd/s/siddart2/dspy_gepa/run_eval_comparison.sh
