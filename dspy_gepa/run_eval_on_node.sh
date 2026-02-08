#!/bin/bash
# Run on compute node: kills old container, starts fresh eval
export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd $SCRATCH

# Force kill old containers
podman-hpc kill skyrl_gepa 2>/dev/null || true
podman-hpc rm -f skyrl_gepa 2>/dev/null || true
podman-hpc kill skyrl_eval 2>/dev/null || true
podman-hpc rm -f skyrl_eval 2>/dev/null || true

sleep 2

# Start fresh container with eval
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
