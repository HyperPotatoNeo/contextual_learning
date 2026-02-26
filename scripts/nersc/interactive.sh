#!/usr/bin/env bash
# Open an interactive shell inside the SkyRL container
#
# Usage:
#   salloc -A m5017 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 --gpus-per-node 4
#   bash scripts/nersc/interactive.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

if [ ! -d "${REPO_DIR}/.venv" ]; then
    echo "ERROR: .venv not found at ${REPO_DIR}/.venv"
    echo "Run setup.sh first: bash scripts/nersc/setup.sh"
    exit 1
fi

# Container name
CONTAINER_NAME="prime-rl-interactive"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    CONTAINER_NAME="prime-rl-interactive-${SLURM_JOB_ID}"
fi

# Pass through environment variables from host (only if set)
ENV_ARGS=()
for var in WANDB_API_KEY WANDB_MODE HF_TOKEN; do
    if [ -n "${!var:-}" ]; then
        ENV_ARGS+=(-e "${var}=${!var}")
    fi
done

export PODMANHPC_PODMAN_BIN
podman-hpc run --rm -it --gpu --user 0:0 \
    --name "${CONTAINER_NAME}" \
    -v "${SCRATCH}:${SCRATCH}" \
    -v "${HOME}:${HOME}:ro" \
    -w "${REPO_DIR}" \
    -e "HOME=${SCRATCH}" \
    -e "UV_CACHE_DIR=${UV_CACHE}" \
    -e "UV_PYTHON_INSTALL_DIR=${UV_PYTHON_DIR}" \
    -e "HF_HOME=${HF_CACHE}" \
    "${ENV_ARGS[@]}" \
    "${SKYRL_IMAGE}" \
    bash -c '
        unset NCCL_SOCKET_IFNAME
        source .venv/bin/activate

        echo "============================================"
        echo " prime-rl Interactive Shell"
        echo "============================================"
        echo ""
        echo "Quick reference:"
        echo "  trainer @ configs/debug/rl/train.toml          # Debug RL (fake data, ~30s)"
        echo "  sft @ examples/reverse_text/sft.toml           # SFT example"
        echo "  rl @ experiments/context_distill/rl.toml \\     # Context distillation"
        echo "    --inference_gpu_ids 0 --teacher_gpu_ids 1 --trainer_gpu_ids 2 3"
        echo ""
        echo "  python -c \"import torch; print(torch.cuda.device_count())\"  # Check GPUs"
        echo ""
        echo "  export WANDB_MODE=disabled  # Skip W&B logging"
        echo "============================================"
        echo ""

        exec bash
    '
