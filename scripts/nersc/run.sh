#!/usr/bin/env bash
# Run a command inside the SkyRL container on NERSC Perlmutter
#
# Usage:
#   bash scripts/nersc/run.sh trainer @ configs/debug/rl/train.toml
#   bash scripts/nersc/run.sh rl @ experiments/context_distill/rl.toml --inference_gpu_ids 0 --teacher_gpu_ids 1 --trainer_gpu_ids 2 3
#   bash scripts/nersc/run.sh python -c "import torch; print(torch.cuda.is_available())"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/nersc/run.sh <command> [args...]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/nersc/run.sh trainer @ configs/debug/rl/train.toml"
    echo "  bash scripts/nersc/run.sh rl @ experiments/context_distill/rl.toml --inference_gpu_ids 0 --teacher_gpu_ids 1 --trainer_gpu_ids 2 3"
    exit 1
fi

# ============================================================================
# Validation
# ============================================================================
if [ ! -d "${REPO_DIR}/.venv" ]; then
    echo "ERROR: .venv not found at ${REPO_DIR}/.venv"
    echo "Run setup.sh first: bash scripts/nersc/setup.sh"
    exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "disabled" ]; then
    echo "WARNING: WANDB_API_KEY not set and WANDB_MODE != disabled"
    echo "  Set WANDB_API_KEY or run: export WANDB_MODE=disabled"
    echo ""
fi

# ============================================================================
# Build container arguments
# ============================================================================

# TTY detection: use -it for interactive terminals, -i for batch/pipe
if [ -t 0 ]; then
    TTY_FLAG="-it"
else
    TTY_FLAG="-i"
fi

# Container name (include SLURM job ID if available to avoid conflicts)
CONTAINER_NAME="prime-rl"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    CONTAINER_NAME="prime-rl-${SLURM_JOB_ID}"
fi

# Pass through environment variables from host (only if set)
ENV_ARGS=()
for var in WANDB_API_KEY WANDB_MODE HF_TOKEN ANTHROPIC_API_KEY; do
    if [ -n "${!var:-}" ]; then
        ENV_ARGS+=(-e "${var}=${!var}")
    fi
done

# Build the inner command
INNER_CMD="$*"

export PODMANHPC_PODMAN_BIN
podman-hpc run --rm ${TTY_FLAG} --gpu --user 0:0 \
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
    bash -c "
        unset NCCL_SOCKET_IFNAME
        source .venv/bin/activate
        ${INNER_CMD}
    "
