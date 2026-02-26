#!/usr/bin/env bash
# One-time setup for prime-rl on NERSC Perlmutter
# Run from an interactive GPU session:
#   salloc -A m5017 -C "gpu&hbm80g" --qos=interactive --time 2:00:00 --gpus-per-node 1
#   bash scripts/nersc/setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

echo "=== prime-rl NERSC Setup ==="
echo "Container image: ${SKYRL_IMAGE}"
echo "Repo dir (SCRATCH): ${REPO_DIR}"
echo ""

# ============================================================================
# Step 1: Pull the SkyRL container image
# ============================================================================
echo "--- Step 1: Pulling container image ---"
export PODMANHPC_PODMAN_BIN
podman-hpc pull "${SKYRL_IMAGE}"
echo "Image pulled successfully."
echo ""

# ============================================================================
# Step 2: Clone/update repo on $SCRATCH
# ============================================================================
echo "--- Step 2: Syncing repo to SCRATCH ---"
if [ -d "${REPO_DIR}/.git" ]; then
    echo "Repo already exists at ${REPO_DIR}, pulling latest..."
    git -C "${REPO_DIR}" pull
elif [ -d "${REPO_DIR}" ]; then
    # Directory exists but isn't a git repo (e.g. only logs/ was created)
    echo "Directory exists at ${REPO_DIR} but no .git — cloning into it..."
    git clone "${REPO_HOME_DIR}" "${REPO_DIR}.tmp"
    # Move .git and working tree into existing dir (preserves logs/ etc.)
    cp -a "${REPO_DIR}.tmp/." "${REPO_DIR}/"
    rm -rf "${REPO_DIR}.tmp"
else
    echo "Cloning from ${REPO_HOME_DIR} to ${REPO_DIR}..."
    git clone "${REPO_HOME_DIR}" "${REPO_DIR}"
fi
echo ""

# ============================================================================
# Step 3: Create cache and log directories
# ============================================================================
echo "--- Step 3: Creating cache directories ---"
mkdir -p "${UV_CACHE}" "${UV_PYTHON_DIR}" "${HF_CACHE}" "${LOG_DIR}"
echo "Created: ${UV_CACHE}"
echo "Created: ${UV_PYTHON_DIR}"
echo "Created: ${HF_CACHE}"
echo "Created: ${LOG_DIR}"
echo ""

# ============================================================================
# Step 4: Install dependencies inside the container
# ============================================================================
echo "--- Step 4: Installing dependencies inside container ---"
export PODMANHPC_PODMAN_BIN
podman-hpc run --rm --gpu --user 0:0 \
    -v "${SCRATCH}:${SCRATCH}" \
    -w "${REPO_DIR}" \
    -e "HOME=${SCRATCH}" \
    -e "UV_CACHE_DIR=${UV_CACHE}" \
    -e "UV_PYTHON_INSTALL_DIR=${UV_PYTHON_DIR}" \
    -e "HF_HOME=${HF_CACHE}" \
    "${SKYRL_IMAGE}" \
    bash -c '
        set -euo pipefail
        echo "Inside container: $(hostname)"
        echo "Python (system): $(python3 --version 2>&1 || echo "not found")"
        echo "uv location: $(which uv 2>/dev/null || echo "not found")"
        echo ""

        echo "Installing Python 3.12 via uv..."
        uv python install 3.12

        echo ""
        echo "Running uv sync --all-extras..."
        uv sync --all-extras

        echo ""
        echo "Dependencies installed successfully."
    '
echo ""

# ============================================================================
# Step 5: Sanity check — verify CUDA works inside container
# ============================================================================
echo "--- Step 5: Sanity check ---"
export PODMANHPC_PODMAN_BIN
podman-hpc run --rm --gpu --user 0:0 \
    -v "${SCRATCH}:${SCRATCH}" \
    -w "${REPO_DIR}" \
    -e "HOME=${SCRATCH}" \
    -e "UV_CACHE_DIR=${UV_CACHE}" \
    -e "UV_PYTHON_INSTALL_DIR=${UV_PYTHON_DIR}" \
    "${SKYRL_IMAGE}" \
    bash -c '
        set -euo pipefail
        source .venv/bin/activate
        python3 -c "
import torch
print(f\"PyTorch version: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"GPU count: {torch.cuda.device_count()}\")
    print(f\"GPU name: {torch.cuda.get_device_name(0)}\")
else:
    print(\"WARNING: CUDA not available! Check GPU allocation.\")
    exit(1)
"
    '
echo ""

echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  cd ${REPO_DIR}"
echo "  export WANDB_MODE=disabled"
echo "  bash scripts/nersc/run.sh trainer @ configs/debug/rl/train.toml"
