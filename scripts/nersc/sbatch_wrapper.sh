#!/usr/bin/env bash
#SBATCH -A m5017
#SBATCH -C "gpu&hbm80g"
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.out
#SBATCH --job-name=prime-rl
#
# SLURM batch wrapper for prime-rl on NERSC Perlmutter
#
# Usage:
#   cd $SCRATCH/contextual_learning
#
#   # Default: context distillation experiment
#   sbatch scripts/nersc/sbatch_wrapper.sh
#
#   # Custom command via PRIME_CMD:
#   PRIME_CMD="trainer @ configs/debug/rl/train.toml" sbatch --time 00:30:00 scripts/nersc/sbatch_wrapper.sh
#
#   # Premium QOS for faster scheduling (costs more credits):
#   sbatch --qos=premium scripts/nersc/sbatch_wrapper.sh
set -euo pipefail

# SLURM copies batch scripts to a spool dir, so BASH_SOURCE[0] won't resolve
# to the original path. Use SCRATCH to find the repo and config.
REPO_DIR="${SCRATCH}/contextual_learning"
SCRIPT_DIR="${REPO_DIR}/scripts/nersc"
source "${SCRIPT_DIR}/config.env"

# Load API keys from .env (check HOME repo first, then SCRATCH)
for _env_file in "${HOME}/contextual_learning/.env" "${REPO_DIR}/.env"; do
    if [ -f "${_env_file}" ]; then
        set -a
        source "${_env_file}"
        set +a
        break
    fi
done

# Ensure log directory exists
mkdir -p "${REPO_DIR}/logs"

# Default command: context distillation with 4 GPUs
DEFAULT_CMD="rl @ experiments/context_distill/rl.toml --inference-gpu-ids 0 --teacher-gpu-ids 1 --trainer-gpu-ids '[2,3]'"
CMD="${PRIME_CMD:-${DEFAULT_CMD}}"

echo "=== prime-rl SLURM Job ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${SLURM_GPUS_ON_NODE:-unknown}"
echo "Command:   ${CMD}"
echo "Started:   $(date)"
echo ""

# Run the command inside the container
bash "${SCRIPT_DIR}/run.sh" ${CMD}
# Note: run.sh is invoked via bash (not SLURM), so its BASH_SOURCE resolves fine

echo ""
echo "=== Job finished: $(date) ==="
