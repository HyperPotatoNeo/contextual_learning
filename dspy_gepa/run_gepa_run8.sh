#!/bin/bash
#SBATCH -A m4881
#SBATCH -C gpu
#SBATCH -p gpu_ss11
#SBATCH --qos=gpu_prem
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH -J gepa_run8
#SBATCH -o /pscratch/sd/s/siddart2/dspy_gepa/gepa_opus_run8.log
#SBATCH -e /pscratch/sd/s/siddart2/dspy_gepa/gepa_opus_run8.log

export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd $HOME

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Starting container..."

podman-hpc run --rm \
    --user "$(id -u):$(id -g)" \
    --replace \
    --name skyrl_gepa \
    --group-add keep-groups \
    --userns keep-id \
    --gpu \
    --nccl \
    --shm-size=8g \
    -e SCRATCH -e HOME \
    -v "$SCRATCH":"$SCRATCH" \
    -v "$HOME":"$HOME" \
    -w "$PWD" \
    docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
    /bin/bash /pscratch/sd/s/siddart2/dspy_gepa/run_gepa_run8_inner.sh
