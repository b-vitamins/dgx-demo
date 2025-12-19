#!/bin/bash
#SBATCH --job-name=dgx_test_container
#SBATCH --partition=q_1day-1G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

USER_NAME="${USER}"
SCRATCH="/localscratch/${USER_NAME}"
PROJ="${SCRATCH}/dgx-demo"
IMAGE="${USER_NAME}/dgx-demo:torch"

echo "[slurm] JobID=$SLURM_JOB_ID Node=$(hostname) GPUs=$CUDA_VISIBLE_DEVICES"
date

docker run --rm \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --ipc=host --shm-size=2G \
  --user "$(id -u ${USER_NAME}):$(id -g ${USER_NAME})" \
  -v "${SCRATCH}:${SCRATCH}" \
  -w "${PROJ}" \
  "${IMAGE}" \
  bash -lc "chmod +x scripts/sanity_check_container.sh && ./scripts/sanity_check_container.sh"
