#!/bin/bash
#SBATCH --job-name=dgxh100_test_container
#SBATCH --partition=q_12hour-1G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

USER_NAME="${USER}"
SCRATCH="/raid/${USER_NAME}"
SCRATCH_CONT="/workspace${SCRATCH}"
PROJ_HOST="${SCRATCH}/dgx-demo"
PROJ_CONT="${SCRATCH_CONT}/dgx-demo"
IMAGE="${USER_NAME}/dgx-demo:torch"

echo "[slurm] JobID=$SLURM_JOB_ID Node=$(hostname) GPUs=$CUDA_VISIBLE_DEVICES"
date

docker run -t --rm \
  --gpus '"device='"${CUDA_VISIBLE_DEVICES}"'"' \
  --name "${SLURM_JOB_ID}" \
  --ipc=host --shm-size=2G \
  --user "$(id -u ${USER_NAME}):$(id -g ${USER_NAME})" \
  -v "${SCRATCH}:${SCRATCH_CONT}" \
  -w "${PROJ_CONT}" \
  "${IMAGE}" \
  bash -lc "chmod +x scripts/sanity_check_container.sh && ./scripts/sanity_check_container.sh"
