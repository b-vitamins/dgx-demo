#!/bin/bash
#SBATCH --job-name=dgx_sweep_array
#SBATCH --partition=q_1day-1G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --array=0-31%4
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

USER_NAME="${USER}"
SCRATCH="/localscratch/${USER_NAME}"
SCRATCH_CONT="/workspace${SCRATCH}"
PROJ_HOST="${SCRATCH}/dgx-demo"
PROJ_CONT="${SCRATCH_CONT}/dgx-demo"
RUNS_HOST="${PROJ_HOST}/runs"
RUNS_CONT="${PROJ_CONT}/runs"
IMAGE="${USER_NAME}/dgx-demo:torch"

OUTROOT_HOST="${RUNS_HOST}/sweep_${SLURM_ARRAY_JOB_ID}"
OUTROOT_CONT="${RUNS_CONT}/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${OUTROOT_HOST}"

echo "[slurm] ArrayJobID=$SLURM_ARRAY_JOB_ID TaskID=$SLURM_ARRAY_TASK_ID Node=$(hostname) GPUs=$CUDA_VISIBLE_DEVICES"
date

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export NCCL_DEBUG=warn
export PYTHONUNBUFFERED=1

docker run -t --rm \
  --gpus '"device='"${CUDA_VISIBLE_DEVICES}"'"' \
  --name "${SLURM_JOB_ID}" \
  --ipc=host --shm-size=20G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u ${USER_NAME}):$(id -g ${USER_NAME})" \
  -v "${SCRATCH}:${SCRATCH_CONT}" \
  -w "${PROJ_CONT}" \
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
  -e "MKL_NUM_THREADS=${MKL_NUM_THREADS}" \
  -e "NCCL_DEBUG=${NCCL_DEBUG}" \
  -e "PYTHONUNBUFFERED=1" \
  "${IMAGE}" \
  bash -lc "exec python -m src.sweep \
      --trial_id ${SLURM_ARRAY_TASK_ID} \
      --outroot ${OUTROOT_CONT} \
      --max_steps 3000 \
      --batch_size 256 \
      --dataset_size 200000 \
      --num_workers 4 \
      --amp"
