#!/bin/bash
#SBATCH --job-name=dgx_chain_1gpu
#SBATCH --partition=q_1day-1G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

USER_NAME="${USER}"
SCRATCH="/localscratch/${USER_NAME}"
PROJ="${SCRATCH}/dgx-demo"
RUNS="${PROJ}/runs"
IMAGE="${USER_NAME}/dgx-demo:torch"

# Stable run id across chained jobs:
RUN_ID="${RUN_ID:-longrun_demo}"
OUTDIR="${RUNS}/${RUN_ID}"
mkdir -p "${OUTDIR}"

echo "[slurm] JobID=$SLURM_JOB_ID RUN_ID=$RUN_ID Node=$(hostname) GPUs=$CUDA_VISIBLE_DEVICES"
date

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=warn
export PYTHONUNBUFFERED=1

docker run --rm \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --ipc=host --shm-size=20G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u ${USER_NAME}):$(id -g ${USER_NAME})" \
  -v "${SCRATCH}:${SCRATCH}" \
  -w "${PROJ}" \
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
  -e "MKL_NUM_THREADS=${MKL_NUM_THREADS}" \
  -e "NCCL_DEBUG=${NCCL_DEBUG}" \
  -e "PYTHONUNBUFFERED=1" \
  "${IMAGE}" \
  bash -lc "exec python -m src.train \
      --outdir ${OUTDIR} \
      --run_name chained_1gpu \
      --max_steps 1000000 \
      --batch_size 256 \
      --lr 3e-4 \
      --wd 1e-4 \
      --dropout 0.1 \
      --dataset_size 200000 \
      --num_workers 4 \
      --checkpoint_every 500 \
      --log_every 50 \
      --amp \
      --resume auto"
