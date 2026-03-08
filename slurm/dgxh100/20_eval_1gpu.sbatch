#!/bin/bash
#SBATCH --job-name=dgxh100_eval_1gpu
#SBATCH --partition=q_12hour-1G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

USER_NAME="${USER}"
SCRATCH="/raid/${USER_NAME}"
SCRATCH_CONT="/workspace${SCRATCH}"
PROJ_HOST="${SCRATCH}/dgx-demo"
PROJ_CONT="${SCRATCH_CONT}/dgx-demo"
RUNS_HOST="${PROJ_HOST}/runs"
RUNS_CONT="${PROJ_CONT}/runs"
IMAGE="${USER_NAME}/dgx-demo:torch"

RUN_REF="${RUN_REF:?set RUN_REF to a run directory (for example: 12345 or my_long_run)}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-ckpt_last.pt}"
CKPT_HOST="${RUNS_HOST}/${RUN_REF}/checkpoints/${CHECKPOINT_NAME}"
CKPT_CONT="${RUNS_CONT}/${RUN_REF}/checkpoints/${CHECKPOINT_NAME}"
OUTDIR_HOST="${RUNS_HOST}/${RUN_REF}/eval_${SLURM_JOB_ID}"
OUTDIR_CONT="${RUNS_CONT}/${RUN_REF}/eval_${SLURM_JOB_ID}"

mkdir -p "${OUTDIR_HOST}"

echo "[slurm] JobID=$SLURM_JOB_ID RUN_REF=$RUN_REF CHECKPOINT=$CKPT_HOST"
date

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONUNBUFFERED=1

docker run --rm \
  -t --gpus '"device='"${CUDA_VISIBLE_DEVICES}"'"' \
  --cpus="${SLURM_CPUS_PER_TASK:-6}" \
  --name "${SLURM_JOB_ID}" \
  --ipc=host --shm-size=20G \
  --user "$(id -u ${USER_NAME}):$(id -g ${USER_NAME})" \
  -v "${SCRATCH}:${SCRATCH_CONT}" \
  -w "${PROJ_CONT}" \
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
  -e "MKL_NUM_THREADS=${MKL_NUM_THREADS}" \
  -e "PYTHONUNBUFFERED=1" \
  "${IMAGE}" \
  bash -lc "exec python -m src.eval \
      --checkpoint ${CKPT_CONT} \
      --outdir ${OUTDIR_CONT} \
      --batch_size 256 \
      --num_workers 4 \
      --write_predictions"
