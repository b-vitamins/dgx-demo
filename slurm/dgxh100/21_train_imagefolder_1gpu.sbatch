#!/bin/bash
#SBATCH --job-name=dgxh100_imagefolder_1gpu
#SBATCH --partition=q_12hour-1G
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
SCRATCH="/raid/${USER_NAME}"
SCRATCH_CONT="/workspace${SCRATCH}"
PROJ_HOST="${SCRATCH}/dgx-demo"
PROJ_CONT="${SCRATCH_CONT}/dgx-demo"
RUNS_HOST="${PROJ_HOST}/runs"
RUNS_CONT="${PROJ_CONT}/runs"
IMAGE="${USER_NAME}/dgx-demo:torch"

DATA_ROOT_HOST="${DATA_ROOT:?set DATA_ROOT to a staged imagefolder root under ${SCRATCH}}"
case "${DATA_ROOT_HOST}" in
  "${SCRATCH}"/*) DATA_ROOT_CONT="${SCRATCH_CONT}${DATA_ROOT_HOST#${SCRATCH}}" ;;
  *) echo "DATA_ROOT must live under ${SCRATCH}" >&2; exit 2 ;;
esac

RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID}}"
OUTDIR_HOST="${RUNS_HOST}/${RUN_NAME}"
OUTDIR_CONT="${RUNS_CONT}/${RUN_NAME}"
mkdir -p "${OUTDIR_HOST}"

echo "[slurm] JobID=$SLURM_JOB_ID RUN_NAME=$RUN_NAME DATA_ROOT=$DATA_ROOT_HOST"
date

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=warn
export PYTHONUNBUFFERED=1

docker run -t --rm \
  --gpus '"device='"${CUDA_VISIBLE_DEVICES}"'"' \
  --cpus="${SLURM_CPUS_PER_TASK:-8}" \
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
  bash -lc "exec python -m src.train \
      --outdir ${OUTDIR_CONT} \
      --run_name ${RUN_NAME} \
      --dataset_type imagefolder \
      --data_root ${DATA_ROOT_CONT} \
      --max_steps 200000 \
      --batch_size 256 \
      --lr 1e-3 \
      --wd 1e-4 \
      --dropout 0.1 \
      --num_workers 4 \
      --checkpoint_every 500 \
      --log_every 50 \
      --amp \
      --resume auto"
