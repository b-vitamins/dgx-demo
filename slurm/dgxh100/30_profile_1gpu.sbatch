#!/bin/bash
#SBATCH --job-name=dgxh100_profile_1gpu
#SBATCH --partition=q_12hour-1G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

USER_NAME="${USER}"
SCRATCH="/raid/${USER_NAME}"
SCRATCH_CONT="/workspace${SCRATCH}"
PROJ_HOST="${SCRATCH}/dgx-demo"
PROJ_CONT="${SCRATCH_CONT}/dgx-demo"
PROFILE_HOST="${PROJ_HOST}/profiles/${SLURM_JOB_ID}"
PROFILE_CONT="${PROJ_CONT}/profiles/${SLURM_JOB_ID}"
IMAGE="${USER_NAME}/dgx-demo:torch"

DATASET_TYPE="${DATASET_TYPE:-synthetic}"
DATA_ROOT_CONT=""
if [[ "${DATASET_TYPE}" == "imagefolder" ]]; then
  DATA_ROOT_HOST="${DATA_ROOT:?set DATA_ROOT when DATASET_TYPE=imagefolder}"
  case "${DATA_ROOT_HOST}" in
    "${SCRATCH}"/*) DATA_ROOT_CONT="${SCRATCH_CONT}${DATA_ROOT_HOST#${SCRATCH}}" ;;
    *) echo "DATA_ROOT must live under ${SCRATCH}" >&2; exit 2 ;;
  esac
fi

mkdir -p "${PROFILE_HOST}"

echo "[slurm] JobID=$SLURM_JOB_ID DATASET_TYPE=$DATASET_TYPE OUTDIR=$PROFILE_HOST"
date

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

docker run --rm \
  -t --gpus '"device='"${CUDA_VISIBLE_DEVICES}"'"' \
  --cpus="${SLURM_CPUS_PER_TASK:-8}" \
  --name "${SLURM_JOB_ID}" \
  --ipc=host --shm-size=20G \
  --user "$(id -u ${USER_NAME}):$(id -g ${USER_NAME})" \
  -v "${SCRATCH}:${SCRATCH_CONT}" \
  -w "${PROJ_CONT}" \
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
  -e "MKL_NUM_THREADS=${MKL_NUM_THREADS}" \
  -e "PYTHONUNBUFFERED=1" \
  "${IMAGE}" \
  bash -lc "exec python -m src.profile_train \
      --outdir ${PROFILE_CONT} \
      --dataset_type ${DATASET_TYPE} \
      ${DATA_ROOT_CONT:+--data_root ${DATA_ROOT_CONT}} \
      --batch_size 256 \
      --dataset_size 20000 \
      --num_workers 4 \
      --warmup_steps 10 \
      --profile_steps 50 \
      --amp"
