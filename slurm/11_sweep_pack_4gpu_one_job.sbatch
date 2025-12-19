#!/bin/bash
#SBATCH --job-name=dgx_sweep_pack4
#SBATCH --partition=q_1day-4G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

USER_NAME="${USER}"
SCRATCH="/localscratch/${USER_NAME}"
PROJ="${SCRATCH}/dgx-demo"
RUNS="${PROJ}/runs"
IMAGE="${USER_NAME}/dgx-demo:torch"

OUTROOT="${RUNS}/sweep_pack_${SLURM_JOB_ID}"
mkdir -p "${OUTROOT}"

echo "[slurm] JobID=$SLURM_JOB_ID Node=$(hostname) AllocGPUs=$CUDA_VISIBLE_DEVICES"
date

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NCCL_DEBUG=warn
export PYTHONUNBUFFERED=1

# One container, 4 concurrent 1-GPU trials. We also forward SLURM signals to all child processes.
docker run --rm \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --ipc=host --shm-size=40G \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u ${USER_NAME}):$(id -g ${USER_NAME})" \
  -v "${SCRATCH}:${SCRATCH}" \
  -w "${PROJ}" \
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
  -e "MKL_NUM_THREADS=${MKL_NUM_THREADS}" \
  -e "NCCL_DEBUG=${NCCL_DEBUG}" \
  -e "PYTHONUNBUFFERED=1" \
  "${IMAGE}" \
  bash -lc '
set -euo pipefail

echo "[container] visible GPUs:"; python - <<PY
import torch
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

pids=()

forward_usr1() {
  echo "[container] got SIGUSR1 -> forwarding to child pids: ${pids[*]}"
  for pid in "${pids[@]}"; do
    kill -USR1 "$pid" 2>/dev/null || true
  done
}

forward_term() {
  echo "[container] got SIGTERM -> forwarding to child pids: ${pids[*]}"
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}

trap forward_usr1 USR1
trap forward_term TERM

# Launch 4 trials at once.
# IMPORTANT: batch_size and num_workers are per trial. Don'\''t oversubscribe CPU/RAM.
for i in 0 1 2 3; do
  echo "[container] launching trial $i on GPU $i"
  CUDA_VISIBLE_DEVICES=$i python -m src.sweep \
    --trial_id $((100 + i)) \
    --outroot "'"${OUTROOT}"'" \
    --max_steps 3000 \
    --batch_size 256 \
    --dataset_size 200000 \
    --num_workers 2 \
    --amp &
  pids+=($!)
done

# Wait for all trials; if we get SIGUSR1/SIGTERM, they checkpoint and exit cleanly.
for pid in "${pids[@]}"; do
  wait "$pid" || true
done

echo "[container] all trials done"
'
