#!/bin/bash
#SBATCH --job-name=dgx_demo_ddp4
#SBATCH --partition=q_1day-4G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
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

OUTDIR="${RUNS}/${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"

echo "[slurm] JobID=$SLURM_JOB_ID Node=$(hostname) GPUs=$CUDA_VISIBLE_DEVICES"
date

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NCCL_DEBUG=warn
export PYTHONUNBUFFERED=1

# We wrap torchrun so the container PID1 can catch SLURM signals and forward them
# to the torchrun process group (and therefore to all worker processes).
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

echo "[container] starting torchrun DDP"
torchrun --standalone --nproc_per_node=4 -m src.train \
  --outdir "'"${OUTDIR}"'" \
  --run_name hello_ddp4 \
  --max_steps 200000 \
  --batch_size 128 \
  --lr 1e-3 \
  --wd 1e-4 \
  --dropout 0.1 \
  --dataset_size 400000 \
  --num_workers 4 \
  --checkpoint_every 500 \
  --log_every 50 \
  --amp \
  --resume auto &
TR_PID=$!

# Forward signals to the whole process group for torchrun (negative PID does that).
trap "echo \"[container] got SIGUSR1 -> forwarding\"; kill -USR1 -$TR_PID 2>/dev/null || true" USR1
trap "echo \"[container] got SIGTERM -> forwarding\"; kill -TERM -$TR_PID 2>/dev/null || true" TERM

wait $TR_PID
echo "[container] torchrun finished"
'
