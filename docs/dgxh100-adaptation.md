# Running this repo on DGX-H100 (`dgxh100`)

This repo’s `slurm/dgx1/*.sbatch` scripts are written for DGX-1 conventions (`/localscratch`, DGX-1 partition names). DGX-H100 differs mainly in:

- login host: `dgxh100.serc.iisc.ac.in`
- scratch path: `/raid/<user>` (not `/localscratch/<user>`)
- partitions/queue names: see `docs/serc-dgxh100.md`
- recommended container base: prefer a **CUDA 12 + recent PyTorch** image for H100

## Quick checklist

1. Read `docs/serc-dgxh100.md` (policies, storage, queue names).
2. Copy the repo to scratch:

   ```bash
   rsync -av dgx-demo/ <YOUR_USER>@dgxh100.serc.iisc.ac.in:/raid/<YOUR_USER>/dgx-demo/
   ssh <YOUR_USER>@dgxh100.serc.iisc.ac.in
   cd /raid/$USER/dgx-demo
   ```

3. Build a container image:
   - Use `Dockerfile.dgxh100` for the published CUDA 12.2 base path.
   - Keep the UID/GID build args so file permissions work correctly on bind mounts.

4. Verify available partitions:

   ```bash
   sinfo
   ```

5. Start by running a 1-GPU sanity test (adapted scripts are in `slurm/dgxh100/`):

   ```bash
   sbatch slurm/dgxh100/00_test_container_1gpu.sbatch
   ```

   Then run a 1-GPU training job (12 hours, signal handling enabled):

   ```bash
   sbatch slurm/dgxh100/01_train_1gpu_12h_signal.sbatch
   ```

## Included DGX-H100 scripts

The `slurm/dgxh100/` folder mirrors the DGX-1 workflows in `slurm/dgx1/`, with DGX-H100 paths and partitions:

- `00_test_container_1gpu.sbatch`: container sanity check (1 GPU)
- `01_train_1gpu_12h_signal.sbatch`: 1-GPU training with SIGUSR1 handling (12h)
- `02_train_1gpu_12h_chain.sbatch`: chained 12h segments (with `chain_submit.sh`)
- `03_train_2gpu_ddp_12h_signal.sbatch`: 2-GPU DDP via `torchrun` (uses `q_1day-2G`)
- `10_sweep_array_1gpu.sbatch`: SLURM array sweep (1 GPU per trial)
- `11_sweep_pack_2gpu_one_job.sbatch`: pack 2 trials into a 2-GPU job
- `chain_submit.sh`: helper to submit chained segments

Note: DGX-H100 publishes 1-GPU and 2-GPU queues (no 4-GPU queue). Use `sinfo` to confirm
available partitions and adjust `#SBATCH --partition` as needed.

## What to change if you adapt scripts yourself

In any DGX-1 `sbatch` script, update:

1. Scratch root:
   - DGX-1: `SCRATCH="/localscratch/${USER_NAME}"`
   - DGX-H100: `SCRATCH="/raid/${USER_NAME}"`
2. Project path:
   - `PROJ="${SCRATCH}/dgx-demo"`
3. Volume mount:
   - `-v "${SCRATCH}:${SCRATCH}"`
4. `#SBATCH --partition=...` and `#SBATCH --gres=gpu:<n>`
   - Use `sinfo` to confirm what exists and what your job can request.
