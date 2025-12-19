# Running this repo on DGX-H100 (`dgxh100`)

This repo’s `slurm/*.sbatch` scripts are written for DGX-1 conventions (`/localscratch`, DGX-1 partition names). DGX-H100 differs mainly in:

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
   - Use `Dockerfile.modern`, but consider switching the `FROM pytorch/pytorch:...` tag to a CUDA 12 variant.
   - Keep the UID/GID build args so file permissions work correctly on bind mounts.

4. Check available partitions (don’t guess):

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
