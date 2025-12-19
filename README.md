# IISc SERC DGX starter kit (hello-world training, real HPC behavior)

This project is intentionally small in *ML complexity* and heavy on *HPC correctness*:
- SLURM wall-clock time limits
- clean shutdown via signal handling + checkpointing (Option C)
- chained multi-segment runs via SLURM dependencies (Option B)
- single-GPU, multi-GPU DDP, and hyperparameter sweeps
- realistic data handling patterns (stage-in/out)

You run on IISc SERC DGX clusters like this:
- **SLURM allocates resources** (GPUs/CPU/RAM/time)
- **Docker provides your environment**
- **Your code lives on fast scratch** (`/localscratch` on DGX-1, `/raid` on DGX-H100) which is purged periodically, so you copy results off-node

## Start here (recommended reading order)
1. Cluster policies and queue names:
   - DGX-1 (`nvidia-dgx`): [`docs/serc-dgx1.md`](docs/serc-dgx1.md)
   - DGX-H100 (`dgxh100`): [`docs/serc-dgxh100.md`](docs/serc-dgxh100.md)
2. Run the “first job” workflow in this README (sanity test → 1-GPU training → stage results out).
3. Read [`INSTRUCTIONS.md`](INSTRUCTIONS.md) for the full end-to-end walkthrough (DDP, sweeps, data staging, backups).
4. Skim [`docs/hpc-patterns.md`](docs/hpc-patterns.md) for the reasoning behind the patterns used here.
5. If something fails, start with [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Non-negotiable policies (SERC)
- Run compute via **SLURM only** (jobs run outside SLURM may lead to account action).
- Treat scratch (`/localscratch` / `/raid`) as **temporary** (data can be deleted after ~2 weeks).
- Treat Docker images as **temporary** (images can be deleted after ~2 weeks); keep Dockerfiles/requirements and/or backups.

## Which cluster are you on?

The step-by-step commands in this README use **DGX-1** defaults (`nvidia-dgx`, `/localscratch`). If you are running on **DGX-H100** (`dgxh100`, `/raid`), start with:
- [`docs/dgxh100-adaptation.md`](docs/dgxh100-adaptation.md)
- `slurm/dgxh100/` (adapted example `sbatch` scripts)

---

## Project layout
- `Dockerfile.modern` : recommended base (pytorch/pytorch + cuda11.8)
- `Dockerfile.compat` : older CUDA 11.0.3 base + older PyTorch wheels (use if needed)
- `src/train.py` : training loop + checkpointing + SIGUSR1/SIGTERM handling + optional DDP via torchrun
- `src/sweep.py` : simple grid runner reading `configs/grid.json`
- `slurm/*.sbatch` : job scripts (1 GPU, DDP 4 GPU, arrays, packed sweeps, chaining)
- `scripts/` : small helper scripts

## Cluster reference docs
- DGX-1 (`nvidia-dgx`): [`docs/serc-dgx1.md`](docs/serc-dgx1.md)
- DGX-H100 (`dgxh100`): [`docs/serc-dgxh100.md`](docs/serc-dgxh100.md)
- Docs index: [`docs/README.md`](docs/README.md)

---

## 0) One-time setup on the DGX login node
### SSH
From inside the IISc network:
```bash
ssh <your_user>@nvidia-dgx.serc.iisc.ac.in
```

If you are using DGX-H100, the login host differs:
```bash
ssh <your_user>@dgxh100.serc.iisc.ac.in
```
For DGX-H100-specific notes and adapted SLURM scripts, see `docs/dgxh100-adaptation.md` and `slurm/dgxh100/`.

### Create a working area on localscratch
```bash
mkdir -p /localscratch/$USER
cd /localscratch/$USER
```
On DGX-H100, use `/raid/$USER` instead of `/localscratch/$USER`.

### Put this project on the DGX
Option A: copy from your laptop
```bash
# On your laptop:
rsync -av dgx-demo/ <your_user>@nvidia-dgx.serc.iisc.ac.in:/localscratch/<your_user>/dgx-demo/
```

Option B: if you have it already on DGX, move/copy into `/localscratch/$USER/dgx-demo`.

---

## 1) Build your Docker image (once, then reuse)
Inside `/localscratch/$USER/dgx-demo`:
```bash
cd /localscratch/$USER/dgx-demo

# Build modern image (recommended)
docker build -f Dockerfile.modern \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg USERNAME=$USER \
  -t $USER/dgx-demo:torch .

# If the modern one fails, try compatibility mode:
# docker build -f Dockerfile.compat ... -t $USER/dgx-demo:torch .
```

Verify the image exists:
```bash
docker image list | grep dgx-demo
```

---

## 2) Sanity-check the container on a GPU (SLURM, not freestyle)
```bash
cd /localscratch/$USER/dgx-demo
sbatch slurm/00_test_container_1gpu.sbatch
squeue -u $USER
```

When it finishes, read the output:
```bash
cat dgx_test_container_<JOBID>.out
```

---

## 3) Run the single-GPU training demo (Option C: graceful time-limit exit)
This uses:
- `--time=12:00:00` (inside a 24h partition)
- `#SBATCH --signal=B:USR1@600` (SIGUSR1 10 minutes before wall-clock limit)
- training code catches SIGUSR1/SIGTERM, checkpoints `ckpt_last.pt`, and exits

```bash
sbatch slurm/01_train_1gpu_12h_signal.sbatch
```

Outputs go here:
```bash
/localscratch/$USER/dgx-demo/runs/<SLURM_JOB_ID>/
```

---

## 4) Option B: chained jobs for multi-day runs
Submit 3 chained segments of 12 hours each (total 36h possible runtime), all resuming from the same run directory:

```bash
cd /localscratch/$USER/dgx-demo
chmod +x slurm/chain_submit.sh
./slurm/chain_submit.sh my_long_run 3
```

Check progress:
```bash
squeue -u $USER
ls -R /localscratch/$USER/dgx-demo/runs/my_long_run/checkpoints
```

---

## 5) Multi-GPU DDP demo (4 GPUs)
```bash
sbatch slurm/03_train_4gpu_ddp_12h_signal.sbatch
```

This uses:
- `torchrun --standalone --nproc_per_node=4`
- DDP + DistributedSampler
- main-rank-only checkpoint writing + barriers

---

## 6) Hyperparameter sweeps
### 6A) Cleanest: SLURM job array (1 GPU per trial)
```bash
sbatch slurm/10_sweep_array_1gpu.sbatch
```
It submits 32 tasks and runs at most 4 concurrently (`%4`).

### 6B) Fast if queue overhead is high: pack 4 trials into one 4-GPU job
```bash
sbatch slurm/11_sweep_pack_4gpu_one_job.sbatch
```
It launches 4 independent trials in parallel inside a single container, pinned to GPUs 0..3.

---

## 7) Getting results back to your computer
From your laptop:
```bash
mkdir -p results_dgx

# Copy a single run
rsync -av <your_user>@nvidia-dgx.serc.iisc.ac.in:/localscratch/<your_user>/dgx-demo/runs/<RUN_DIR> ./results_dgx/

# Or copy all runs (watch size)
rsync -av <your_user>@nvidia-dgx.serc.iisc.ac.in:/localscratch/<your_user>/dgx-demo/runs/ ./results_dgx/runs/
```
On DGX-H100, change the host and scratch path:
`<your_user>@dgxh100.serc.iisc.ac.in:/raid/<your_user>/dgx-demo/runs/`.

---

## 8) Large real datasets (what changes)
Replace `SyntheticImageDataset` with your real dataset class.
Then choose one of these patterns:

### A) Stage-in to scratch (best when it fits)
- Canonical dataset on persistent storage
- rsync to scratch (`/localscratch/$USER/...` on DGX-1, `/raid/$USER/...` on DGX-H100) at job start
- train from scratch
- rsync outputs back

Templates:
- `scripts/stage_in_dataset.sh`
- `scripts/stage_out_results.sh`

Both scripts accept explicit source/destination arguments; their defaults try to pick the right scratch root.

### B) Stream from shared storage (when it doesn't fit)
- use sharded formats
- cache hot shards to scratch if possible
- avoid millions of tiny files on network FS

---

## 9) Docker image backup (optional)
If the system clears images, you can save your image and copy it off:
```bash
docker save -o /localscratch/$USER/dgx-demo_image.tar $USER/dgx-demo:torch
# then rsync/scp that tarball to your long-term storage
```

See [`docs/README.md`](docs/README.md) for more docs, and [`docs/hpc-patterns.md`](docs/hpc-patterns.md) for the "why" behind these patterns.
