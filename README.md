# IISc SERC DGX training workflow

This project keeps the ML workload simple and focuses on the operational patterns needed to run reliably on IISc SERC DGX clusters:
- SLURM wall-clock time limits
- clean shutdown via signal handling + checkpointing (Option C)
- chained multi-segment runs via SLURM dependencies (Option B)
- single-GPU, single-node DDP/FSDP, and hyperparameter sweeps
- realistic data handling patterns (stage-in/out)

What it does not try to be:
- a model zoo
- a multi-node distributed training stack
- a turn-key tensor/model-parallel training framework

For a direct map of supported and unsupported workflows, see [`docs/workflow-coverage.md`](docs/workflow-coverage.md).

Jobs on IISc SERC DGX clusters follow this model:
- **SLURM allocates resources** (GPUs/CPU/RAM/time)
- **Docker provides your environment**
- **Your code lives on fast scratch** (`/localscratch` on DGX-1, `/raid` on DGX-H100) which is purged periodically, so you copy results off-node

## Start here (recommended reading order)
1. Cluster policies and queue names:
   - DGX-1 (`nvidia-dgx`): [`docs/serc-dgx1.md`](docs/serc-dgx1.md)
   - DGX-H100 (`dgxh100`): [`docs/serc-dgxh100.md`](docs/serc-dgxh100.md)
2. Run the “first job” workflow in this README (sanity test → 1-GPU training → stage results out).
3. Read [`INSTRUCTIONS.md`](INSTRUCTIONS.md) for the full end-to-end walkthrough (DDP, sweeps, data staging, backups).
4. Check [`docs/workflow-coverage.md`](docs/workflow-coverage.md) to see what the repo covers and where it stops.
5. Read [`docs/ddp-vs-fsdp.md`](docs/ddp-vs-fsdp.md) before choosing a multi-GPU path.
6. Use [`docs/performance-checklist.md`](docs/performance-checklist.md) when a run is slower than expected.
7. Skim [`docs/hpc-patterns.md`](docs/hpc-patterns.md) for the reasoning behind the patterns used here.
8. If something fails, start with [`docs/troubleshooting.md`](docs/troubleshooting.md).

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
- `Dockerfile.dgxh100` : DGX-H100 build path (published CUDA 12.2 base + PyTorch CUDA wheels)
- `Dockerfile.compat` : older CUDA 11.0.3 base + older PyTorch wheels (use if needed)
- `src/train.py` : training loop + checkpointing + SIGUSR1/SIGTERM handling + optional DDP/FSDP
- `src/eval.py` : checkpoint evaluation / batch inference
- `src/profile_train.py` : short throughput and dataloader profiling entrypoint
- `src/sweep.py` : simple grid runner reading `configs/grid.json`
- `slurm/dgx1/` : DGX-1 job scripts
- `slurm/dgxh100/` : DGX-H100 job scripts
- `scripts/` : helper scripts (preflight, sanity checks, stage-in/out)

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
Inside your scratch checkout:

- DGX-1: `/localscratch/$USER/dgx-demo`
- DGX-H100: `/raid/$USER/dgx-demo`

```bash
cd /localscratch/$USER/dgx-demo

# Build DGX-1 image (recommended)
docker build -f Dockerfile.modern \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg USERNAME=$USER \
  -t $USER/dgx-demo:torch .

# DGX-H100 users should prefer the dedicated CUDA 12 path:
# docker build -f Dockerfile.dgxh100 \
#   --build-arg UID=$(id -u) \
#   --build-arg GID=$(id -g) \
#   --build-arg USERNAME=$USER \
#   -t $USER/dgx-demo:torch .

# If the DGX-1 modern image fails, try compatibility mode:
# docker build -f Dockerfile.compat ... -t $USER/dgx-demo:torch .
```

On DGX-H100, replace `/localscratch/$USER/dgx-demo` with `/raid/$USER/dgx-demo`.

Verify the image exists:
```bash
docker image list | grep dgx-demo
```

Run the login-node preflight before submitting jobs:
```bash
bash scripts/preflight_cluster.sh
```

---

## 2) Sanity-check the container on a GPU (via SLURM)
```bash
cd /localscratch/$USER/dgx-demo
sbatch slurm/dgx1/00_test_container_1gpu.sbatch
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
sbatch slurm/dgx1/01_train_1gpu_12h_signal.sbatch
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
chmod +x slurm/dgx1/chain_submit.sh
./slurm/dgx1/chain_submit.sh my_long_run 3
```

Check progress:
```bash
squeue -u $USER
ls -R /localscratch/$USER/dgx-demo/runs/my_long_run/checkpoints
```

---

## 5) Multi-GPU DDP demo (4 GPUs)
```bash
sbatch slurm/dgx1/03_train_4gpu_ddp_12h_signal.sbatch
```

This uses:
- `torchrun --standalone --nproc_per_node=4`
- DDP + DistributedSampler
- rank-0 checkpoint writing with synchronization barriers

---

## 6) Hyperparameter sweeps
### 6A) SLURM job array (1 GPU per trial)
```bash
sbatch slurm/dgx1/10_sweep_array_1gpu.sbatch
```
It submits 32 tasks and runs at most 4 concurrently (`%4`).

### 6B) Pack 4 trials into one 4-GPU job
```bash
sbatch slurm/dgx1/11_sweep_pack_4gpu_one_job.sbatch
```
It launches 4 independent trials in parallel inside a single container, pinned to GPUs 0..3.

### 6C) Profile throughput before tuning blindly
```bash
sbatch slurm/dgx1/30_profile_1gpu.sbatch
```

Start with synthetic data to isolate framework and hardware behavior, then rerun against a staged real dataset:

```bash
sbatch --export=ALL,DATASET_TYPE=imagefolder,DATA_ROOT=/localscratch/$USER/datasets/mydataset slurm/dgx1/30_profile_1gpu.sbatch
```

See [`docs/performance-checklist.md`](docs/performance-checklist.md) for how to read `profile_summary.json`.

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

## 8) Large datasets
For standard class-folder datasets, you can now use the built-in image-folder path with:

```bash
sbatch --export=ALL,DATA_ROOT=/localscratch/$USER/datasets/mydataset slurm/dgx1/21_train_imagefolder_1gpu.sbatch
```

If your data format is more specialized, use the image-folder path as the first template and replace the loader later. Then choose one of these storage patterns:

### A) Stage-in to scratch (when it fits in scratch)
- Canonical dataset on persistent storage
- rsync to scratch (`/localscratch/$USER/...` on DGX-1, `/raid/$USER/...` on DGX-H100) at job start
- train from scratch
- rsync outputs back

Templates:
- `scripts/stage_in_dataset.sh`
- `scripts/stage_out_results.sh`
- `docs/real-dataset-template.md`

Both scripts accept explicit source/destination arguments; their defaults try to pick the right scratch root.

### B) Stream from shared storage (when it does not fit)
- use sharded formats
- cache hot shards to scratch if possible
- avoid millions of small files on the network filesystem

---

## 9) Docker image backup (optional)
If the system clears images, you can save your image and copy it off:
```bash
docker save -o /localscratch/$USER/dgx-demo_image.tar $USER/dgx-demo:torch
# then rsync/scp that tarball to your long-term storage
```

See [`docs/README.md`](docs/README.md) for the full documentation set and [`docs/hpc-patterns.md`](docs/hpc-patterns.md) for the design rationale behind these patterns.
