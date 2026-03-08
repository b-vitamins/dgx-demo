# End-to-end workflow

This document describes an end-to-end workflow for running deep learning jobs on IISc/SERC DGX clusters using SLURM + Docker. The ML workload is intentionally simple so the *cluster mechanics* (SLURM, time limits, checkpointing, DDP, sweeps, and data logistics) remain the focus.

This walkthrough is written for DGX-1 (`nvidia-dgx`) by default. If you are using DGX-H100 (`dgxh100`), the concepts are the same but the **login host**, **scratch path**, and **partition names** differ; start with:
- [`docs/serc-dgxh100.md`](docs/serc-dgxh100.md)
- [`docs/serc-dgx1.md`](docs/serc-dgx1.md)
- [`docs/dgxh100-adaptation.md`](docs/dgxh100-adaptation.md) (how to adapt this repo, including `slurm/dgxh100/` examples)

---

## Operating rules

- Run compute via **SLURM only** (SERC explicitly warns against running jobs outside SLURM).
- Work from **scratch** (`/localscratch` on DGX-1, `/raid` on DGX-H100), not your home directory.
- Assume **scratch data** and **Docker images** can be deleted after ~2 weeks; keep backups.

---

## Workflow overview

1. Put this project on the cluster scratch filesystem.
2. Build a Docker image that matches your host UID/GID (avoids permission problems on bind mounts).
3. Submit jobs via SLURM:
   - container sanity test
   - 1-GPU training with time-limit-safe behavior (Option C)
   - multi-segment continuation using SLURM dependencies (Option B)
   - 4-GPU single-node DDP training (DGX-1 partitions)
   - hyperparameter sweep via job arrays
   - hyperparameter sweep by packing multiple trials into one multi-GPU allocation
4. Copy results back to persistent storage (or your laptop) before scratch is purged.

---

## 0) Get the project onto the cluster

### Option A (recommended): `rsync` from your laptop to scratch

From your laptop (inside the IISc network / connectivity you normally use):

```bash
rsync -av dgx-demo/ <YOUR_USER>@nvidia-dgx.serc.iisc.ac.in:/localscratch/<YOUR_USER>/dgx-demo/
```

Then SSH and go to the project:

```bash
ssh <YOUR_USER>@nvidia-dgx.serc.iisc.ac.in
cd /localscratch/$USER/dgx-demo
```

### Option B: clone/zip (only if your environment supports it)

If you have a git remote:

```bash
git clone <REPO_URL> dgx-demo
cd dgx-demo
```

---

## 1) Build the Docker image

From inside `/localscratch/$USER/dgx-demo`:

### Recommended (modern) image

```bash
docker build -f Dockerfile.modern \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg USERNAME=$USER \
  -t $USER/dgx-demo:torch .
```

### Compatibility fallback (CUDA 11.0.3 base)

If the modern image fails due to a driver/toolchain mismatch, try:

```bash
docker build -f Dockerfile.compat \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg USERNAME=$USER \
  -t $USER/dgx-demo:torch .
```

Verify:

```bash
docker image list | grep dgx-demo
```

---

## 2) Sanity test the container on a GPU (SLURM-only)

Submit a short test job:

```bash
sbatch slurm/00_test_container_1gpu.sbatch
squeue -u $USER
```

After it finishes:

```bash
cat dgx_test_container_<JOBID>.out
```

If it prints `cuda available: True` and shows a GPU name, the container is ready for training jobs.

---

## 3) Single-GPU training with time-limit-safe behavior (Option C)

Submit:

```bash
sbatch slurm/01_train_1gpu_12h_signal.sbatch
```

What this job demonstrates:

- `#SBATCH --time=12:00:00`
- `#SBATCH --signal=B:USR1@600` (10-minute warning before time limit)
- `python -m src.train` catches SIGUSR1/SIGTERM, saves `ckpt_last.pt`, and exits cleanly

Outputs:

```bash
/localscratch/$USER/dgx-demo/runs/<SLURM_JOB_ID>/
```

Watch logs:

```bash
tail -f dgx_demo_1gpu_<JOBID>.out
```

---

## 4) Multi-segment runs for multi-day experiments (Option B: chaining)

This pattern checkpoints, resumes, and chains jobs using SLURM dependencies.

Submit 3 chained segments:

```bash
cd /localscratch/$USER/dgx-demo
chmod +x slurm/chain_submit.sh
./slurm/chain_submit.sh my_long_run 3
```

All segments write to the same run directory:

```bash
/localscratch/$USER/dgx-demo/runs/my_long_run/
```

---

## 5) 4-GPU DDP demo (single node, torchrun)

This uses a 4-GPU DGX-1 partition and launches DDP with `torchrun`:

```bash
sbatch slurm/03_train_4gpu_ddp_12h_signal.sbatch
```

Notes:
- DDP is enabled automatically when `WORLD_SIZE>1`.
- Checkpoints are written on rank 0 only.
- The script forwards SLURM signals so time-limit-safe behavior still works under DDP.

If you are on DGX-H100, check available partitions with `sinfo` and adapt the `#SBATCH --partition` / `--gres=gpu:<n>` lines accordingly.

---

## 6) Hyperparameter sweeps

### 6A) SLURM job array (1 GPU per trial)

```bash
sbatch slurm/10_sweep_array_1gpu.sbatch
```

It submits 32 tasks and caps concurrency (`%4`).

Aggregate results (run inside the container):

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --ipc=host \
  -v /localscratch/$USER:/localscratch/$USER \
  -w /localscratch/$USER/dgx-demo \
  $USER/dgx-demo:torch \
  python -m src.aggregate --sweep_root /localscratch/$USER/dgx-demo/runs/sweep_<ARRAY_JOB_ID>
```

### 6B) Pack multiple 1-GPU trials into one 4-GPU job

```bash
sbatch slurm/11_sweep_pack_4gpu_one_job.sbatch
```

This launches 4 trials concurrently, each pinned to a GPU via `CUDA_VISIBLE_DEVICES`.

---

## 7) Data staging (real datasets)

This demo uses a synthetic dataset (no downloads). For real work:

### Stage-in to scratch (best if it fits)

```bash
bash scripts/stage_in_dataset.sh /path/to/persistent/dataset /localscratch/$USER/datasets/mydataset
```

### Stage results out of scratch (before purge)

```bash
bash scripts/stage_out_results.sh /localscratch/$USER/dgx-demo/runs /path/to/persistent/results
```

---

## 8) Copy results back to your laptop

From your laptop:

```bash
mkdir -p results_dgx

# Copy one run:
rsync -av <YOUR_USER>@nvidia-dgx.serc.iisc.ac.in:/localscratch/<YOUR_USER>/dgx-demo/runs/<RUN_DIR> ./results_dgx/

# Or copy all runs:
rsync -av <YOUR_USER>@nvidia-dgx.serc.iisc.ac.in:/localscratch/<YOUR_USER>/dgx-demo/runs/ ./results_dgx/runs/
```

---

## 9) Backup your Docker image (optional)

If the system clears Docker images, you can export your image and copy it off scratch:

```bash
docker save -o /localscratch/$USER/dgx-demo_image.tar $USER/dgx-demo:torch
```

---

## Next reading

- [`docs/README.md`](docs/README.md) (docs index)
- [`docs/hpc-patterns.md`](docs/hpc-patterns.md) (design rationale)
