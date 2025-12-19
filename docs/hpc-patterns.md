# Practical HPC patterns

## 1) GPU-hours math and resource requests

GPU-hour usage is roughly:

> `allocated GPUs × wall time used`

Example if you run for 2 hours:
- 1 GPU job = 2 GPU-hours
- 4 GPU job = 8 GPU-hours
- 4 GPUs requested but only 1 used = still 8 GPU-hours

So:
- Debug on 1 GPU with tiny `--time`, tiny `--max_steps`, tiny `--dataset_size`.
- Use 4 GPUs only when you truly want DDP speed or you pack multiple trials inside the allocation.

## 2) Data lifecycle: persistent -> scratch -> persistent
On SERC DGX systems, fast local storage is also temporary:

- DGX-1: `/localscratch`
- DGX-H100: `/raid`

Good pattern:
1. Keep the canonical dataset in persistent storage (shared/project storage, lab server, etc).
2. At job start: `rsync` to scratch (for example: `/localscratch/$USER/datasets/<name>` or `/raid/$USER/datasets/<name>`) if feasible.
3. Train reading from scratch.
4. Stage results out (rsync logs/checkpoints to persistent).
5. If scratch gets purged, re-stage and resume from your persisted checkpoints/logs.

For huge datasets that do not fit on scratch:
- Use sharded formats (webdataset/tar shards, parquet) and stream from a shared FS.
- Cache only "hot" shards to scratch.
- Avoid millions of tiny files on network storage (metadata overhead can dominate throughput).

## 3) Wall-clock limits: assume you will be killed
Even if nobody preempts you:
- time limit ends the job
- node issues happen
- your code crashes
- your future self makes a mistake

So:
- checkpoint frequently
- resume automatically
- log enough to diagnose failures

This kit uses `#SBATCH --signal=B:USR1@600` to get a 10-minute warning.
The training code catches SIGUSR1/SIGTERM and saves `ckpt_last.pt` before exiting.

## 4) DDP on a single DGX node (fast path)
Use `torchrun --standalone --nproc_per_node=<gpus>`.

DDP gotchas:
- Use DistributedSampler + `sampler.set_epoch(epoch)`
- Save checkpoints only on rank 0
- Increase batch size carefully; tune LR (linear scaling rule is a starting point, not a law of nature)

## 5) Hyperparameter sweeps: job arrays vs packing
Two sane options:

### A) SLURM array (cleanest)
- 1 GPU per trial
- `#SBATCH --array=0-31%4` caps concurrency
- easy accounting and isolation

### B) Pack trials into a multi-GPU allocation (fast if queue overhead is high)
- request 4 GPUs
- run 4 independent 1-GPU trials in parallel inside one job
- do NOT oversubscribe CPU/RAM
- you must pin each trial to its own GPU (this kit does it with CUDA_VISIBLE_DEVICES)

## 6) Container image longevity

On these systems, images older than ~14 days may be deleted.
So:
- keep Dockerfile + requirements in git
- optionally `docker save` the image to a tarball and copy it out
