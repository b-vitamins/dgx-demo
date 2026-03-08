# Performance and profiling checklist

This repo includes a short profiling path for single-node runs:

- Python entrypoint: `python -m src.profile_train`
- DGX-1 script: `slurm/dgx1/30_profile_1gpu.sbatch`
- DGX-H100 script: `slurm/dgxh100/30_profile_1gpu.sbatch`

## What it measures

- average dataloader wait time
- average training step time
- average total step time
- samples per second
- peak allocated GPU memory

It writes `profile_metrics.jsonl` and `profile_summary.json`.

## How to use it

Start with synthetic data to isolate framework and hardware behavior:

```bash
sbatch slurm/dgx1/30_profile_1gpu.sbatch
```

Then rerun against a real staged dataset:

```bash
sbatch --export=ALL,DATASET_TYPE=imagefolder,DATA_ROOT=/localscratch/$USER/datasets/mydataset slurm/dgx1/30_profile_1gpu.sbatch
```

## How to read the results

- If `data_time` is a large fraction of `total_time`, you are data-bound.
- If `step_time` dominates and GPU memory is low, increase batch size until memory or throughput says stop.
- If GPU memory is near the limit, compare DDP and FSDP instead of only shrinking batch size.
- If synthetic data is fast but your real dataset is slow, the bottleneck is in file IO or preprocessing, not NCCL or the model.

## Practical checklist

- run `bash scripts/preflight_cluster.sh` first
- confirm GPU visibility with the `00_*` sanity job
- profile synthetic data before profiling a real dataset
- sweep `batch_size` and `num_workers`
- keep an eye on `OMP_NUM_THREADS` and `SLURM_CPUS_PER_TASK`
- compare 1-GPU profiling with your DDP or FSDP run before blaming NCCL
- for multi-GPU sweeps, remember CPU and RAM are shared across all packed trials
