# Workflow coverage

This repo is a cluster-orientation and boilerplate kit for IISc/SERC DGX systems. Its purpose is to shorten the time spent learning the cluster mechanics around a deep learning job: SLURM, Docker, scratch storage, time limits, resume logic, and basic multi-GPU patterns.

## What it covers well

- First-job checks: `scripts/preflight_cluster.sh`, `scripts/sanity_check_container.sh`, and `slurm/*/00_test_container_1gpu.sbatch`
- Single-GPU training that survives wall-clock limits: `src/train.py` and `slurm/*/01_train_1gpu_12h_signal.sbatch`
- Multi-segment continuation for multi-day experiments: `slurm/*/02_train_1gpu_12h_chain.sbatch` and `slurm/*/chain_submit.sh`
- Single-node data parallel training: `slurm/dgx1/03_train_4gpu_ddp_12h_signal.sbatch` and `slurm/dgxh100/03_train_2gpu_ddp_12h_signal.sbatch`
- Hyperparameter sweeps: SLURM arrays (`10_*`) and packed trials inside one multi-GPU allocation (`11_*`)
- Scratch data flow: `scripts/stage_in_dataset.sh` and `scripts/stage_out_results.sh`
- Cluster-specific container builds: `Dockerfile.modern`, `Dockerfile.compat`, and `Dockerfile.dgxh100`

## What it does not cover yet

- Multi-node DDP across multiple DGX nodes
- FSDP, ZeRO, DeepSpeed, Megatron-LM, NeMo, or launcher stacks built around them
- Model parallelism, tensor parallelism, and pipeline parallelism
- Inference serving, Triton, or batch-inference pipelines
- Shared-storage streaming strategies beyond the high-level notes in `docs/hpc-patterns.md`

These omissions are intentional. The repo is trying to solve the cluster-facing problems that almost every workflow shares before you layer on framework-specific distributed training stacks.

## Fast path by scenario

- I need to know whether the cluster setup is sane: run `bash scripts/preflight_cluster.sh`, then submit `slurm/<cluster>/00_test_container_1gpu.sbatch`
- I want one reliable single-GPU training template: start from `slurm/<cluster>/01_train_1gpu_12h_signal.sbatch`
- I have a run longer than one queue segment: use `slurm/<cluster>/02_train_1gpu_12h_chain.sbatch` with `chain_submit.sh`
- I want single-node data parallel training: adapt `slurm/dgx1/03_train_4gpu_ddp_12h_signal.sbatch` or `slurm/dgxh100/03_train_2gpu_ddp_12h_signal.sbatch`
- I want many independent trials: use the array script first, then the packed sweep script if you intentionally want one allocation to host several 1-GPU trials

## How advanced users should use this repo

If your real workload needs FSDP, DeepSpeed, tensor parallelism, or a model-specific launcher, keep the cluster-facing pieces from this repo and replace the training invocation:

- keep the cluster-specific Dockerfile
- keep the scratch-mount pattern
- keep the signal handling and checkpoint/resume approach when possible
- keep the SLURM resource request structure and adapt only the launcher command

That is the boundary of this repo's value: it removes a large amount of environment and scheduler trial-and-error, but it does not pretend to be a turnkey large-model training stack.
