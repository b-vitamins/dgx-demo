# Docs

This directory contains the quickstart, walkthrough, cluster reference, and troubleshooting notes for running this repo on IISc/SERC clusters.

## Recommended reading order

1. [`README.md`](../README.md) (quickstart workflow)
2. [`INSTRUCTIONS.md`](../INSTRUCTIONS.md) (end-to-end walkthrough)
3. Cluster reference (policies, storage, queues):
   - [`docs/serc-dgx1.md`](serc-dgx1.md)
   - [`docs/serc-dgxh100.md`](serc-dgxh100.md)
4. [`docs/workflow-coverage.md`](workflow-coverage.md) (what this repo covers and where it stops)
5. [`docs/ddp-vs-fsdp.md`](ddp-vs-fsdp.md) (single-node distributed tradeoffs)
6. [`docs/eval-inference.md`](eval-inference.md) (checkpoint scoring template)
7. [`docs/real-dataset-template.md`](real-dataset-template.md) (image-folder dataset path)
8. [`docs/hpc-patterns.md`](hpc-patterns.md) (design rationale)
9. [`docs/troubleshooting.md`](troubleshooting.md) (common failure modes and fixes)

## Reference pages

- [`docs/serc-dgx1.md`](serc-dgx1.md): IISc SERC DGX-1 (`nvidia-dgx`) reference: access, storage policy, SLURM queues, Docker usage.
- [`docs/serc-dgxh100.md`](serc-dgxh100.md): IISc SERC DGX-H100 (`dgxh100`) reference: access, storage policy, SLURM queues, Docker usage.
- [`docs/dgxh100-adaptation.md`](dgxh100-adaptation.md): How to adapt this repo’s DGX-1 scripts to DGX-H100 (`/raid`, partition names, etc).
- [`docs/workflow-coverage.md`](workflow-coverage.md): Supported workflows, missing workflows, and the repo's intended scope.
- [`docs/ddp-vs-fsdp.md`](ddp-vs-fsdp.md): When to use single-node DDP vs single-node FSDP in this repo.
- [`docs/eval-inference.md`](eval-inference.md): Batch evaluation and prediction template for existing checkpoints.
- [`docs/real-dataset-template.md`](real-dataset-template.md): Real dataset path using a simple image-folder layout.
- [`docs/hpc-patterns.md`](hpc-patterns.md): Practical HPC patterns (checkpointing, sweeps, data staging, GPU-hours math).
- [`docs/troubleshooting.md`](troubleshooting.md): Troubleshooting guide for SLURM + Docker workflows.
