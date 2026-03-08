# Docs

This directory contains the quickstart, walkthrough, cluster reference, and troubleshooting notes for running this repo on IISc/SERC clusters.

## Recommended reading order

1. [`README.md`](../README.md) (quickstart workflow)
2. [`INSTRUCTIONS.md`](../INSTRUCTIONS.md) (end-to-end walkthrough)
3. Cluster reference (policies, storage, queues):
   - [`docs/serc-dgx1.md`](serc-dgx1.md)
   - [`docs/serc-dgxh100.md`](serc-dgxh100.md)
4. [`docs/workflow-coverage.md`](workflow-coverage.md) (what this repo covers and where it stops)
5. [`docs/hpc-patterns.md`](hpc-patterns.md) (design rationale)
6. [`docs/troubleshooting.md`](troubleshooting.md) (common failure modes and fixes)

## Reference pages

- [`docs/serc-dgx1.md`](serc-dgx1.md): IISc SERC DGX-1 (`nvidia-dgx`) reference: access, storage policy, SLURM queues, Docker usage.
- [`docs/serc-dgxh100.md`](serc-dgxh100.md): IISc SERC DGX-H100 (`dgxh100`) reference: access, storage policy, SLURM queues, Docker usage.
- [`docs/dgxh100-adaptation.md`](dgxh100-adaptation.md): How to adapt this repo’s DGX-1 scripts to DGX-H100 (`/raid`, partition names, etc).
- [`docs/workflow-coverage.md`](workflow-coverage.md): Supported workflows, missing workflows, and the repo's intended scope.
- [`docs/hpc-patterns.md`](hpc-patterns.md): Practical HPC patterns (checkpointing, sweeps, data staging, GPU-hours math).
- [`docs/troubleshooting.md`](troubleshooting.md): Troubleshooting guide for SLURM + Docker workflows.
