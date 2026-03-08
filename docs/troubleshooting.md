# Troubleshooting (SLURM + Docker on SERC DGX)

This repo assumes the standard SERC pattern:

1. You SSH to a **login node**.
2. You submit jobs with **SLURM**.
3. The SLURM job launches **Docker**, bind-mounting your scratch directory.

If something goes wrong, start by identifying *where* you are running a command.

## Where am I running this?

- **Laptop**: `ssh`, `rsync`, editing files locally.
- **Login node** (`nvidia-dgx` / `dgxh100`): `sbatch`, `squeue`, building Docker images, editing code in scratch.
- **Compute node**: you generally should not log in directly; your code runs inside the SLURM job container.

## I can’t SSH to the cluster

- DGX-1: `ssh <user>@nvidia-dgx.serc.iisc.ac.in`
- DGX-H100: `ssh <user>@dgxh100.serc.iisc.ac.in`

If SSH fails, you may need to be on the IISc network or use your group’s standard remote-access path. Check with SERC or your lab if you need VPN or jump-host access.

## `sbatch` / `squeue` not found

You are likely not on the login node, or your environment modules/profile are not loaded.

Confirm:

```bash
hostname
which sbatch
```

## Docker permission errors

Symptoms:
- `permission denied` when running `docker …`
- `Got permission denied while trying to connect to the Docker daemon socket`

Checks:

```bash
groups
docker ps
```

If you are not in the `docker` group (or Docker is otherwise restricted), contact SERC support / your admin contact.

## Job is pending forever

Check status and reason:

```bash
squeue -u $USER
```

Check partitions/availability:

```bash
sinfo
```

Common causes:
- wrong `#SBATCH --partition=...`
- requesting too many GPUs/CPUs/memory for the chosen partition
- cluster is busy

## GPU not visible inside the container

Run the provided sanity job and inspect its output:

```bash
sbatch slurm/dgx1/00_test_container_1gpu.sbatch
cat dgx_test_container_<JOBID>.out
```

Things to verify:
- Your SLURM job actually requested GPUs (`#SBATCH --gres=gpu:<n>`).
- `CUDA_VISIBLE_DEVICES` is set inside the job (SLURM sets it).
- Your `docker run` uses `--gpus '"device='${CUDA_VISIBLE_DEVICES}'"'` (this repo’s scripts do).

## Permission problems writing checkpoints/results

This is almost always UID/GID mismatch between host and container.

Fix:
- Rebuild the image using the build args in `README.md` / `INSTRUCTIONS.md` so the container user matches your host `id`.

## My job hit the wall-clock limit

This repo is designed to handle time limits:
- SLURM sends SIGUSR1 (10 minutes early) and SIGTERM at the end.
- `src/train.py` traps signals, writes `ckpt_last.pt`, and exits cleanly.

For long experiments, use the chaining pattern:

```bash
./slurm/dgx1/chain_submit.sh my_long_run 3
```

## I lost data in scratch

Scratch is designed to be temporary. Treat it as a cache:
- stage-in datasets at job start (if feasible)
- stage-out results/checkpoints frequently

Templates:
- `scripts/stage_in_dataset.sh`
- `scripts/stage_out_results.sh`
