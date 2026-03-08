# Evaluation and inference template

This repo includes a single-node batch-scoring path for checkpoints.

## What it does

- loads a checkpoint (`ckpt_last.pt` or a chosen checkpoint file)
- rebuilds the model from checkpoint metadata
- runs a batched evaluation loop
- writes per-batch metrics to `eval_metrics.jsonl`
- writes `eval_summary.json`
- optionally writes per-example predictions to `predictions.jsonl`

## Entry points

- Python entrypoint: `python -m src.eval`
- DGX-1 job script: `slurm/dgx1/20_eval_1gpu.sbatch`
- DGX-H100 job script: `slurm/dgxh100/20_eval_1gpu.sbatch`

## Typical usage

Submit against an existing run directory:

```bash
sbatch --export=ALL,RUN_REF=<RUN_DIR> slurm/dgx1/20_eval_1gpu.sbatch
```

or on DGX-H100:

```bash
sbatch --export=ALL,RUN_REF=<RUN_DIR> slurm/dgxh100/20_eval_1gpu.sbatch
```

The scripts assume checkpoints live under `runs/<RUN_DIR>/checkpoints/`.

## Why this matters

Evaluation is one of the most common follow-on jobs after training. It is lower risk than training, but users still need the same cluster mechanics:

- run through SLURM
- mount scratch correctly
- pick a checkpoint
- write outputs back into the run tree

This template keeps that pattern explicit and reusable.
