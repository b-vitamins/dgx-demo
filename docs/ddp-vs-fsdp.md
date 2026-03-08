# DDP vs FSDP

This repo now includes both single-node DDP and single-node FSDP examples. Both are valid on SERC DGX systems, but they solve different problems.

## Quick rule of thumb

- Use `DDP` when the model already fits comfortably on each GPU and you want the simplest distributed baseline.
- Use `FSDP` when the model or optimizer state is becoming memory-bound on each GPU and you still want to stay within a single node.

## What the repo provides

- DGX-1 DDP: `slurm/dgx1/03_train_4gpu_ddp_12h_signal.sbatch`
- DGX-1 FSDP: `slurm/dgx1/04_train_4gpu_fsdp_12h_signal.sbatch`
- DGX-H100 DDP: `slurm/dgxh100/03_train_2gpu_ddp_12h_signal.sbatch`
- DGX-H100 FSDP: `slurm/dgxh100/04_train_2gpu_fsdp_12h_signal.sbatch`

## Side-by-side comparison

| Topic | DDP | FSDP |
|---|---|---|
| Main benefit | Simple and robust | Lower per-GPU memory use |
| Model copy per GPU | Full replica | Sharded |
| Optimizer state per GPU | Full replica | Sharded |
| Performance profile | Usually simpler and often faster for smaller models | Extra communication overhead, better when memory is the bottleneck |
| Best use here | Standard single-node scaling | Larger single-node experiments that no longer fit cleanly under DDP |

## Checkpoint behavior in this repo

- Both paths still save `ckpt_last.pt` and periodic `ckpt_step_<N>.pt` files.
- The DDP path saves ordinary full checkpoints.
- The FSDP path saves a full, portable checkpoint on rank 0 after gathering model state and optimizer state out of the sharded runtime.
- Resume stays `--resume auto` for both, but FSDP has more checkpointing overhead because it has to reconstruct full state for saving.

## What breaks or changes when you move from DDP to FSDP

- Checkpoint save/load is no longer a trivial `model.state_dict()` on rank 0.
- Memory behavior improves, but communication overhead increases.
- Tiny models may get slower under FSDP and teach the wrong lesson. FSDP becomes useful when memory pressure is real.
- The same global batch size can imply different headroom because optimizer and parameter memory are handled differently.

## Practical way to compare them

1. Start with the DDP script for your cluster.
2. Keep the same dataset size, batch size, and wall-clock budget.
3. Run the matching FSDP script.
4. Compare:
   - whether DDP was close to out-of-memory
   - throughput
   - checkpoint overhead
   - stability at the target batch size

If DDP is already stable and comfortably within memory, it is usually the better default.
