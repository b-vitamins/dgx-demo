# SLURM scripts

This directory is organized by cluster:

- `slurm/dgx1/`: DGX-1 (`nvidia-dgx`) job scripts
- `slurm/dgxh100/`: DGX-H100 (`dgxh100`) job scripts

The two folders mirror the same basic workflows where possible:

- `00_*`: container sanity check
- `01_*`: 1-GPU training with signal handling
- `02_*`: chained continuation run
- `03_*`: single-node DDP run
- `10_*`: SLURM array sweep
- `11_*`: packed multi-GPU sweep
- `chain_submit.sh`: helper for chained runs
