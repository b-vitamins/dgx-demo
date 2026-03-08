# Real dataset template

This repo now supports a simple `imagefolder` dataset path for real data.

## Supported layout

Use a class-folder layout under your staged dataset root:

```text
mydataset/
  cats/
    0001.jpg
    0002.jpg
  dogs/
    0001.jpg
    0002.jpg
```

The loader resizes images to `--image_size`, converts them to RGB, and maps class-directory names to integer labels.

## Train on a staged dataset

DGX-1:

```bash
sbatch --export=ALL,DATA_ROOT=/localscratch/$USER/datasets/mydataset slurm/dgx1/21_train_imagefolder_1gpu.sbatch
```

DGX-H100:

```bash
sbatch --export=ALL,DATA_ROOT=/raid/$USER/datasets/mydataset slurm/dgxh100/21_train_imagefolder_1gpu.sbatch
```

## Evaluate a checkpoint on a real dataset root

Use `src.eval` directly when you want a different dataset root than the one stored in the training config:

```bash
python -m src.eval \
  --checkpoint /path/to/ckpt_last.pt \
  --outdir /path/to/eval_out \
  --dataset_type imagefolder \
  --data_root /path/to/val_or_test_split
```

## When this template is enough

- classification data already organized by class directory
- a single-node job is enough
- you want a low-friction path from staged files to a working run

## When to replace it

Replace this loader if your real data uses:

- a manifest or metadata table
- segmentation masks
- sequence or video data
- multimodal inputs
- dataset-specific augmentations or preprocessing
