import argparse
import json
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_dataset
from .models import TinyCNN
from .utils import append_jsonl, atomic_write_json, ensure_dir, now_iso


def amp_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dataset_type", default="synthetic", choices=["synthetic", "imagefolder"])
    parser.add_argument("--data_root", default="")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dataset_size", type=int, default=20_000)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--profile_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    metrics_path = outdir / "profile_metrics.jsonl"
    summary_path = outdir / "profile_summary.json"

    dataset, resolved_num_classes = build_dataset(
        dataset_type=args.dataset_type,
        data_root=args.data_root,
        dataset_size=args.dataset_size,
        image_size=args.image_size,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("DataLoader has zero batches. Reduce --batch_size or provide more data.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN(num_classes=resolved_num_classes, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler(
        amp_device_type(),
        enabled=args.amp and torch.cuda.is_available(),
    )

    total_steps = args.warmup_steps + args.profile_steps
    data_iter = iter(loader)
    collected = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for step in tqdm(range(total_steps), desc="profile"):
        data_wait_start = time.perf_counter()
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        data_time = time.perf_counter() - data_wait_start

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=amp_device_type(), enabled=scaler.is_enabled()):
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start

        if step < args.warmup_steps:
            continue

        global_batch = int(yb.shape[0])
        total_time = data_time + step_time
        samples_per_sec = global_batch / max(1e-9, total_time)
        max_gpu_mem_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available()
            else 0.0
        )
        row = {
            "time": now_iso(),
            "step": step - args.warmup_steps,
            "data_time_s": data_time,
            "step_time_s": step_time,
            "total_time_s": total_time,
            "samples_per_sec": samples_per_sec,
            "loss": float(loss.item()),
            "batch_size": global_batch,
            "max_gpu_mem_mb": max_gpu_mem_mb,
        }
        append_jsonl(metrics_path, row)
        collected.append(row)

    avg_data = sum(r["data_time_s"] for r in collected) / len(collected)
    avg_step = sum(r["step_time_s"] for r in collected) / len(collected)
    avg_total = sum(r["total_time_s"] for r in collected) / len(collected)
    avg_samples = sum(r["samples_per_sec"] for r in collected) / len(collected)
    max_mem = max(r["max_gpu_mem_mb"] for r in collected) if collected else 0.0

    atomic_write_json(summary_path, {
        "time": now_iso(),
        "dataset_type": args.dataset_type,
        "data_root": args.data_root,
        "dataset_size": len(dataset),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "warmup_steps": args.warmup_steps,
        "profile_steps": args.profile_steps,
        "avg_data_time_s": avg_data,
        "avg_step_time_s": avg_step,
        "avg_total_time_s": avg_total,
        "avg_samples_per_sec": avg_samples,
        "max_gpu_mem_mb": max_mem,
        "device": str(device),
    })

    print(json.dumps({
        "dataset_type": args.dataset_type,
        "dataset_size": len(dataset),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "avg_data_time_s": avg_data,
        "avg_step_time_s": avg_step,
        "avg_total_time_s": avg_total,
        "avg_samples_per_sec": avg_samples,
        "max_gpu_mem_mb": max_mem,
        "outdir": str(outdir),
    }, indent=2))


if __name__ == "__main__":
    main()
