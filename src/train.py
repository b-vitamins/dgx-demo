import argparse
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .data import SyntheticImageDataset
from .models import TinyCNN
from .utils import ensure_dir, now_iso, append_jsonl, atomic_write_json, find_latest_checkpoint

# -----------------------
# SLURM time-limit hygiene
# -----------------------
_SHOULD_EXIT = False

def _handle_signal(signum, frame):
    global _SHOULD_EXIT
    _SHOULD_EXIT = True

def install_signal_handlers():
    # SLURM can send SIGUSR1 (configured via #SBATCH --signal=B:USR1@<seconds>)
    signal.signal(signal.SIGUSR1, _handle_signal)
    # SLURM typically sends SIGTERM at time limit, then SIGKILL later.
    signal.signal(signal.SIGTERM, _handle_signal)

# -----------------------
# DDP helpers
# -----------------------
def ddp_is_active() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def ddp_setup():
    if not ddp_is_active():
        return
    torch.cuda.set_device(ddp_local_rank())
    dist.init_process_group(backend="nccl", init_method="env://")

def ddp_cleanup():
    if ddp_is_active():
        dist.destroy_process_group()

def is_main_process() -> bool:
    return ddp_rank() == 0

def barrier():
    if ddp_is_active():
        dist.barrier()

# -----------------------
# Training
# -----------------------
@dataclass
class TrainConfig:
    outdir: str
    run_name: str
    max_steps: int
    batch_size: int
    lr: float
    wd: float
    dropout: float
    image_size: int
    num_classes: int
    dataset_size: int
    num_workers: int
    log_every: int
    checkpoint_every: int
    seed: int
    amp: bool
    resume: str  # "auto" or path or "none"

def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))

def save_checkpoint(ckpt_path: Path, model, optimizer, scaler, step: int, epoch: int, cfg: TrainConfig):
    ckpt = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": asdict(cfg),
        "time": now_iso(),
    }
    tmp = ckpt_path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(ckpt_path)

def load_checkpoint(path: Path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--run_name", default="demo")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256, help="Per-process batch size (DDP global batch = batch_size * world_size).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--dataset_size", type=int, default=200_000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default="auto", help='auto | none | /path/to/ckpt.pt')

    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))

    install_signal_handlers()
    ddp_setup()

    rank = ddp_rank()
    world = ddp_world_size()
    local_rank = ddp_local_rank()

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    outdir = ensure_dir(cfg.outdir)
    ckpt_dir = ensure_dir(outdir / "checkpoints")
    metrics_path = outdir / "metrics.jsonl"
    summary_path = outdir / "summary.json"

    if is_main_process():
        atomic_write_json(outdir / "config.json", asdict(cfg))

    # Dataset & loader
    dataset = SyntheticImageDataset(size=cfg.dataset_size, image_size=cfg.image_size, num_classes=cfg.num_classes, seed=cfg.seed)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if ddp_is_active() else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
        drop_last=True,
    )

    # Model
    model = TinyCNN(num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    if ddp_is_active():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and torch.cuda.is_available())

    # Resume
    step = 0
    start_epoch = 0
    if cfg.resume != "none":
        resume_path: Optional[Path] = None
        if cfg.resume == "auto":
            resume_path = find_latest_checkpoint(ckpt_dir)
        else:
            resume_path = Path(cfg.resume)

        if resume_path is not None and resume_path.exists():
            # Only main loads first, then broadcast weights via DDP; but simplest is load on all ranks.
            step, start_epoch = load_checkpoint(resume_path, model, optimizer, scaler)
            if is_main_process():
                print(f"[resume] loaded {resume_path} (step={step}, epoch={start_epoch})")
        elif is_main_process():
            print("[resume] no checkpoint found; starting fresh")

    # Training loop: step-based, not epoch-based (better for time-limited jobs)
    model.train()
    t0 = time.time()
    epoch = start_epoch
    pbar = tqdm(total=cfg.max_steps, initial=step, disable=not is_main_process(), desc=f"train (world={world})")

    while step < cfg.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for xb, yb in loader:
            if step >= cfg.max_steps:
                break

            if _SHOULD_EXIT:
                # Time-limit or termination signal received: checkpoint and exit cleanly.
                if is_main_process():
                    ckpt_last = ckpt_dir / "ckpt_last.pt"
                    save_checkpoint(ckpt_last, model, optimizer, scaler, step=step, epoch=epoch, cfg=cfg)
                    print(f"[signal] received; saved {ckpt_last} and exiting at step={step}")
                barrier()
                ddp_cleanup()
                return

            xb = xb.to(device, non_blocking=True)
            yb = torch.tensor(yb, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(xb)
                loss = torch.nn.functional.cross_entropy(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging (main only)
            if is_main_process() and (step % cfg.log_every == 0):
                elapsed = time.time() - t0
                ips = (cfg.batch_size * world) / max(1e-9, (elapsed / max(1, step - (start_epoch * len(loader)))))
                append_jsonl(metrics_path, {
                    "time": now_iso(),
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu().item()),
                    "world_size": world,
                    "global_batch": cfg.batch_size * world,
                })

            # Checkpointing
            if (step > 0) and (step % cfg.checkpoint_every == 0):
                if is_main_process():
                    ckpt_step = ckpt_dir / f"ckpt_step_{step}.pt"
                    ckpt_last = ckpt_dir / "ckpt_last.pt"
                    save_checkpoint(ckpt_step, model, optimizer, scaler, step=step, epoch=epoch, cfg=cfg)
                    save_checkpoint(ckpt_last, model, optimizer, scaler, step=step, epoch=epoch, cfg=cfg)
                barrier()

            step += 1
            if is_main_process():
                pbar.update(1)

        epoch += 1

    # Final save
    if is_main_process():
        ckpt_last = ckpt_dir / "ckpt_last.pt"
        save_checkpoint(ckpt_last, model, optimizer, scaler, step=step, epoch=epoch, cfg=cfg)
        atomic_write_json(summary_path, {
            "time": now_iso(),
            "final_step": step,
            "final_epoch": epoch,
            "outdir": str(outdir),
            "world_size": world,
        })
        print(f"[done] saved final checkpoint to {ckpt_last}")

    barrier()
    ddp_cleanup()

if __name__ == "__main__":
    main()
