import argparse
import json
import os
import subprocess
from pathlib import Path

def load_grid(path: Path):
    grid = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(grid, list) and len(grid) > 0
    return grid

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="configs/grid.json")
    p.add_argument("--trial_id", type=int, required=True)
    p.add_argument("--outroot", required=True, help="Root output dir; each trial gets its own subdir.")
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--dataset_size", type=int, default=200_000)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    grid = load_grid(Path(args.grid))
    cfg = grid[args.trial_id % len(grid)]

    run_name = f"trial{args.trial_id:03d}_lr{cfg['lr']}_wd{cfg['wd']}_do{cfg['dropout']}"
    outdir = Path(args.outroot) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.train",
        "--outdir", str(outdir),
        "--run_name", run_name,
        "--max_steps", str(args.max_steps),
        "--batch_size", str(args.batch_size),
        "--dataset_size", str(args.dataset_size),
        "--image_size", str(args.image_size),
        "--num_classes", str(args.num_classes),
        "--num_workers", str(args.num_workers),
        "--lr", str(cfg["lr"]),
        "--wd", str(cfg["wd"]),
        "--dropout", str(cfg["dropout"]),
        "--checkpoint_every", "500",
        "--log_every", "50",
        "--resume", "auto",
    ]
    if args.amp:
        cmd.append("--amp")

    print("[sweep] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    import sys
    main()
