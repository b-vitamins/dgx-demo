import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")

def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    # Prefer ckpt_last.pt, otherwise choose the highest step checkpoint.
    last = ckpt_dir / "ckpt_last.pt"
    if last.exists():
        return last

    pattern = re.compile(r"ckpt_step_(\d+)\.pt$")
    best_step = -1
    best_path = None
    for p in ckpt_dir.glob("ckpt_step_*.pt"):
        m = pattern.search(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = p
    return best_path

def human_bytes(n: int) -> str:
    # minimal, not fancy
    units = ["B","KB","MB","GB","TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024
    return f"{x:.2f}TB"
