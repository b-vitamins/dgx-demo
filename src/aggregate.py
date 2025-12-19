import argparse
import json
from pathlib import Path

def best_loss(metrics_file: Path):
    if not metrics_file.exists():
        return None
    best = None
    for line in metrics_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        loss = obj.get("loss")
        step = obj.get("step")
        if loss is None:
            continue
        if best is None or loss < best[0]:
            best = (float(loss), int(step))
    return best

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", required=True, help="Directory containing trial subfolders.")
    args = p.parse_args()

    root = Path(args.sweep_root)
    rows = []
    for trial in sorted(root.glob("*")):
        if not trial.is_dir():
            continue
        cfg_path = trial / "config.json"
        metrics_path = trial / "metrics.jsonl"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        best = best_loss(metrics_path)
        if best is None:
            continue
        rows.append({
            "trial": trial.name,
            "best_loss": best[0],
            "best_step": best[1],
            "lr": cfg.get("lr"),
            "wd": cfg.get("wd"),
            "dropout": cfg.get("dropout"),
        })

    rows.sort(key=lambda r: r["best_loss"])
    print("trial,best_loss,best_step,lr,wd,dropout")
    for r in rows:
        print(f"{r['trial']},{r['best_loss']:.6f},{r['best_step']},{r['lr']},{r['wd']},{r['dropout']}")

if __name__ == "__main__":
    main()
