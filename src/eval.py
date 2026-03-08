import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SyntheticImageDataset
from .models import TinyCNN
from .utils import append_jsonl, atomic_write_json, ensure_dir, now_iso


def normalize_state_dict_keys(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    return state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--write_predictions", action="store_true")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_cfg = ckpt.get("cfg", {})

    image_size = args.image_size or int(train_cfg.get("image_size", 32))
    num_classes = args.num_classes or int(train_cfg.get("num_classes", 10))
    dataset_size = args.dataset_size or int(train_cfg.get("dataset_size", 50_000))
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 123))
    dropout = float(train_cfg.get("dropout", 0.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = ensure_dir(args.outdir)
    metrics_path = outdir / "eval_metrics.jsonl"
    predictions_path = outdir / "predictions.jsonl"
    summary_path = outdir / "eval_summary.json"

    dataset = SyntheticImageDataset(
        size=dataset_size,
        image_size=image_size,
        num_classes=num_classes,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    model = TinyCNN(num_classes=num_classes, dropout=dropout).to(device)
    model.load_state_dict(normalize_state_dict_keys(ckpt["model"]))
    model.eval()

    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    max_batches = args.max_batches if args.max_batches > 0 else None

    with torch.inference_mode():
        for batch_idx, (xb, yb) in enumerate(tqdm(loader, desc="eval")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            preds = torch.argmax(logits, dim=1)

            batch_size = int(yb.shape[0])
            total_examples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == yb).sum().item())

            append_jsonl(metrics_path, {
                "time": now_iso(),
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "loss": float(loss.item()),
                "correct": int((preds == yb).sum().item()),
            })

            if args.write_predictions:
                start_idx = batch_idx * args.batch_size
                for offset, (target, pred) in enumerate(zip(yb.tolist(), preds.tolist())):
                    append_jsonl(predictions_path, {
                        "sample_idx": start_idx + offset,
                        "target": int(target),
                        "pred": int(pred),
                        "correct": bool(target == pred),
                    })

    avg_loss = total_loss / max(1, total_examples)
    accuracy = total_correct / max(1, total_examples)

    atomic_write_json(summary_path, {
        "time": now_iso(),
        "checkpoint": str(ckpt_path),
        "outdir": str(outdir),
        "dataset_size": dataset_size,
        "num_examples": total_examples,
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "write_predictions": args.write_predictions,
    })

    print(json.dumps({
        "checkpoint": str(ckpt_path),
        "num_examples": total_examples,
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "outdir": str(outdir),
    }, indent=2))


if __name__ == "__main__":
    main()
