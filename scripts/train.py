"""Train a segmentation model on the patches.

Run from the repo root:
    python -m scripts.train --train-config configs/train/unet_resnet34.yaml
    python -m scripts.train --train-config configs/train/unet_resnet34.yaml --timing-run

A timing run trains 2 short epochs into a separate run folder, to measure speed and
memory before committing to a full run.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.data.dataset import (  # noqa: E402
    PatchDataset, load_or_compute_band_stats, loss_patch_sampler, worker_init,
)
from src.data.gee import NODATA  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.config import load_train_config  # noqa: E402
from src.training.losses import DiceFocalLoss  # noqa: E402
from src.training.loop import evaluate, train_one_epoch  # noqa: E402

TIMING_EPOCHS = 2
TIMING_STEPS = 30
CURVE_METRICS = ("loss", "f1", "iou", "precision", "recall")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_curves(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, len(CURVE_METRICS), figsize=(4 * len(CURVE_METRICS), 3.2))
    for axis, metric in zip(axes, CURVE_METRICS):
        if metric == "loss":
            axis.plot(epochs, [row["train_loss"] for row in history], label="train")
        axis.plot(epochs, [row[f"val_{metric}"] for row in history], label="val")
        axis.set_title(metric if metric == "loss" else f"val {metric} (loss class)")
        axis.set_xlabel("epoch")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--timing-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    train_config = load_train_config(args.train_config)
    seed_everything(train_config.seed)
    if not torch.cuda.is_available():
        print("CUDA is not available; training on CPU would be impractically slow.")
        return 1
    device = torch.device("cuda")

    run_name = f"{train_config.run_name}_timing" if args.timing_run else train_config.run_name
    run_dir = config.paths.outputs_dir / "runs" / run_name
    if (run_dir / "best.pt").exists() and not args.timing_run:
        print(f"{run_dir} already has a trained model. Rename run_name or move the folder first.")
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.train_config, run_dir / "train_config.yaml")

    patch_dir = config.paths.patches_dir / train_config.area
    stats = load_or_compute_band_stats(patch_dir, NODATA)
    train_set = PatchDataset(patch_dir, "train", stats, NODATA, train=True)
    val_set = PatchDataset(patch_dir, "val", stats, NODATA)

    data = train_config.data
    loader_args = dict(
        batch_size=data.batch_size, num_workers=data.num_workers, pin_memory=True,
        persistent_workers=data.num_workers > 0, worker_init_fn=worker_init,
    )
    sampler = loss_patch_sampler(train_set.records, data.oversample_min_loss_fraction,
                                 data.oversample_weight, train_config.seed)
    train_loader = DataLoader(train_set, sampler=sampler, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = build_model(train_config.model, in_channels=len(stats["mean"])).to(device)
    loss_fn = DiceFocalLoss(train_config.loss)
    epochs = TIMING_EPOCHS if args.timing_run else train_config.optim.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.optim.lr,
                                  weight_decay=train_config.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.amp)
    max_steps = TIMING_STEPS if args.timing_run else None

    print(f"Run {run_name}: {len(train_set)} train, {len(val_set)} val patches, {epochs} epochs", flush=True)
    history, best_f1, stale_epochs = [], -1.0, 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device,
                                     train_config.amp, max_steps)
        train_seconds = time.perf_counter() - start
        val = evaluate(model, val_loader, loss_fn, device, train_config.amp, train_config.threshold, max_steps)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()},
               "lr": optimizer.param_groups[0]["lr"], "train_seconds": train_seconds,
               "epoch_seconds": time.perf_counter() - start}
        history.append(row)
        print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  val loss {val['loss']:.4f}  "
              f"F1 {val['f1']:.4f}  IoU {val['iou']:.4f}  P {val['precision']:.4f}  R {val['recall']:.4f}  "
              f"{row['epoch_seconds']:.0f}s", flush=True)

        checkpoint = {"model": model.state_dict(), "epoch": epoch, "val": val, "band_stats": stats,
                      "model_config": vars(train_config.model), "threshold": train_config.threshold}
        torch.save(checkpoint, run_dir / "last.pt")
        if val["f1"] > best_f1:
            best_f1, stale_epochs = val["f1"], 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale_epochs += 1

        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        save_curves(history, run_dir / "curves.png")

        if stale_epochs >= train_config.optim.early_stopping_patience:
            print(f"Stopping early: val F1 has not improved for {stale_epochs} epochs.")
            break

    summary = {"best_val_f1": best_f1, "epochs_run": len(history),
               "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
