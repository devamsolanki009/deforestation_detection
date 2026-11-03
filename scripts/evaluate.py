"""Tune post-processing on val, then score the held-out test region.

Steps:
1. Predict every val and test patch, and stitch each split into one map, together
   with a forest mask built from Hansen data from before the period.
2. For each pipeline (with and without the forest mask), sweep thresholds on val with
   small detections removed, and keep the threshold with the best F1.
3. Score test with each pipeline at its val threshold. The primary pipeline is the one
   with the better val F1; test plays no part in choosing it.
4. Compare predicted and labeled hectares, and save per-patch errors for the plots.

Run from the repo root:
    python -m scripts.evaluate --run unet_resnet34_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from src.config import DEFAULT_CONFIG, load_config
from src.data.dataset import PatchDataset
from src.data.gee import NODATA
from src.data.labels import IGNORE, LOSS, forest_at_start
from src.data.patches import read_cell, source_prefixes
from src.inference.mosaic import MosaicBuilder, confusion_counts, predict_loss, scores
from src.inference.predict import load_checkpoint, predict_patches

# Finer steps at the low end: with the forest mask, the best threshold can sit below 0.05.
THRESHOLDS = np.round(np.concatenate([np.arange(0.01, 0.05, 0.01), np.arange(0.05, 0.951, 0.05)]), 2)
MIN_DETECTION_HA = 1.0
SQUARE_M_PER_HA = 10_000
PIPELINES = {"no_forest_mask": False, "forest_mask": True}


def predict_split(model, checkpoint, split: str, config, area: str, device):
    """Stitch predictions, labels and the forest mask for one split."""
    patch_dir = config.paths.patches_dir / area
    prefixes = source_prefixes(config, area)
    manifest = json.loads((config.paths.raw_dir / f"{prefixes[0]}.manifest.json").read_text(encoding="utf-8"))
    bands = manifest["bands"]
    raw_files = sorted(p for prefix in prefixes for p in config.paths.raw_dir.glob(f"{prefix}*.tif")
                       if not p.name.endswith(".local.tif"))

    dataset = PatchDataset(patch_dir, split, checkpoint["band_stats"], NODATA)
    builder = MosaicBuilder(dataset.records, config.patch.size, config.export.scale_m)
    sources = [rasterio.open(path) for path in raw_files]
    try:
        for record, probability, label in predict_patches(model, dataset, device):
            # Patches keep only the label, so the forest mask comes from the raw Hansen bands.
            stack = read_cell(sources, int(float(record["left"])), int(float(record["top"])), config.patch.size,
                              config.export.scale_m, len(bands), manifest["nodata"])
            forest = forest_at_start(stack[bands.index("lossyear")], stack[bands.index("treecover2000")],
                                     config.years.before, config.labels.forest_min_treecover, manifest["nodata"])
            builder.add(record, probability, label, forest)
    finally:
        for source in sources:
            source.close()
    print(f"{split}: {len(dataset)} patches, map {builder.mosaic.label.shape}", flush=True)
    return builder.mosaic, dataset.records


def sweep(mosaic, min_pixels: int, use_forest_mask: bool) -> list[dict]:
    return [
        {"threshold": float(t), **scores(*confusion_counts(predict_loss(mosaic, t, min_pixels, use_forest_mask), mosaic.label))}
        for t in THRESHOLDS
    ]


def hectares(mosaic, predicted: np.ndarray, pixel_ha: float) -> dict[str, float]:
    """Predicted area split by label class, so area in ignored pixels stays visible."""
    label = mosaic.label
    return {
        "labeled_loss_ha": float(np.count_nonzero(label == LOSS) * pixel_ha),
        "predicted_loss_ha_on_labeled_pixels": float(np.count_nonzero(predicted & (label != IGNORE)) * pixel_ha),
        "predicted_loss_ha_on_ignored_pixels": float(np.count_nonzero(predicted & (label == IGNORE)) * pixel_ha),
        "correctly_predicted_ha": float(np.count_nonzero(predicted & (label == LOSS)) * pixel_ha),
    }


def per_patch_errors(mosaic, records: list[dict], predicted: np.ndarray, size: int) -> list[dict]:
    rows = []
    for record in records:
        cells = mosaic.patch_slice(record, size)
        tp, fp, fn = confusion_counts(predicted[cells], mosaic.label[cells])
        rows.append({"patch_id": record["patch_id"], "left": record["left"], "top": record["top"],
                     "tp": tp, "fp": fp, "fn": fn})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", required=True, help="folder name under outputs/runs")
    parser.add_argument("--area", default="full")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.paths.outputs_dir / "runs" / args.run
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    model, checkpoint = load_checkpoint(run_dir / "best.pt", device)
    scale = config.export.scale_m
    pixel_ha = scale * scale / SQUARE_M_PER_HA
    min_pixels = round(MIN_DETECTION_HA / pixel_ha)

    # Val chooses thresholds and the primary pipeline; test does not influence either.
    val, _ = predict_split(model, checkpoint, "val", config, args.area, device)
    pipelines = {}
    for name, use_mask in PIPELINES.items():
        rows = sweep(val, min_pixels, use_mask)
        write_csv(output_dir / f"val_threshold_sweep_{name}.csv", rows)
        best = max(rows, key=lambda row: row["f1"])
        pipelines[name] = {"use_forest_mask": use_mask, "threshold": best["threshold"],
                           "val_best_f1": best["f1"], "splits": {}}
        print(f"{name}: best val threshold {best['threshold']}, F1 {best['f1']:.4f}", flush=True)
    primary = max(pipelines, key=lambda name: pipelines[name]["val_best_f1"])

    test, test_records = predict_split(model, checkpoint, "test", config, args.area, device)
    for split, mosaic in (("val", val), ("test", test)):
        mosaic.save(output_dir / f"{split}_mosaic.npz")
        for name, pipeline in pipelines.items():
            predicted = predict_loss(mosaic, pipeline["threshold"], min_pixels, pipeline["use_forest_mask"])
            unfiltered = predict_loss(mosaic, pipeline["threshold"], 1, pipeline["use_forest_mask"])
            pipeline["splits"][split] = {
                "metrics": scores(*confusion_counts(predicted, mosaic.label)),
                "metrics_without_size_filter": scores(*confusion_counts(unfiltered, mosaic.label)),
                "hectares": hectares(mosaic, predicted, pixel_ha),
            }
            if split == "test" and name == primary:
                write_csv(output_dir / "test_patch_errors.csv",
                          per_patch_errors(mosaic, test_records, predicted, config.patch.size))

    results = {"run": args.run, "checkpoint_epoch": checkpoint["epoch"], "min_detection_pixels": min_pixels,
               "primary_pipeline": primary, "pipelines": pipelines}
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
