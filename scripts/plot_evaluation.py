"""Draw test overlays and a failure gallery, and tag likely error causes.

Needs scripts/evaluate.py to have run first. Run from the repo root:
    python -m scripts.plot_evaluation --run unet_resnet34_baseline

Cause tags are rough heuristics on the pixels a patch got wrong, to point at what to
inspect; they are not verified causes:
    low_clear_count  few cloud-free Sentinel-2 images, so the optical composite is unreliable
    water            water-like spectra (NDWI above 0) in either year
    open_in_2020     already low vegetation (NDVI below 0.6) in 2020: pasture, crops, or,
                     for a miss, a label that may be wrong
    unclear          none of the above
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.data.labels import IGNORE, LOSS  # noqa: E402
from src.data.patches import image_band_indices, source_prefixes  # noqa: E402
from src.inference.mosaic import Mosaic, predict_loss  # noqa: E402

GALLERY_SIZE = 6
OVERLAY_SAMPLES = 6
LOW_CLEAR_COUNT = 5
OPEN_NDVI = 0.6
TAG_SHARE = 0.5      # share of a patch's error pixels that must match a tag
WATER_SHARE = 0.3    # water is rarely most of an error, so it needs less
FALSE_COLOR_MAX = 4000
SEED = 7

# Outcome colors: hit, false alarm, miss.
COLORS = {"tp": (1.0, 0.85, 0.0), "fp": (0.9, 0.1, 0.1), "fn": (0.0, 0.8, 1.0)}


def index(image: np.ndarray, names: list[str], a: str, b: str, year: int) -> np.ndarray:
    first, second = image[names.index(f"{a}_{year}")].astype(float), image[names.index(f"{b}_{year}")].astype(float)
    return (first - second) / np.maximum(first + second, 1)


def tag_errors(patch: dict, names: list[str], errors: np.ndarray, before: int, after: int) -> str:
    if not errors.any():
        return "none"
    image = patch["image"]
    clear = patch["clear_count"].min(axis=0)
    ndwi = np.maximum(index(image, names, "B3", "B8", before), index(image, names, "B3", "B8", after))
    ndvi_before = index(image, names, "B8", "B4", before)

    if np.median(clear[errors]) < LOW_CLEAR_COUNT:
        return "low_clear_count"
    if np.mean(ndwi[errors] > 0) > WATER_SHARE:
        return "water"
    if np.mean(ndvi_before[errors] < OPEN_NDVI) > TAG_SHARE:
        return "open_in_2020"
    return "unclear"


def false_color(image: np.ndarray, names: list[str], year: int) -> np.ndarray:
    bands = [image[names.index(f"{band}_{year}")] for band in ("B11", "B8", "B4")]
    return np.clip(np.dstack(bands) / FALSE_COLOR_MAX, 0, 1)


def outcome_overlay(image: np.ndarray, names: list[str], year: int, predicted: np.ndarray, label: np.ndarray):
    base = false_color(image, names, year).mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.8
    outcomes = {"tp": predicted & (label == LOSS),
                "fp": predicted & (label != LOSS) & (label != IGNORE),
                "fn": ~predicted & (label == LOSS)}
    for name, mask in outcomes.items():
        base[mask] = COLORS[name]
    return base


def draw(rows: list[dict], path: Path, title: str, names, before, after, stride_m) -> None:
    figure, axes = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)), squeeze=False)
    headers = [f"{before} false color", f"{after} false color", f"VH change (dB)",
               "yellow hit, red false alarm, cyan miss"]
    for row, item in enumerate(rows):
        image = item["patch"]["image"]
        vh = (image[names.index(f"VH_{after}")].astype(float) - image[names.index(f"VH_{before}")]) / 100
        panels = [false_color(image, names, before), false_color(image, names, after), vh,
                  outcome_overlay(image, names, after, item["predicted"], item["label"])]
        for column, panel in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(panel, cmap="RdBu" if column == 2 else None, vmin=-6 if column == 2 else None,
                        vmax=6 if column == 2 else None, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(headers[column], fontsize=9)
        axes[row, 0].set_ylabel(f"{item['tag']}\nFP {item['fp']}  FN {item['fn']}", fontsize=8)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", required=True)
    parser.add_argument("--area", default="full")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluation_dir = config.paths.outputs_dir / "runs" / args.run / "evaluation"
    metrics = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    patch_dir = config.paths.patches_dir / args.area
    manifest = json.loads((config.paths.raw_dir / f"{source_prefixes(config, args.area)[0]}.manifest.json")
                          .read_text(encoding="utf-8"))
    names = [manifest["bands"][i] for i in image_band_indices(manifest["bands"])]
    before, after, size = config.years.before, config.years.after, config.patch.size

    pipeline = metrics["pipelines"][metrics["primary_pipeline"]]
    mosaic = Mosaic.load(evaluation_dir / "test_mosaic.npz")
    predicted_map = predict_loss(mosaic, pipeline["threshold"], metrics["min_detection_pixels"],
                                 pipeline["use_forest_mask"])
    with (evaluation_dir / "test_patch_errors.csv").open(encoding="utf-8") as handle:
        errors = [{**row, "fp": int(row["fp"]), "fn": int(row["fn"]), "tp": int(row["tp"])} for row in csv.DictReader(handle)]

    # Tag every test patch, and total error pixels by tag across the whole test region.
    pixel_totals = {"fp": Counter(), "fn": Counter()}
    items = {}
    for row in errors:
        cells = mosaic.patch_slice(row, size)
        predicted, label = predicted_map[cells], mosaic.label[cells]
        patch = np.load(patch_dir / f"{row['patch_id']}.npz")
        item = {**row, "patch": patch, "predicted": predicted, "label": label}
        masks = {"fp": predicted & (label != LOSS) & (label != IGNORE), "fn": ~predicted & (label == LOSS)}
        for kind, mask in masks.items():
            tag = tag_errors(patch, names, mask, before, after)
            item[f"{kind}_tag"] = tag
            pixel_totals[kind][tag] += int(mask.sum())
        items[row["patch_id"]] = item

    worst_fp = sorted(items.values(), key=lambda item: -item["fp"])[:GALLERY_SIZE]
    worst_fn = sorted(items.values(), key=lambda item: -item["fn"])[:GALLERY_SIZE]
    stride_m = config.patch.stride * config.export.scale_m
    draw([{**i, "tag": i["fp_tag"]} for i in worst_fp], evaluation_dir / "gallery_false_alarms.png",
         "Test patches with the most false alarms (after cleaning)", names, before, after, stride_m)
    draw([{**i, "tag": i["fn_tag"]} for i in worst_fn], evaluation_dir / "gallery_misses.png",
         "Test patches with the most missed loss (after cleaning)", names, before, after, stride_m)

    with_loss = [i for i in items.values() if i["tp"] + i["fn"] > 0]
    rng = np.random.default_rng(SEED)
    samples = [with_loss[k] for k in rng.choice(len(with_loss), size=min(OVERLAY_SAMPLES, len(with_loss)), replace=False)]
    draw([{**i, "tag": "random test"} for i in samples], evaluation_dir / "test_overlays.png",
         "Random test patches that contain labeled loss", names, before, after, stride_m)

    summary = {
        kind: {tag: {"pixels": count, "share": count / max(1, sum(totals.values()))}
               for tag, count in totals.most_common()}
        for kind, totals in pixel_totals.items()
    }
    (evaluation_dir / "error_tags.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved figures in {evaluation_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
