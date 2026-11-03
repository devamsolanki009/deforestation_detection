"""Check test errors against the raw Hansen bands and the visible optical change.

False alarms: which Hansen loss year sits under them? Loss dated to a later year can
mean Hansen recorded the clearing late, so the label says no loss.
Misses: did the vegetation visibly drop? A miss with no NDVI drop may be loss that
optical imagery cannot show (degradation, fire, or a label error).

Needs scripts/evaluate.py first. Run from the repo root:
    python -m scripts.diagnose_errors --run unet_resnet34_baseline
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

from src.config import DEFAULT_CONFIG, load_config
from src.data.dataset import read_metadata
from src.data.labels import IGNORE, LOSS
from src.data.patches import image_band_indices, read_cell, source_prefixes
from src.inference.mosaic import Mosaic, predict_loss

NDVI_DROP_VISIBLE = 0.1  # an NDVI fall at least this large counts as a visible change


def lossyear_group(code: int, before: int, after: int) -> str:
    if code == 0:
        return "no Hansen loss"
    year = 2000 + code
    if year < before:
        return f"lost before {before}"
    if year > after:
        return f"lost after {after} ({year})"
    return f"lost {year}"


def ndvi(image: np.ndarray, names: list[str], year: int) -> np.ndarray:
    nir, red = image[names.index(f"B8_{year}")].astype(float), image[names.index(f"B4_{year}")].astype(float)
    return (nir - red) / np.maximum(nir + red, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", required=True)
    parser.add_argument("--area", default="full")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--pipeline", help="defaults to the primary pipeline in metrics.json")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluation_dir = config.paths.outputs_dir / "runs" / args.run / "evaluation"
    metrics = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    before, after, size = config.years.before, config.years.after, config.patch.size

    prefixes = source_prefixes(config, args.area)
    manifest = json.loads((config.paths.raw_dir / f"{prefixes[0]}.manifest.json").read_text(encoding="utf-8"))
    bands = manifest["bands"]
    names = [bands[i] for i in image_band_indices(bands)]
    raw_files = sorted(p for prefix in prefixes for p in config.paths.raw_dir.glob(f"{prefix}*.tif")
                       if not p.name.endswith(".local.tif"))

    pipeline = metrics["pipelines"][args.pipeline or metrics["primary_pipeline"]]
    mosaic = Mosaic.load(evaluation_dir / f"{args.split}_mosaic.npz")
    predicted_map = predict_loss(mosaic, pipeline["threshold"], metrics["min_detection_pixels"],
                                 pipeline["use_forest_mask"])
    records = read_metadata(config.paths.patches_dir / args.area, args.split)

    fp_years, drops = Counter(), {"tp": [], "fn": []}
    min_clear = []
    sources = [rasterio.open(path) for path in raw_files]
    try:
        for record in records:
            cells = mosaic.patch_slice(record, size)
            predicted, label = predicted_map[cells], mosaic.label[cells]
            stack = read_cell(sources, int(float(record["left"])), int(float(record["top"])), size,
                              config.export.scale_m, len(bands), manifest["nodata"])
            image = stack[image_band_indices(bands)]

            false_alarm = predicted & (label != LOSS) & (label != IGNORE)
            for code, count in zip(*np.unique(stack[bands.index("lossyear")][false_alarm], return_counts=True)):
                fp_years[lossyear_group(int(code), before, after)] += int(count)

            drop = ndvi(image, names, before) - ndvi(image, names, after)
            drops["tp"].append(drop[predicted & (label == LOSS)])
            drops["fn"].append(drop[~predicted & (label == LOSS)])
            clear = np.minimum(stack[bands.index(f"s2_clear_count_{before}")], stack[bands.index(f"s2_clear_count_{after}")])
            valid_clear = clear[clear != manifest["nodata"]]
            if valid_clear.size:
                min_clear.append(int(valid_clear.min()))
    finally:
        for source in sources:
            source.close()

    total_fp = sum(fp_years.values())
    visible = {}
    for kind, parts in drops.items():
        values = np.concatenate(parts)
        visible[kind] = {
            "pixels": int(values.size),
            "median_ndvi_drop": float(np.median(values)),
            "share_with_visible_drop": float(np.mean(values >= NDVI_DROP_VISIBLE)),
        }

    report = {
        "false_alarm_pixels_by_hansen_loss_year": {
            group: {"pixels": count, "share": count / max(1, total_fp)} for group, count in fp_years.most_common()
        },
        "ndvi_drop_on_labeled_loss": visible,
        "split": args.split,
        "lowest_clear_image_count": min(min_clear),
    }
    (evaluation_dir / f"error_diagnosis_{args.split}_{args.pipeline or metrics['primary_pipeline']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
