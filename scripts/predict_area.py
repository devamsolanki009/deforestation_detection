"""Detect forest loss in one area: fetch imagery, run the model, report hectares.

The first run for an area and pair of years requests pixels from Earth Engine and uses
quota; later runs read the cache under data/app_cache.

Run from the repo root:
    python -m scripts.predict_area                       default demo box, 2020 -> 2022
    python -m scripts.predict_area --bbox -52.28 -6.81 -52.24 -6.77 --before 2021 --after 2023
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import BBox, ConfigError
from src.inference.area import comparison_image, false_color, loss_overlay, save_geotiff
from src.inference.detector import DEFAULT_APP_CONFIG, Detector, load_app_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app-config", type=Path, default=DEFAULT_APP_CONFIG)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--before", type=int, default=2020)
    parser.add_argument("--after", type=int, default=2022)
    args = parser.parse_args()

    app = load_app_config(args.app_config)
    detector = Detector(app)
    bbox = BBox(*args.bbox) if args.bbox else app.default_bbox
    try:
        request = detector.validate(bbox, args.before, args.after)
    except ConfigError as exc:
        parser.error(str(exc))
    print(f"Area {request.grid.width} x {request.grid.height} px ({request.grid.area_km2:.1f} km2), "
          f"{args.before} -> {args.after}, model {json.dumps(detector.model_summary)}", flush=True)

    started = time.perf_counter()
    result, stack, bands, fetch_info = detector.run(
        request, progress=lambda done, total: print(f"  fetched tile {done}/{total}", flush=True))
    total_seconds = time.perf_counter() - started

    output_dir = app.outputs_dir / f"{args.before}_{args.after}_{request.grid.key()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_geotiff(output_dir / "loss.tif", result)
    after_image = false_color(stack, bands, args.after)
    plt.imsave(output_dir / f"false_color_{args.before}.png", false_color(stack, bands, args.before))
    plt.imsave(output_dir / f"false_color_{args.after}.png", after_image)
    plt.imsave(output_dir / "loss_overlay.png", loss_overlay(after_image, result.loss))
    plt.imsave(output_dir / "vs_hansen.png", comparison_image(after_image, result))

    summary = {
        "bbox": vars(bbox), "before": args.before, "after": args.after,
        "grid": {"crs": request.grid.crs, "left": request.grid.left, "top": request.grid.top,
                 "width": request.grid.width, "height": request.grid.height},
        "model": detector.model_summary, "hectares": result.hectares, "quality": result.quality,
        "warnings": result.warnings, "fetch": fetch_info, "total_seconds": round(total_seconds, 1),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
