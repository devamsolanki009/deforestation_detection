"""Report Sentinel-1 image counts per pixel for each composite year.

Runs at a coarse scale to keep Earth Engine usage low, so results describe the
region's coverage rather than exact counts along scene edges.

Run from the repo root:
    python -m scripts.check_s1_coverage
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import ee

from src.config import DEFAULT_CONFIG, load_config
from src.data import gee

# Grid used for the counts. Coarse on purpose: 1 km keeps the region near 12,000 pixels.
COUNT_SCALE_M = 1000
COUNT_CRS = "EPSG:32722"  # UTM 22S, which covers the study region


def scene_dates(collection: ee.ImageCollection) -> list[str]:
    """List acquisition days and passes from metadata, without reading any pixels."""
    rows = collection.reduceColumns(
        ee.Reducer.toList(3), ["system:time_start", "orbitProperties_pass", "relativeOrbitNumber_start"]
    ).get("list").getInfo()
    days = Counter(
        (datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d"), orbit_pass, orbit)
        for time_ms, orbit_pass, orbit in rows
    )
    return [f"{day} {orbit_pass[:4]} rel{orbit} ({scenes} scenes)"
            for (day, orbit_pass, orbit), scenes in sorted(days.items())]


def count_histogram(collection: ee.ImageCollection, geometry: ee.Geometry) -> dict[int, int]:
    """Histogram of per-pixel image counts, including pixels with no image at all."""
    counts = collection.select("VV").count().unmask(0).rename("count")
    histogram = counts.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geometry,
        crs=COUNT_CRS,
        scale=COUNT_SCALE_M,
        maxPixels=1_000_000,
    ).get("count").getInfo()
    return {int(float(count)): round(pixels) for count, pixels in sorted(histogram.items(), key=lambda kv: float(kv[0]))}


def summarize(histogram: dict[int, int]) -> dict:
    total = sum(histogram.values())
    weighted = sorted(histogram.items())
    cumulative, median = 0, None
    for count, pixels in weighted:
        cumulative += pixels
        if median is None and cumulative >= total / 2:
            median = count
    return {
        "pixels": total,
        "min": weighted[0][0],
        "median": median,
        "max": weighted[-1][0],
        "share_zero": sum(p for c, p in histogram.items() if c == 0) / total,
        "share_under_3": sum(p for c, p in histogram.items() if c < 3) / total,
        "histogram": {str(c): p for c, p in histogram.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    gee.initialize(config)
    geometry = gee.bbox_geometry(config.region.bbox)

    report: dict = {"scale_m": COUNT_SCALE_M, "region": config.region.name, "years": {}}
    for year in (config.years.before, config.years.after):
        start, end = gee.season_dates(config, year)
        season = gee.s1_collection(geometry, start, end)
        full_year = gee.s1_collection(geometry, ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year + 1, 1, 1))

        summary = summarize(count_histogram(season, geometry))
        dates = scene_dates(full_year)
        report["years"][str(year)] = {"season": summary, "full_year_dates": dates}

        print(f"\n=== {year}, months {config.season.start_month}-{config.season.end_month} ===")
        print(f"images per pixel: min {summary['min']}, median {summary['median']}, max {summary['max']}")
        print(f"share of pixels with 0 images: {summary['share_zero']:.2%}, under 3 images: {summary['share_under_3']:.2%}")
        print(f"histogram (images -> 1 km pixels): {summary['histogram']}")
        print(f"all acquisition days in {year}:")
        for line in dates:
            print(f"  {line}")

    output = config.paths.outputs_dir / "phase1" / "s1_coverage.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
