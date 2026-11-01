"""Report how many Hansen loss pixels each labeling choice keeps or drops.

Counts native 30 m Hansen pixels by tree-cover class and loss year, per quarter of
the region. One query covers every combination, so choices can be compared offline.

Run from the repo root:
    python -m scripts.report_labels
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import ee

from src.config import DEFAULT_CONFIG, load_config
from src.data import gee

# Lower bound of each tree-cover class, as percent cover in the year 2000.
TREECOVER_BINS = (0, 30, 50, 75)
CODE_BASE = 100  # lossyear runs from 0 to 25 so it fits below this
HANSEN_DEGREES = 0.00025
LAND = 1  # Hansen datamask value for mapped land
MASKED = (-1, -1)  # counts key for pixels outside the land mask


def hansen_codes(asset: str) -> ee.Image:
    """Combine tree-cover class and loss year into one integer per land pixel."""
    hansen = ee.Image(asset)
    cover_class = ee.Image(0)
    for index, lower in enumerate(TREECOVER_BINS[1:], start=1):
        cover_class = cover_class.where(hansen.select("treecover2000").gte(lower), index)
    code = cover_class.multiply(CODE_BASE).add(hansen.select("lossyear").unmask(0))
    return code.updateMask(hansen.select("datamask").eq(LAND)).toInt().rename("code")


def query_counts(asset: str, regions: dict) -> dict[str, dict[tuple[int, int], int]]:
    """Return {region: {(cover class, loss year): pixel count}}."""
    features = ee.FeatureCollection(
        [ee.Feature(gee.bbox_geometry(box), {"name": name}) for name, box in regions.items()]
    )
    result = hansen_codes(asset).reduceRegions(
        collection=features,
        reducer=ee.Reducer.frequencyHistogram().unweighted(),
        crs="EPSG:4326",
        crsTransform=[HANSEN_DEGREES, 0, -180, 0, -HANSEN_DEGREES, 80],
    ).getInfo()

    counts: dict[str, dict[tuple[int, int], int]] = {}
    for feature in result["features"]:
        region_counts: dict[tuple[int, int], int] = defaultdict(int)
        for code, pixels in feature["properties"]["histogram"].items():
            if code == "null":  # non-land pixels masked out in hansen_codes
                region_counts[MASKED] += round(pixels)
                continue
            cover_class, loss_year = divmod(int(float(code)), CODE_BASE)
            region_counts[(cover_class, loss_year)] += round(pixels)
        counts[feature["properties"]["name"]] = dict(region_counts)
    return counts


def pixels_where(counts: dict[tuple[int, int], int], min_cover: int, years) -> int:
    """Sum pixels with tree cover of at least min_cover whose loss year is in years."""
    return sum(
        pixels for (cover_class, loss_year), pixels in counts.items()
        if (cover_class, loss_year) != MASKED
        and TREECOVER_BINS[cover_class] >= min_cover and loss_year in years
    )


def summarize(counts: dict[tuple[int, int], int], before: int, after: int, hectares_per_pixel: float) -> dict:
    before_code, after_code = before - 2000, after - 2000
    prior_loss = set(range(1, before_code))
    middle = set(range(before_code + 1, after_code))
    edges = {before_code, after_code}
    later = set(range(after_code + 1, 100))

    masked = counts.get(MASKED, 0)
    summary: dict = {"land_pixels": sum(counts.values()) - masked, "non_land_pixels": masked}
    for min_cover in TREECOVER_BINS[1:]:
        forest = pixels_where(counts, min_cover, {0} | middle | edges | later)
        row = {
            "forest_at_before_year": forest,
            "removed_as_lost_before_period": pixels_where(counts, min_cover, prior_loss),
            "loss_middle_years_kept": pixels_where(counts, min_cover, middle),
            f"loss_{before}_ignored": pixels_where(counts, min_cover, {before_code}),
            f"loss_{after}_ignored": pixels_where(counts, min_cover, {after_code}),
            "loss_after_period_treated_as_no_loss": pixels_where(counts, min_cover, later),
        }
        row["kept_share_of_all_period_loss"] = row["loss_middle_years_kept"] / max(
            1, pixels_where(counts, min_cover, middle | edges)
        )
        row["kept_share_of_forest"] = row["loss_middle_years_kept"] / max(1, forest)
        row["kept_hectares_approx"] = row["loss_middle_years_kept"] * hectares_per_pixel
        summary[f"treecover_ge_{min_cover}"] = row
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    gee.initialize(config)
    box = config.region.bbox

    counts = query_counts(config.earth_engine.hansen_asset, gee.quarters(box))
    total: dict[tuple[int, int], int] = defaultdict(int)
    for region_counts in counts.values():
        for key, pixels in region_counts.items():
            total[key] += pixels
    counts["FULL"] = dict(total)

    # Pixel area at the box's mid latitude; approximate because it varies slightly with latitude.
    metres = HANSEN_DEGREES * 111_320
    hectares_per_pixel = metres * metres * math.cos(math.radians((box.south + box.north) / 2)) / 10_000

    before, after = config.years.before, config.years.after
    report = {name: summarize(region, before, after, hectares_per_pixel) for name, region in counts.items()}
    report["loss_year_counts_full_box"] = {
        f"treecover_ge_{cover}": {
            str(2000 + year): pixels_where(counts["FULL"], cover, {year}) for year in range(1, 26)
        }
        for cover in TREECOVER_BINS[1:]
    }

    output = config.paths.outputs_dir / "phase1" / "label_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
