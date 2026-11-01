"""Cut exported GeoTIFFs into patches and assign geographic splits."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds

from src.config import Config
from src.data.labels import IGNORE, LOSS, derive_label

TRAIN, VAL, TEST, BUFFER = "train", "val", "test", "buffer"


def image_band_indices(bands: list[str]) -> list[int]:
    """Positions of the model input bands, excluding quality and Hansen bands."""
    excluded = ("s2_clear_count", "lossyear", "treecover2000")
    return [i for i, name in enumerate(bands) if not name.startswith(excluded)]


def iter_windows(width: int, height: int, size: int, stride: int) -> Iterator[Window]:
    """Full-size windows only, so a partial strip at the edge is skipped."""
    for row in range(0, height - size + 1, stride):
        for col in range(0, width - size + 1, stride):
            yield Window(col, row, size, size)


def cut_file(path: Path, manifest: dict, config: Config, output_dir: Path) -> tuple[list[dict], int]:
    """Write patches from one GeoTIFF, reading one window at a time to stay light on RAM.

    Returns the patch records and the number of windows skipped for missing imagery.
    """
    bands = manifest["bands"]
    image_indices = image_band_indices(bands)
    clear_indices = [i for i, name in enumerate(bands) if name.startswith("s2_clear_count")]
    lossyear_index, treecover_index = bands.index("lossyear"), bands.index("treecover2000")
    nodata = manifest["nodata"]

    records, skipped = [], 0
    with rasterio.open(path) as source:
        if source.count != len(bands):
            raise ValueError(f"{path} has {source.count} bands but the manifest lists {len(bands)}")

        for window in iter_windows(source.width, source.height, config.patch.size, config.patch.stride):
            data = source.read(window=window)
            image = data[image_indices]
            missing = (image == nodata).any(axis=0).mean()
            if missing > config.patch.max_nodata_fraction:
                skipped += 1
                continue

            label = derive_label(
                data[lossyear_index], data[treecover_index], config.years.before, config.years.after,
                config.labels.forest_min_treecover, nodata,
            )
            left, bottom, right, top = window_bounds(window, source.transform)
            patch_id = f"{path.stem}_r{window.row_off:05d}_c{window.col_off:05d}"
            np.savez(output_dir / f"{patch_id}.npz", image=image, label=label, clear_count=data[clear_indices])
            records.append({
                "patch_id": patch_id,
                "left": left, "bottom": bottom, "right": right, "top": top,
                "loss_pixels": int((label == LOSS).sum()),
                "ignored_pixels": int((label == IGNORE).sum()),
                "total_pixels": int(label.size),
                "missing_fraction": float(missing),
            })
    return records, skipped


def assign_splits(records: list[dict], config: Config) -> None:
    """Assign each record a north-to-south stripe, in place.

    Rows of patches are grouped so a split holds its target share of loss pixels, not
    of area, because loss clusters unevenly. Rows at each boundary become BUFFER, so
    pixels from neighbouring splits cannot share a patch footprint or sit side by side.
    """
    loss_by_row: dict[float, int] = defaultdict(int)
    for record in records:
        loss_by_row[record["top"]] += record["loss_pixels"]
    rows = sorted(loss_by_row, reverse=True)
    total_loss = sum(loss_by_row.values())

    split = config.split
    train_end, val_end = split.train, split.train + split.val
    row_split: dict[float, str] = {}
    cumulative = 0
    for row in rows:
        # A row's position is the midpoint of its loss, so a heavy row lands where most of it falls.
        midpoint = (cumulative + loss_by_row[row] / 2) / max(1, total_loss)
        row_split[row] = TRAIN if midpoint < train_end else VAL if midpoint < val_end else TEST
        cumulative += loss_by_row[row]

    # Dropping k rows leaves a gap of (k + 1) * stride - size pixels.
    size, stride = config.patch.size, config.patch.stride
    buffer_px = split.buffer_m / config.export.scale_m
    buffer_rows = max(0, math.ceil((buffer_px + size) / stride) - 1)

    for index in range(1, len(rows)):
        if row_split[rows[index]] != BUFFER and row_split[rows[index - 1]] not in (BUFFER, row_split[rows[index]]):
            for offset in range(buffer_rows):
                if index + offset < len(rows):
                    row_split[rows[index + offset]] = BUFFER

    for record in records:
        record["split"] = row_split[record["top"]]


def split_summary(records: list[dict]) -> dict:
    """Patch counts and pixel shares per split, computed over non-ignored pixels."""
    summary = {}
    for name in (TRAIN, VAL, TEST, BUFFER):
        group = [r for r in records if r["split"] == name]
        total = sum(r["total_pixels"] for r in group)
        ignored = sum(r["ignored_pixels"] for r in group)
        loss = sum(r["loss_pixels"] for r in group)
        summary[name] = {
            "patches": len(group),
            "patches_with_loss": sum(1 for r in group if r["loss_pixels"] > 0),
            "loss_pixels": loss,
            "ignored_pixels": ignored,
            "total_pixels": total,
            "loss_share_of_labeled_pixels": loss / max(1, total - ignored),
        }
    return summary
