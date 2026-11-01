"""Cut exported GeoTIFFs into patches and assign geographic splits."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.windows import Window

from src.config import Config
from src.data.labels import IGNORE, LOSS, derive_label

TRAIN, VAL, TEST, BUFFER = "train", "val", "test", "buffer"


def image_band_indices(bands: list[str]) -> list[int]:
    """Positions of the model input bands, excluding quality and Hansen bands."""
    excluded = ("s2_clear_count", "lossyear", "treecover2000")
    return [i for i, name in enumerate(bands) if not name.startswith(excluded)]


def source_prefixes(config: Config, area: str) -> list[str]:
    """File name prefixes of the exports that make up an area."""
    # The subset is the NE quarter, so the full region is that plus the other three.
    exports = ["subset", "nw", "sw", "se"] if area == "full" else [area]
    years = f"{config.years.before}_{config.years.after}"
    return [f"stack_{config.region.name}_{name}_{years}" for name in exports]


def iter_grid_cells(sources: list, size_m: int, stride_m: int) -> Iterator[tuple[int, int]]:
    """Yield (left, top) of patch cells on one grid anchored at the CRS origin.

    Anchoring to the origin, not to a file's corner, means every export shares the
    grid, so patches from neighbouring files cannot partly overlap.
    """
    west = min(source.bounds.left for source in sources)
    east = max(source.bounds.right for source in sources)
    south = min(source.bounds.bottom for source in sources)
    north = max(source.bounds.top for source in sources)

    first_left = math.floor(west / stride_m) * stride_m
    first_top = math.ceil(north / stride_m) * stride_m
    for top in range(first_top, math.floor(south), -stride_m):
        for left in range(first_left, math.ceil(east), stride_m):
            yield left, top


def read_cell(sources: list, left: int, top: int, size: int, scale_m: int, bands: int, nodata: int) -> np.ndarray:
    """Read one patch cell across files, keeping the first valid value for each pixel."""
    size_m = size * scale_m
    cell = np.full((bands, size, size), nodata, dtype=np.int16)
    for source in sources:
        box = source.bounds
        x_start, x_end = max(left, box.left), min(left + size_m, box.right)
        y_start, y_end = max(top - size_m, box.bottom), min(top, box.top)
        if x_start >= x_end or y_start >= y_end:
            continue

        width, height = round((x_end - x_start) / scale_m), round((y_end - y_start) / scale_m)
        window = Window(round((x_start - box.left) / scale_m), round((box.top - y_end) / scale_m), width, height)
        data = source.read(window=window)

        col, row = round((x_start - left) / scale_m), round((top - y_end) / scale_m)
        target = cell[:, row:row + height, col:col + width]
        empty = target == nodata
        target[empty] = data[empty]
    return cell


def cut_mosaic(paths: list[Path], manifest: dict, config: Config, output_dir: Path) -> tuple[list[dict], int]:
    """Write patches from a set of GeoTIFFs, holding one patch in RAM at a time.

    Returns the patch records and the number of grid cells skipped because imagery was
    missing, which includes partial cells along the mosaic's outer edge.
    """
    bands = manifest["bands"]
    image_indices = image_band_indices(bands)
    clear_indices = [i for i, name in enumerate(bands) if name.startswith("s2_clear_count")]
    lossyear_index, treecover_index = bands.index("lossyear"), bands.index("treecover2000")
    nodata = manifest["nodata"]
    size, scale_m = config.patch.size, config.export.scale_m
    size_m, stride_m = size * scale_m, config.patch.stride * scale_m

    sources = [rasterio.open(path) for path in paths]
    try:
        for source in sources:
            if source.count != len(bands):
                raise ValueError(f"{source.name} has {source.count} bands but the manifest lists {len(bands)}")
            if source.bounds.left % scale_m or source.bounds.top % scale_m or source.res != (scale_m, scale_m):
                raise ValueError(f"{source.name} is not aligned to the {scale_m} m pixel grid")

        records, skipped = [], 0
        for left, top in iter_grid_cells(sources, size_m, stride_m):
            data = read_cell(sources, left, top, size, scale_m, len(bands), nodata)
            image = data[image_indices]
            missing = (image == nodata).any(axis=0).mean()
            if missing > config.patch.max_nodata_fraction:
                skipped += 1
                continue

            label = derive_label(
                data[lossyear_index], data[treecover_index], config.years.before, config.years.after,
                config.labels.forest_min_treecover, nodata,
            )
            patch_id = f"n{top}_e{left}"
            np.savez(output_dir / f"{patch_id}.npz", image=image, label=label, clear_count=data[clear_indices])
            records.append({
                "patch_id": patch_id,
                "left": left, "bottom": top - size_m, "right": left + size_m, "top": top,
                "loss_pixels": int((label == LOSS).sum()),
                "ignored_pixels": int((label == IGNORE).sum()),
                "total_pixels": int(label.size),
                "missing_fraction": float(missing),
            })
    finally:
        for source in sources:
            source.close()
    return records, skipped

def assign_splits(records: list[dict], config: Config) -> None:
    """Assign each record a north-to-south stripe, in place.

    Rows of patches are grouped so each split holds close to its target share of loss
    pixels, not of area, because loss clusters unevenly. Rows at each boundary become
    BUFFER, so pixels from neighbouring splits cannot share a footprint or sit side by
    side. Both cut points are searched together, buffers included, because one
    loss-heavy row can otherwise leave a split empty.
    """
    loss_by_row: dict[float, int] = defaultdict(int)
    for record in records:
        loss_by_row[record["top"]] += record["loss_pixels"]
    rows = sorted(loss_by_row, reverse=True)
    prefix = [0]
    for row in rows:
        prefix.append(prefix[-1] + loss_by_row[row])

    # Dropping k rows leaves a gap of (k + 1) * stride - size pixels.
    size, stride = config.patch.size, config.patch.stride
    buffer_px = config.split.buffer_m / config.export.scale_m
    buffer_rows = max(0, math.ceil((buffer_px + size) / stride) - 1)

    targets = (config.split.train, config.split.val, config.split.test)
    best, best_error = None, math.inf
    for train_end in range(1, len(rows)):
        val_start = train_end + buffer_rows
        for val_end in range(val_start + 1, len(rows)):
            test_start = val_end + buffer_rows
            if test_start >= len(rows):
                break
            losses = (prefix[train_end], prefix[val_end] - prefix[val_start], prefix[-1] - prefix[test_start])
            if min(losses) == 0:
                continue
            kept = sum(losses)
            error = sum(abs(loss / kept - target) for loss, target in zip(losses, targets))
            if error < best_error:
                best, best_error = (train_end, val_start, val_end, test_start), error

    if best is None:
        raise ValueError(f"Cannot form three splits containing loss from {len(rows)} rows of patches")

    train_end, val_start, val_end, test_start = best
    row_split = {}
    for index, row in enumerate(rows):
        if index < train_end:
            row_split[row] = TRAIN
        elif val_start <= index < val_end:
            row_split[row] = VAL
        elif index >= test_start:
            row_split[row] = TEST
        else:
            row_split[row] = BUFFER

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
