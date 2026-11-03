"""Fetch the composite stack for a small area directly from Earth Engine.

Batch exports to Drive suit whole regions but take an hour or more. The app needs a
small area in minutes, so it requests pixels interactively, split into tiles small
enough for one request each. Tiles are cached on disk, keyed by area and years, so a
repeated request reads the cache and uses no Earth Engine quota.
"""

from __future__ import annotations

import dataclasses
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ee
import numpy as np
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

from src.config import BBox, Config, Years
from src.data import gee

RETRIES = 3
RETRY_SECONDS = 10


@dataclass(frozen=True)
class Grid:
    """A north-up pixel grid in a projected CRS, snapped to multiples of the pixel size.

    Snapping puts every pixel on the same grid as the training exports.
    """

    left: int
    top: int
    width: int
    height: int
    scale_m: int
    crs: str

    @property
    def transform(self) -> Affine:
        return Affine(self.scale_m, 0, self.left, 0, -self.scale_m, self.top)

    @property
    def area_km2(self) -> float:
        return self.width * self.height * self.scale_m ** 2 / 1e6

    def key(self) -> str:
        return f"e{self.left}_n{self.top}_{self.width}x{self.height}"


def grid_for_bbox(bbox: BBox, crs: str, scale_m: int) -> Grid:
    """The smallest snapped grid covering a lon/lat box."""
    west, south, east, north = transform_bounds("EPSG:4326", crs, bbox.west, bbox.south, bbox.east, bbox.north)
    left, right = math.floor(west / scale_m) * scale_m, math.ceil(east / scale_m) * scale_m
    bottom, top = math.floor(south / scale_m) * scale_m, math.ceil(north / scale_m) * scale_m
    return Grid(left, top, (right - left) // scale_m, (top - bottom) // scale_m, scale_m, crs)


def with_years(config: Config, before: int, after: int) -> Config:
    """A copy of the config for another pair of years."""
    return dataclasses.replace(config, years=Years(before, after))


def tile_boxes(grid: Grid, tile_px: int) -> list[tuple[int, int, int, int]]:
    """(row, col, height, width) of request tiles covering the grid."""
    return [(row, col, min(tile_px, grid.height - row), min(tile_px, grid.width - col))
            for row in range(0, grid.height, tile_px) for col in range(0, grid.width, tile_px)]


def _request_tile(image: ee.Image, grid: Grid, bands: list[str], row: int, col: int, height: int, width: int):
    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": width, "height": height},
            "affineTransform": {
                "scaleX": grid.scale_m, "shearX": 0, "translateX": grid.left + col * grid.scale_m,
                "shearY": 0, "scaleY": -grid.scale_m, "translateY": grid.top - row * grid.scale_m,
            },
            "crsCode": grid.crs,
        },
    }
    for attempt in range(RETRIES):
        try:
            data = ee.data.computePixels(request)
            break
        except ee.EEException as exc:
            # Quota and busy errors are transient; anything else will not fix itself.
            if attempt == RETRIES - 1 or not any(s in str(exc) for s in ("429", "Too many", "timed out", "503")):
                raise
            time.sleep(RETRY_SECONDS * (attempt + 1))
    # The result is a structured array with one field per band; field names guard the order.
    return np.stack([data[name] for name in bands]).astype(np.int16)


def fetch_stack(
    config: Config, grid: Grid, cache_dir: Path, tile_px: int = 512, workers: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, dict]:
    """Return the (bands, H, W) Int16 stack for the grid, and request statistics.

    Band order is gee.stack_band_names(config). Each tile of 28 Int16 bands at 512 x 512
    is about 15 MB, below Earth Engine's per-request limit.
    """
    bands = gee.stack_band_names(config)
    folder = cache_dir / f"{config.years.before}_{config.years.after}_{grid.key()}"
    folder.mkdir(parents=True, exist_ok=True)
    boxes = tile_boxes(grid, tile_px)
    stack = np.full((len(bands), grid.height, grid.width), gee.NODATA, dtype=np.int16)

    missing = [box for box in boxes if not (folder / f"r{box[0]}_c{box[1]}.npy").is_file()]
    started = time.perf_counter()
    if missing:
        gee.initialize(config)
        geometry = ee.Geometry.Rectangle(
            [grid.left, grid.top - grid.height * grid.scale_m, grid.left + grid.width * grid.scale_m, grid.top],
            proj=grid.crs, geodesic=False,
        )
        image = gee.build_stack(config, geometry)

        def fetch(box):
            tile = _request_tile(image, grid, bands, *box)
            np.save(folder / f"r{box[0]}_c{box[1]}.npy", tile)

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(fetch, missing):
                done += 1
                if progress:
                    progress(done, len(missing))

    for row, col, height, width in boxes:
        stack[:, row:row + height, col:col + width] = np.load(folder / f"r{row}_c{col}.npy")

    return stack, {
        "tiles": len(boxes), "tiles_requested": len(missing), "tiles_from_cache": len(boxes) - len(missing),
        "request_seconds": round(time.perf_counter() - started, 1) if missing else 0.0,
        "cache_folder": str(folder),
    }
