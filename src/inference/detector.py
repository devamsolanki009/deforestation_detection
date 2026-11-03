"""Load the chosen model once and run area requests against it."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from src.config import DEFAULT_CONFIG, REPO_ROOT, BBox, Config, ConfigError, _build, load_config
from src.data import gee
from src.inference.area import AreaResult, analyze
from src.inference.fetch import Grid, fetch_stack, grid_for_bbox, with_years
from src.inference.predict import load_checkpoint

DEFAULT_APP_CONFIG = REPO_ROOT / "configs" / "app.yaml"


@dataclass(frozen=True)
class AppConfig:
    run: str
    max_area_km2: float
    fetch_tile_px: int
    fetch_workers: int
    cache_dir: Path
    inference_overlap_px: int
    min_detection_ha: float
    first_year: int
    last_year: int
    trained_year_gap: int
    outputs_dir: Path
    default_bbox: BBox


def load_app_config(path: str | Path = DEFAULT_APP_CONFIG) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw = dict(_build(dict, raw, "app"))
    raw["default_bbox"] = _build(BBox, raw.get("default_bbox"), "default_bbox")
    raw["cache_dir"] = REPO_ROOT / raw.get("cache_dir", "")
    raw["outputs_dir"] = REPO_ROOT / raw.get("outputs_dir", "")
    return _build(AppConfig, raw, "app")


@dataclass
class Request:
    """One area request after validation."""

    bbox: BBox
    before: int
    after: int
    grid: Grid
    warnings: list[str]


KM_PER_DEGREE_LAT = 110.574
KM_PER_DEGREE_LON_AT_EQUATOR = 111.320


def bbox_around(lon: float, lat: float, size_km: float) -> BBox:
    """A roughly square box of the given side length centred on a point."""
    half_lat = size_km / 2 / KM_PER_DEGREE_LAT
    half_lon = size_km / 2 / (KM_PER_DEGREE_LON_AT_EQUATOR * math.cos(math.radians(lat)))
    return BBox(round(lon - half_lon, 5), round(lat - half_lat, 5), round(lon + half_lon, 5), round(lat + half_lat, 5))


def validate_request(app: AppConfig, config: Config, bbox: BBox, before: int, after: int) -> Request:
    """Check a request before any quota is used; raise ConfigError with a readable message."""
    if not (-180 <= bbox.west < bbox.east <= 180 and -90 <= bbox.south < bbox.north <= 90):
        raise ConfigError("The box needs west < east and south < north, in degrees.")
    if not app.first_year <= before < after <= app.last_year:
        raise ConfigError(f"Years must satisfy {app.first_year} <= before < after <= {app.last_year}.")

    grid = grid_for_bbox(bbox, config.export.crs, config.export.scale_m)
    if grid.area_km2 > app.max_area_km2:
        raise ConfigError(f"The box covers {grid.area_km2:.0f} km2; the limit is {app.max_area_km2:.0f} km2.")

    warnings = []
    region = config.region.bbox
    inside = (region.west <= bbox.west and bbox.east <= region.east
              and region.south <= bbox.south and bbox.north <= region.north)
    if not inside:
        warnings.append("This area is outside the training region (Sao Felix do Xingu); accuracy there is untested.")
    if after - before != app.trained_year_gap:
        warnings.append(f"The model was trained on {app.trained_year_gap}-year gaps; "
                        f"a {after - before}-year gap is untested.")
    if (before, after) == (config.years.before, config.years.after) and inside:
        warnings.append("Most of this region was used for training or tuning; only the southern test "
                        "stripe gives an honest picture of accuracy.")
    return Request(bbox, before, after, grid, warnings)


class Detector:
    def __init__(self, app: AppConfig, config_path: Path = DEFAULT_CONFIG, device: str | None = None):
        self.app = app
        self.config = load_config(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        run_dir = self.config.paths.outputs_dir / "runs" / app.run
        self.model, checkpoint = load_checkpoint(run_dir / "best.pt", self.device)
        self.band_stats = checkpoint["band_stats"]

        # Threshold and forest mask come from the run's val-tuned evaluation.
        metrics = json.loads((run_dir / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
        pipeline = metrics["pipelines"]["forest_mask"]
        self.threshold = pipeline["threshold"]
        self.model_summary = {
            "run": app.run, "threshold": self.threshold, "device": str(self.device),
            "val_f1": round(pipeline["val_best_f1"], 3),
            "test_f1": round(pipeline["splits"]["test"]["metrics"]["f1"], 3),
        }

    def validate(self, bbox: BBox, before: int, after: int) -> Request:
        return validate_request(self.app, self.config, bbox, before, after)

    def run(
        self, request: Request, progress: Callable[[int, int], None] | None = None,
    ) -> tuple[AreaResult, np.ndarray, list[str], dict]:
        """Fetch imagery for a validated request and detect loss. Returns the result,
        the raw stack and its band names (for display), and fetch statistics."""
        config = with_years(self.config, request.before, request.after)
        stack, fetch_info = fetch_stack(config, request.grid, self.app.cache_dir, self.app.fetch_tile_px,
                                        self.app.fetch_workers, progress)
        bands = gee.stack_band_names(config)
        result = analyze(self.model, self.band_stats, stack, bands, request.grid, config, self.threshold,
                         self.app.min_detection_ha, self.device, self.app.inference_overlap_px)
        result.warnings = request.warnings + result.warnings
        return result, stack, bands, fetch_info
