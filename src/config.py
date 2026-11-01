"""Load and validate the project's YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# Encoders downsample five times, so patch sides must divide by 2**5.
PATCH_SIZE_MULTIPLE = 32


class ConfigError(ValueError):
    """Raised when a config file is missing keys or has invalid values."""


@dataclass(frozen=True)
class EarthEngine:
    project: str
    hansen_asset: str


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float


@dataclass(frozen=True)
class Region:
    name: str
    bbox: BBox


@dataclass(frozen=True)
class Years:
    before: int
    after: int


@dataclass(frozen=True)
class Season:
    start_month: int
    end_month: int


@dataclass(frozen=True)
class Composite:
    s2_max_cloud_percent: float


@dataclass(frozen=True)
class Export:
    crs: str
    scale_m: int
    drive_folder: str
    file_dimensions: int


@dataclass(frozen=True)
class Labels:
    forest_min_treecover: int


@dataclass(frozen=True)
class Patch:
    size: int
    stride: int
    max_nodata_fraction: float


@dataclass(frozen=True)
class Split:
    train: float
    val: float
    test: float
    buffer_m: int


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    patches_dir: Path
    outputs_dir: Path


@dataclass(frozen=True)
class Config:
    earth_engine: EarthEngine
    region: Region
    subset: Region
    years: Years
    season: Season
    composite: Composite
    export: Export
    labels: Labels
    patch: Patch
    split: Split
    paths: Paths


def _build(cls: type, raw: Any, section: str) -> Any:
    """Build a dataclass from a YAML mapping, naming the section on failure."""
    if not isinstance(raw, dict):
        raise ConfigError(f"'{section}' must be a mapping, got {type(raw).__name__}")
    try:
        return cls(**raw)
    except TypeError as exc:
        # Covers missing keys and misspelled keys.
        raise ConfigError(f"Invalid keys in '{section}': {exc}") from exc


def _validate(config: Config) -> None:
    for box in (config.region.bbox, config.subset.bbox):
        if not (-180 <= box.west < box.east <= 180):
            raise ConfigError(f"bbox needs -180 <= west < east <= 180, got {box}")
        if not (-90 <= box.south < box.north <= 90):
            raise ConfigError(f"bbox needs -90 <= south < north <= 90, got {box}")

    outer, inner = config.region.bbox, config.subset.bbox
    if not (outer.west <= inner.west and inner.east <= outer.east
            and outer.south <= inner.south and inner.north <= outer.north):
        raise ConfigError(f"subset bbox must lie inside the region bbox, got {inner}")

    if config.years.before >= config.years.after:
        raise ConfigError(f"years.before must be earlier than years.after, got {config.years}")

    season = config.season
    if not 1 <= season.start_month <= season.end_month <= 12:
        raise ConfigError(f"season needs 1 <= start_month <= end_month <= 12, got {season}")

    size, stride = config.patch.size, config.patch.stride
    if size <= 0 or size % PATCH_SIZE_MULTIPLE != 0:
        raise ConfigError(f"patch.size must be a positive multiple of {PATCH_SIZE_MULTIPLE}, got {size}")
    if not 0 < stride <= size:
        raise ConfigError(f"patch.stride must be between 1 and patch.size, got {stride}")
    if not 0 <= config.patch.max_nodata_fraction <= 1:
        raise ConfigError(f"patch.max_nodata_fraction must be between 0 and 1, got {config.patch.max_nodata_fraction}")

    split = config.split
    if min(split.train, split.val, split.test) <= 0 or abs(split.train + split.val + split.test - 1) > 1e-6:
        raise ConfigError(f"split shares must be positive and sum to 1, got {split}")
    if split.buffer_m < 0:
        raise ConfigError(f"split.buffer_m must not be negative, got {split.buffer_m}")

    if not 0 <= config.labels.forest_min_treecover <= 100:
        raise ConfigError(f"labels.forest_min_treecover must be 0-100, got {config.labels.forest_min_treecover}")


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """Read a YAML config, validate it, and resolve paths against the repo root."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file is empty or not a mapping: {path}")

    region_raw = dict(_build(dict, raw.get("region"), "region"))
    region_raw["bbox"] = _build(BBox, region_raw.get("bbox"), "region.bbox")
    subset_raw = dict(_build(dict, raw.get("subset"), "subset"))
    subset_raw["bbox"] = _build(BBox, subset_raw.get("bbox"), "subset.bbox")

    paths = _build(Paths, raw.get("paths"), "paths")
    resolved_paths = Paths(
        raw_dir=REPO_ROOT / paths.raw_dir,
        patches_dir=REPO_ROOT / paths.patches_dir,
        outputs_dir=REPO_ROOT / paths.outputs_dir,
    )

    config = Config(
        earth_engine=_build(EarthEngine, raw.get("earth_engine"), "earth_engine"),
        region=_build(Region, region_raw, "region"),
        subset=_build(Region, subset_raw, "subset"),
        years=_build(Years, raw.get("years"), "years"),
        season=_build(Season, raw.get("season"), "season"),
        composite=_build(Composite, raw.get("composite"), "composite"),
        export=_build(Export, raw.get("export"), "export"),
        labels=_build(Labels, raw.get("labels"), "labels"),
        patch=_build(Patch, raw.get("patch"), "patch"),
        split=_build(Split, raw.get("split"), "split"),
        paths=resolved_paths,
    )
    _validate(config)
    return config
