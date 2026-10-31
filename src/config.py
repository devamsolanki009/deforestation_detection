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
class Patch:
    size: int
    stride: int


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    patches_dir: Path
    outputs_dir: Path


@dataclass(frozen=True)
class Config:
    region: Region
    years: Years
    patch: Patch
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
    box = config.region.bbox
    if not (-180 <= box.west < box.east <= 180):
        raise ConfigError(f"bbox needs -180 <= west < east <= 180, got {box}")
    if not (-90 <= box.south < box.north <= 90):
        raise ConfigError(f"bbox needs -90 <= south < north <= 90, got {box}")

    if config.years.before >= config.years.after:
        raise ConfigError(f"years.before must be earlier than years.after, got {config.years}")

    size, stride = config.patch.size, config.patch.stride
    if size <= 0 or size % PATCH_SIZE_MULTIPLE != 0:
        raise ConfigError(f"patch.size must be a positive multiple of {PATCH_SIZE_MULTIPLE}, got {size}")
    if not 0 < stride <= size:
        raise ConfigError(f"patch.stride must be between 1 and patch.size, got {stride}")


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

    paths = _build(Paths, raw.get("paths"), "paths")
    resolved_paths = Paths(
        raw_dir=REPO_ROOT / paths.raw_dir,
        patches_dir=REPO_ROOT / paths.patches_dir,
        outputs_dir=REPO_ROOT / paths.outputs_dir,
    )

    config = Config(
        region=_build(Region, region_raw, "region"),
        years=_build(Years, raw.get("years"), "years"),
        patch=_build(Patch, raw.get("patch"), "patch"),
        paths=resolved_paths,
    )
    _validate(config)
    return config
