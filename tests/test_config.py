import copy
from pathlib import Path

import pytest
import yaml

from src.config import DEFAULT_CONFIG, REPO_ROOT, ConfigError, load_config

VALID = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def edited(section: str, key: str, value) -> dict:
    raw = copy.deepcopy(VALID)
    raw[section][key] = value
    return raw


def test_default_config_loads():
    config = load_config()
    assert config.years.before < config.years.after
    assert config.patch.size == 256
    assert config.region.bbox.west < config.region.bbox.east


def test_paths_resolve_against_repo_root():
    assert load_config().paths.raw_dir == REPO_ROOT / "data" / "raw"


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_empty_file_is_reported(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


def test_missing_section_is_reported(tmp_path):
    raw = copy.deepcopy(VALID)
    del raw["years"]
    with pytest.raises(ConfigError, match="years"):
        load_config(write_config(tmp_path, raw))


def test_misspelled_key_is_reported(tmp_path):
    raw = copy.deepcopy(VALID)
    raw["patch"]["sizee"] = raw["patch"].pop("size")
    with pytest.raises(ConfigError, match="patch"):
        load_config(write_config(tmp_path, raw))


@pytest.mark.parametrize(
    "raw",
    [
        edited("years", "before", 2022),  # same year as after
        edited("years", "before", 2024),  # later than after
        edited("patch", "size", 250),     # not divisible by 32
        edited("patch", "size", 0),
        edited("patch", "stride", 0),
        edited("patch", "stride", 512),   # larger than the patch, leaving gaps
    ],
)
def test_invalid_values_are_rejected(tmp_path, raw):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, raw))


@pytest.mark.parametrize(
    "bbox",
    [
        {"west": -51.5, "south": -7.0, "east": -52.5, "north": -6.0},  # west past east
        {"west": -52.5, "south": -6.0, "east": -51.5, "north": -7.0},  # south past north
        {"west": -200.0, "south": -7.0, "east": -51.5, "north": -6.0},  # out of range
    ],
)
def test_invalid_bbox_is_rejected(tmp_path, bbox):
    raw = copy.deepcopy(VALID)
    raw["region"]["bbox"] = bbox
    with pytest.raises(ConfigError, match="bbox"):
        load_config(write_config(tmp_path, raw))
