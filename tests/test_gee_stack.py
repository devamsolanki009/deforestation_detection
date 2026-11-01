"""Needs Earth Engine access. Only reads band names, so it uses almost no quota."""

import pytest

ee = pytest.importorskip("ee")

from src.config import load_config  # noqa: E402
from src.data import gee  # noqa: E402


def test_manifest_band_order_matches_earth_engine_stack():
    config = load_config()
    try:
        gee.initialize(config)
    except ee.EEException as exc:
        pytest.skip(f"Earth Engine not available: {exc}")

    image = gee.build_stack(config, gee.bbox_geometry(config.subset.bbox))
    assert image.bandNames().getInfo() == gee.stack_band_names(config)
