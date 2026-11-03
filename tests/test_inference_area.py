import numpy as np
import pytest
import torch

from src.config import BBox, ConfigError, load_config
from src.data.gee import NODATA, stack_band_names
from src.inference.area import analyze, hansen_last_year
from src.inference.detector import bbox_around, load_app_config, validate_request
from src.inference.fetch import Grid, grid_for_bbox, tile_boxes, with_years
from src.inference.tiled import blend_window, predict_tiled, tile_starts


class FirstBandLogit(torch.nn.Module):
    """Returns input band 0 as the logit, so the right answer is known per pixel."""

    def forward(self, x):
        return x[:, :1]


class ConstantLogit(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x[:, :1], self.value)


def test_tile_starts_cover_the_length_and_end_exactly():
    assert tile_starts(600, 256, 192) == [0, 192, 344]
    assert tile_starts(256, 256, 192) == [0]
    assert tile_starts(100, 256, 192) == [0]


def test_blend_window_is_positive_and_peaks_at_one():
    window = blend_window(256, 64)
    assert window.min() > 0
    assert window[128, 128] == 1


@pytest.mark.parametrize("shape", [(300, 517), (100, 90), (256, 256)])
def test_blending_reproduces_a_position_independent_model_exactly(shape):
    # Any per-pixel model must come back unchanged, whatever the tiling and overlap.
    image = np.random.default_rng(0).normal(size=(3, *shape)).astype(np.float32)
    probability = predict_tiled(FirstBandLogit(), image, torch.device("cpu"), tile=256, overlap=64)
    assert probability.shape == shape
    np.testing.assert_allclose(probability, 1 / (1 + np.exp(-image[0])), atol=1e-5)


def test_grid_is_snapped_to_the_pixel_grid_and_covers_the_box():
    grid = grid_for_bbox(BBox(-52.2814, -6.8133, -52.2350, -6.7668), "EPSG:32722", 10)
    assert grid.left % 10 == 0 and grid.top % 10 == 0
    assert 510 <= grid.width <= 520 and 510 <= grid.height <= 520


def test_request_tiles_cover_the_grid_once():
    grid = Grid(0, 10_000, 1100, 700, 10, "EPSG:32722")
    boxes = tile_boxes(grid, 512)
    assert sum(h * w for _, _, h, w in boxes) == 1100 * 700
    assert len(boxes) == 6


def synthetic_stack(config):
    """Left half forest (80% cover), right half not (10%), with dated Hansen loss blocks."""
    bands = stack_band_names(config)
    stack = np.full((len(bands), 64, 64), 1000, dtype=np.int16)
    lossyear = np.zeros((64, 64), dtype=np.int16)
    lossyear[0:20, 0:20] = 21    # between 2020 and 2022: the label year
    lossyear[30:40, 0:10] = 22   # after year: ambiguous
    lossyear[50:60, 0:10] = 15   # lost before 2020: not forest at start
    treecover = np.full((64, 64), 10, dtype=np.int16)
    treecover[:, :32] = 80
    stack[bands.index("lossyear")] = lossyear
    stack[bands.index("treecover2000")] = treecover
    return stack, bands


def run_analyze(logit):
    config = load_config()
    stack, bands = synthetic_stack(config)
    grid = Grid(0, 640, 64, 64, 10, "EPSG:32722")
    stats = {"mean": [0.0] * 24, "std": [1.0] * 24}
    return analyze(ConstantLogit(logit), stats, stack, bands, grid, config, 0.5, 1.0, torch.device("cpu"))


def test_hectares_follow_the_forest_mask_and_hansen_years():
    result = run_analyze(5.0)
    # Every forest pixel is predicted; 100 pixels were lost in 2015, so 2048 - 100 remain
    # (19.48 ha, reported to one decimal).
    assert result.hectares["predicted_loss_ha"] == pytest.approx(19.5)
    assert result.hectares["predicted_on_hansen_loss_between_years_ha"] == pytest.approx(4.0)
    assert result.hectares["predicted_on_hansen_loss_in_edge_years_ha"] == pytest.approx(1.0)
    assert result.hectares["hansen_loss_between_years_ha"] == pytest.approx(4.0)
    assert result.hectares["area_ha"] == pytest.approx(41.0)


def test_no_loss_when_the_model_is_confident_nothing_changed():
    assert run_analyze(-5.0).hectares["predicted_loss_ha"] == 0


def test_missing_imagery_is_never_reported_as_loss():
    config = load_config()
    stack, bands = synthetic_stack(config)
    stack[0, :, :16] = NODATA
    grid = Grid(0, 640, 64, 64, 10, "EPSG:32722")
    stats = {"mean": [0.0] * 24, "std": [1.0] * 24}
    result = analyze(ConstantLogit(5.0), stats, stack, bands, grid, config, 0.5, 1.0, torch.device("cpu"))
    assert not result.loss[:, :16].any()
    assert result.quality["missing_image_fraction"] == pytest.approx(0.25)


def test_band_names_follow_the_requested_years():
    config = with_years(load_config(), 2021, 2023)
    bands = stack_band_names(config)
    assert bands[0] == "B2_2021" and "VH_2023" in bands


def test_hansen_release_year_is_read_from_the_asset():
    assert hansen_last_year("UMD/hansen/global_forest_change_2025_v1_13") == 2025


def test_box_around_a_point_has_the_requested_size():
    grid = grid_for_bbox(bbox_around(-52.26, -6.79, 5), "EPSG:32722", 10)
    assert grid.width == pytest.approx(500, abs=3) and grid.height == pytest.approx(500, abs=3)


def test_requests_are_checked_before_using_quota():
    app, config = load_app_config(), load_config()
    request = validate_request(app, config, app.default_bbox, 2020, 2022)
    assert request.grid.area_km2 < app.max_area_km2

    with pytest.raises(ConfigError, match="limit"):
        validate_request(app, config, BBox(-52.5, -7.0, -52.0, -6.5), 2020, 2022)
    with pytest.raises(ConfigError, match="Years"):
        validate_request(app, config, app.default_bbox, 2022, 2020)
    with pytest.raises(ConfigError, match="Years"):
        validate_request(app, config, app.default_bbox, 2017, 2019)

    far = validate_request(app, config, BBox(-60.0, -10.0, -59.96, -9.96), 2021, 2024)
    assert any("outside the training region" in w for w in far.warnings)
    assert any("3-year gap" in w for w in far.warnings)
