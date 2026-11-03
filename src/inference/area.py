"""Detect forest loss over one area and measure it in hectares.

Shared by the command-line script and the demo app. Post-processing matches the
evaluated pipeline: threshold, forest mask, then removal of detections under 1 ha.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
import torch

from src.config import Config
from src.data.dataset import normalize
from src.data.gee import NODATA
from src.data.labels import forest_at_start
from src.data.patches import image_band_indices
from src.inference.fetch import Grid
from src.inference.mosaic import remove_small_detections
from src.inference.tiled import predict_tiled

SQUARE_M_PER_HA = 10_000
FALSE_COLOR_MAX = 4000  # reflectance x 10,000; matches the evaluation plots
LOW_CLEAR_COUNT = 5     # clear Sentinel-2 views per pixel below which composites get noisy

# Comparison colors, as in the evaluation plots: both agree, model only, Hansen only.
# Model loss on Hansen loss from the before or after year gets its own color, because
# that loss may truly fall between the composites.
AGREE, MODEL_ONLY, HANSEN_ONLY, EDGE_YEAR = (255, 217, 0), (230, 26, 26), (0, 204, 255), (190, 90, 255)
LOSS_COLOR = (230, 26, 26)


@dataclass
class AreaResult:
    grid: Grid
    before: int
    after: int
    probability: np.ndarray  # float32 (H, W)
    loss: np.ndarray         # bool, final loss mask
    forest: np.ndarray       # bool, forest when the before year began (Hansen)
    hansen_loss: np.ndarray  # bool, Hansen loss dated strictly between the years, in forest
    hansen_edge: np.ndarray  # bool, Hansen loss dated to the before or after year
    hectares: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def hansen_last_year(asset: str) -> int:
    """The last loss year in a Hansen release, read from its asset name."""
    match = re.search(r"global_forest_change_(\d{4})", asset)
    if not match:
        raise ValueError(f"Cannot read the release year from {asset}")
    return int(match.group(1))


def analyze(
    model: torch.nn.Module, band_stats: dict, stack: np.ndarray, bands: list[str], grid: Grid,
    config: Config, threshold: float, min_detection_ha: float, device: torch.device, overlap: int = 64,
) -> AreaResult:
    """Run the model on a fetched stack and apply the evaluated post-processing."""
    before, after = config.years.before, config.years.after
    image_indices = image_band_indices(bands)
    raw = stack[image_indices]
    mean = np.asarray(band_stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(band_stats["std"], dtype=np.float32)[:, None, None]
    probability = predict_tiled(model, normalize(raw, mean, std, NODATA), device, config.patch.size, overlap)

    lossyear, treecover = stack[bands.index("lossyear")], stack[bands.index("treecover2000")]
    min_cover = config.labels.forest_min_treecover
    forest = forest_at_start(lossyear, treecover, before, min_cover, NODATA)
    missing_image = (raw == NODATA).any(axis=0)

    pixel_ha = grid.scale_m ** 2 / SQUARE_M_PER_HA
    min_pixels = max(1, round(min_detection_ha / pixel_ha))
    loss = remove_small_detections((probability >= threshold) & forest & ~missing_image, min_pixels)

    # Hansen dates loss by calendar year; only years strictly between the composites
    # match the training label. Loss in the before or after year may sit either side.
    between = (lossyear > before - 2000) & (lossyear < after - 2000)
    edge = (lossyear == before - 2000) | (lossyear == after - 2000)
    later = lossyear > after - 2000
    hansen_loss = between & forest

    def ha(mask: np.ndarray) -> float:
        return round(float(np.count_nonzero(mask)) * pixel_ha, 1)

    hectares = {
        "predicted_loss_ha": ha(loss),
        "predicted_on_hansen_loss_between_years_ha": ha(loss & between),
        "predicted_on_hansen_loss_in_edge_years_ha": ha(loss & edge),
        "predicted_on_hansen_loss_after_period_ha": ha(loss & later),
        "predicted_with_no_hansen_loss_ha": ha(loss & ~between & ~edge & ~later),
        "hansen_loss_between_years_ha": ha(hansen_loss),
        "hansen_loss_in_edge_years_ha": ha(edge & forest),
        "forest_at_start_ha": ha(forest),
        "area_ha": ha(np.ones_like(loss)),
    }

    clear = {year: stack[bands.index(f"s2_clear_count_{year}")] for year in (before, after)}
    quality = {
        "missing_image_fraction": round(float(missing_image.mean()), 4),
        **{f"median_s2_clear_views_{year}": float(np.median(c[c != NODATA])) if (c != NODATA).any() else 0.0
           for year, c in clear.items()},
        "low_clear_fraction": round(float(np.mean(np.minimum(clear[before], clear[after]) < LOW_CLEAR_COUNT)), 4),
    }

    warnings = []
    if quality["missing_image_fraction"] > 0.05:
        warnings.append(f"{quality['missing_image_fraction']:.0%} of pixels lack imagery; no loss is reported there.")
    if quality["low_clear_fraction"] > 0.2:
        warnings.append(f"{quality['low_clear_fraction']:.0%} of pixels had under {LOW_CLEAR_COUNT} cloud-free "
                        "Sentinel-2 views; composites there are noisier.")
    last = hansen_last_year(config.earth_engine.hansen_asset)
    if after - 1 > last:
        warnings.append(f"Hansen data ends in {last}, so the Hansen comparison misses later loss.")

    return AreaResult(grid, before, after, probability, loss, forest, hansen_loss, edge, hectares, quality, warnings)


def false_color(stack: np.ndarray, bands: list[str], year: int) -> np.ndarray:
    """SWIR, NIR, red as uint8 RGB: forest shows green, cleared land pink or brown."""
    channels = [stack[bands.index(f"{band}_{year}")].astype(np.float32) for band in ("B11", "B8", "B4")]
    rgb = np.clip(np.dstack(channels) / FALSE_COLOR_MAX, 0, 1)
    rgb[stack[bands.index(f"B4_{year}")] == NODATA] = 0
    return (rgb * 255).astype(np.uint8)


def loss_overlay(background: np.ndarray, loss: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Predicted loss painted in red over the after-year image."""
    image = background.astype(np.float32)
    image[loss] = (1 - alpha) * image[loss] + alpha * np.asarray(LOSS_COLOR, dtype=np.float32)
    return image.astype(np.uint8)


def comparison_image(background: np.ndarray, result: AreaResult) -> np.ndarray:
    """Model against Hansen on a dimmed grey background."""
    grey = background.mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.6
    image = grey.astype(np.uint8)
    image[result.loss & result.hansen_loss] = AGREE
    image[result.loss & ~result.hansen_loss & ~result.hansen_edge] = MODEL_ONLY
    image[result.loss & result.hansen_edge] = EDGE_YEAR
    image[~result.loss & result.hansen_loss] = HANSEN_ONLY
    return image


def save_geotiff(path: Path, result: AreaResult) -> None:
    """Two bands: the final loss mask (0 or 1) and the probability in percent."""
    grid = result.grid
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=grid.width, height=grid.height, count=2, dtype="uint8",
        crs=grid.crs, transform=grid.transform, compress="deflate",
    ) as dst:
        dst.write(result.loss.astype(np.uint8), 1)
        dst.write(np.round(result.probability * 100).astype(np.uint8), 2)
        dst.set_band_description(1, f"forest_loss_{result.before}_{result.after}")
        dst.set_band_description(2, "loss_probability_percent")
