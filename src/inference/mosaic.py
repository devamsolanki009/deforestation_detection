"""Stitch patch predictions into one map, and post-process detections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from src.data.labels import IGNORE, LOSS

# 8-connectivity: pixels touching at a corner count as one detection.
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@dataclass
class Mosaic:
    """Probabilities, labels and the forest mask for a split, laid out on the patch grid.

    Cells with no patch are IGNORE in the label, so metrics skip them.
    """

    probability: np.ndarray  # float16, (rows, cols)
    label: np.ndarray        # uint8: 0, 1 or IGNORE
    forest: np.ndarray       # bool: forest when the before year began
    left: float              # map x of column 0
    top: float               # map y of row 0
    scale_m: float

    def patch_slice(self, record: dict, size: int) -> tuple[slice, slice]:
        row = round((self.top - float(record["top"])) / self.scale_m)
        col = round((float(record["left"]) - self.left) / self.scale_m)
        return slice(row, row + size), slice(col, col + size)

    def save(self, path) -> None:
        np.savez_compressed(path, probability=self.probability, label=self.label, forest=self.forest,
                            left=self.left, top=self.top, scale_m=self.scale_m)

    @classmethod
    def load(cls, path) -> "Mosaic":
        data = np.load(path)
        return cls(data["probability"], data["label"], data["forest"],
                   float(data["left"]), float(data["top"]), float(data["scale_m"]))


class MosaicBuilder:
    def __init__(self, records: list[dict], size: int, scale_m: float):
        self.size = size
        left = min(float(r["left"]) for r in records)
        top = max(float(r["top"]) for r in records)
        width = round((max(float(r["right"]) for r in records) - left) / scale_m)
        height = round((top - min(float(r["bottom"]) for r in records)) / scale_m)
        self.mosaic = Mosaic(
            probability=np.zeros((height, width), dtype=np.float16),
            label=np.full((height, width), IGNORE, dtype=np.uint8),
            forest=np.zeros((height, width), dtype=bool),
            left=left, top=top, scale_m=scale_m,
        )

    def add(self, record: dict, probability: np.ndarray, label: np.ndarray, forest: np.ndarray) -> None:
        rows, cols = self.mosaic.patch_slice(record, self.size)
        self.mosaic.probability[rows, cols] = probability
        self.mosaic.label[rows, cols] = label
        self.mosaic.forest[rows, cols] = forest


def remove_small_detections(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop connected groups of predicted loss smaller than min_pixels."""
    if min_pixels <= 1:
        return mask
    components, count = ndimage.label(mask, structure=EIGHT_CONNECTED)
    if count == 0:
        return mask
    sizes = np.bincount(components.ravel())
    keep = sizes >= min_pixels
    keep[0] = False  # background
    return keep[components]


def predict_loss(mosaic: Mosaic, threshold: float, min_pixels: int, use_forest_mask: bool) -> np.ndarray:
    """Turn probabilities into a loss mask: threshold, optional forest mask, then size filter.

    The forest mask goes first, so a detection that spills into old pasture is judged
    only by its forest part when small detections are removed.
    """
    predicted = mosaic.probability >= threshold
    if use_forest_mask:
        predicted &= mosaic.forest
    return remove_small_detections(predicted, min_pixels)


def confusion_counts(predicted: np.ndarray, label: np.ndarray) -> tuple[int, int, int]:
    """True positives, false positives and false negatives, skipping IGNORE pixels."""
    valid = label != IGNORE
    actual = label == LOSS
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & valid & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    return tp, fp, fn


def scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
    }
