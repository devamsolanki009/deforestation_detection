import numpy as np
import pytest

from src.data.labels import IGNORE
from src.inference.mosaic import MosaicBuilder, confusion_counts, predict_loss, remove_small_detections, scores


def record(left, top, size=4, scale=10):
    return {"left": left, "top": top, "right": left + size * scale, "bottom": top - size * scale}


def test_patches_land_in_the_right_place():
    records = [record(0, 80), record(40, 80), record(40, 40)]  # missing cell bottom-left
    builder = MosaicBuilder(records, size=4, scale_m=10)
    for value, rec in enumerate(records, start=1):
        builder.add(rec, np.full((4, 4), value / 10, dtype=np.float16), np.full((4, 4), value, dtype=np.uint8),
                    np.ones((4, 4), dtype=bool))

    label = builder.mosaic.label
    assert label.shape == (8, 8)
    assert (label[:4, :4] == 1).all() and (label[:4, 4:] == 2).all() and (label[4:, 4:] == 3).all()
    assert (label[4:, :4] == IGNORE).all()  # no patch there, so metrics skip it


def test_small_detections_are_removed():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:2, 0:2] = True   # 4 pixels
    mask[5:9, 5:9] = True   # 16 pixels
    cleaned = remove_small_detections(mask, min_pixels=5)
    assert cleaned.sum() == 16 and not cleaned[0, 0]


def test_diagonal_pixels_count_as_one_detection():
    mask = np.eye(5, dtype=bool)  # five pixels touching only at corners
    assert remove_small_detections(mask, min_pixels=5).sum() == 5


def test_empty_mask_stays_empty():
    mask = np.zeros((4, 4), dtype=bool)
    assert not remove_small_detections(mask, min_pixels=3).any()


def test_counts_skip_ignored_pixels():
    predicted = np.array([True, True, False, True])
    label = np.array([1, 0, 1, IGNORE], dtype=np.uint8)
    assert confusion_counts(predicted, label) == (1, 1, 1)
    result = scores(1, 1, 1)
    assert result["iou"] == pytest.approx(1 / 3) and result["f1"] == pytest.approx(0.5)


def test_forest_mask_applies_before_size_filter():
    records = [record(0, 80)]
    builder = MosaicBuilder(records, size=4, scale_m=10)
    forest = np.zeros((4, 4), dtype=bool)
    forest[:, :2] = True  # left half was forest
    builder.add(records[0], np.full((4, 4), 0.9, dtype=np.float16), np.zeros((4, 4), dtype=np.uint8), forest)

    # A detection covering the whole patch keeps only its forest half.
    assert predict_loss(builder.mosaic, 0.5, 1, use_forest_mask=True).sum() == 8
    assert predict_loss(builder.mosaic, 0.5, 1, use_forest_mask=False).sum() == 16
    # The forest half alone (8 px) is now below a 9-pixel minimum, so nothing survives.
    assert not predict_loss(builder.mosaic, 0.5, 9, use_forest_mask=True).any()