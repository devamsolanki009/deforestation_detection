from dataclasses import replace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.config import load_config
from src.data.labels import IGNORE, LOSS
from src.data.patches import assign_splits, cut_file, image_band_indices, iter_windows, split_summary

NODATA = -32768
BANDS = ["B2_2020", "VV_2020", "s2_clear_count_2020", "B2_2022", "VV_2022",
         "s2_clear_count_2022", "lossyear", "treecover2000"]
MANIFEST = {"bands": BANDS, "nodata": NODATA}
CONFIG = load_config()
SIZE = CONFIG.patch.size


def write_raster(path, data):
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0],
        dtype="int16", crs="EPSG:32722", transform=from_origin(600000, 9300000, 10, 10), nodata=NODATA,
    ) as target:
        target.write(data)


def blank_stack(width, height):
    return np.ones((len(BANDS), height, width), dtype=np.int16)


def test_image_bands_exclude_quality_and_hansen():
    assert [BANDS[i] for i in image_band_indices(BANDS)] == ["B2_2020", "VV_2020", "B2_2022", "VV_2022"]


def test_windows_skip_partial_edges():
    windows = list(iter_windows(600, 300, 256, 256))
    assert [(w.col_off, w.row_off) for w in windows] == [(0, 0), (256, 0)]


def test_cut_file_writes_patch_label_and_georeference(tmp_path):
    data = blank_stack(SIZE * 2, SIZE)
    data[BANDS.index("lossyear"), :, :] = 0
    data[BANDS.index("treecover2000"), :, :] = 90
    data[BANDS.index("lossyear"), 5, 7] = 21               # kept loss, first patch
    data[BANDS.index("lossyear"), 5, SIZE + 7] = 20        # ignored, second patch
    path = tmp_path / "stack.tif"
    write_raster(path, data)

    output = tmp_path / "patches"
    output.mkdir()
    records, skipped = cut_file(path, MANIFEST, CONFIG, output)

    assert (len(records), skipped) == (2, 0)
    assert records[0]["loss_pixels"] == 1 and records[0]["ignored_pixels"] == 0
    assert records[1]["loss_pixels"] == 0 and records[1]["ignored_pixels"] == 1
    assert records[0]["top"] == 9300000 and records[0]["left"] == 600000
    assert records[1]["left"] == 600000 + SIZE * 10

    patch = np.load(output / f"{records[0]['patch_id']}.npz")
    assert patch["image"].shape == (4, SIZE, SIZE)
    assert patch["clear_count"].shape == (2, SIZE, SIZE)
    assert patch["label"][5, 7] == LOSS
    assert patch["label"].dtype == np.uint8


def test_cut_file_skips_patches_with_missing_imagery(tmp_path):
    data = blank_stack(SIZE * 2, SIZE)
    data[0, :, :SIZE] = NODATA  # one band missing across the whole first patch
    path = tmp_path / "stack.tif"
    write_raster(path, data)
    output = tmp_path / "patches"
    output.mkdir()

    records, skipped = cut_file(path, MANIFEST, CONFIG, output)
    assert (len(records), skipped) == (1, 1)
    assert records[0]["left"] == 600000 + SIZE * 10


def test_cut_file_rejects_band_count_mismatch(tmp_path):
    path = tmp_path / "stack.tif"
    write_raster(path, np.ones((3, SIZE, SIZE), dtype=np.int16))
    with pytest.raises(ValueError, match="bands"):
        cut_file(path, MANIFEST, CONFIG, tmp_path)


def grid_records(rows, columns=2, loss=lambda row: 100):
    size_m = SIZE * CONFIG.export.scale_m
    return [
        {"patch_id": f"{r}_{c}", "top": 9300000 - r * size_m, "loss_pixels": loss(r),
         "ignored_pixels": 0, "total_pixels": SIZE * SIZE}
        for r in range(rows) for c in range(columns)
    ]


def test_splits_follow_order_and_keep_buffers():
    records = grid_records(40)
    assign_splits(records, CONFIG)

    order = [next(r["split"] for r in records if r["patch_id"] == f"{row}_0") for row in range(40)]
    assert order[0] == "train" and order[-1] == "test"
    # A buffer row always separates two different splits.
    kept = [(i, name) for i, name in enumerate(order) if name != "buffer"]
    for (left_index, left), (right_index, right) in zip(kept, kept[1:]):
        if left != right:
            assert right_index - left_index == 2


def test_a_whole_row_shares_one_split():
    records = grid_records(30, columns=5)
    assign_splits(records, CONFIG)
    for row in range(30):
        assert len({r["split"] for r in records if r["patch_id"].startswith(f"{row}_")}) == 1


def test_splits_balance_loss_not_area():
    # Loss sits entirely in the northern half, so test must still contain some.
    records = grid_records(40, loss=lambda row: 100 if row < 20 else 0)
    assign_splits(records, CONFIG)
    summary = split_summary(records)
    assert summary["test"]["loss_pixels"] > 0
    assert summary["val"]["loss_pixels"] > 0


def test_zero_buffer_drops_nothing():
    config = replace(CONFIG, split=replace(CONFIG.split, buffer_m=0))
    records = grid_records(40)
    assign_splits(records, config)
    assert split_summary(records)["buffer"]["patches"] == 0


def test_summary_excludes_ignored_pixels_from_share():
    records = [{"split": "train", "loss_pixels": 10, "ignored_pixels": 50, "total_pixels": 100}]
    assert split_summary(records)["train"]["loss_share_of_labeled_pixels"] == 0.2
    assert IGNORE == 255  # the value downstream losses must exclude
