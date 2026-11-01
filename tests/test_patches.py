from dataclasses import replace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.config import load_config
from src.data.labels import IGNORE, LOSS
from src.data.patches import assign_splits, cut_mosaic, image_band_indices, source_prefixes, split_summary

NODATA = -32768
BANDS = ["B2_2020", "VV_2020", "s2_clear_count_2020", "B2_2022", "VV_2022",
         "s2_clear_count_2022", "lossyear", "treecover2000"]
MANIFEST = {"bands": BANDS, "nodata": NODATA}
CONFIG = load_config()
SIZE = CONFIG.patch.size
SIZE_M = SIZE * CONFIG.export.scale_m
LEFT, TOP = SIZE_M * 234, SIZE_M * 3633  # a corner on the shared patch grid


def write_raster(path, data, left=LEFT, top=TOP):
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0],
        dtype="int16", crs="EPSG:32722", transform=from_origin(left, top, 10, 10), nodata=NODATA,
    ) as target:
        target.write(data)


def blank_stack(width, height):
    data = np.ones((len(BANDS), height, width), dtype=np.int16)
    data[BANDS.index("lossyear")] = 0
    data[BANDS.index("treecover2000")] = 90
    return data


def cut(tmp_path, paths):
    output = tmp_path / "patches"
    output.mkdir()
    records, skipped = cut_mosaic(paths, MANIFEST, CONFIG, output)
    return records, skipped, output


def test_image_bands_exclude_quality_and_hansen():
    assert [BANDS[i] for i in image_band_indices(BANDS)] == ["B2_2020", "VV_2020", "B2_2022", "VV_2022"]


def test_cut_writes_patch_label_and_georeference(tmp_path):
    data = blank_stack(SIZE * 2, SIZE)
    data[BANDS.index("lossyear"), 5, 7] = 21               # kept loss, first patch
    data[BANDS.index("lossyear"), 5, SIZE + 7] = 20        # ignored, second patch
    path = tmp_path / "stack.tif"
    write_raster(path, data)

    records, skipped, output = cut(tmp_path, [path])

    assert (len(records), skipped) == (2, 0)
    assert records[0]["loss_pixels"] == 1 and records[0]["ignored_pixels"] == 0
    assert records[1]["loss_pixels"] == 0 and records[1]["ignored_pixels"] == 1
    assert records[0]["top"] == TOP and records[0]["left"] == LEFT
    assert records[1]["left"] == LEFT + SIZE_M

    patch = np.load(output / f"{records[0]['patch_id']}.npz")
    assert patch["image"].shape == (4, SIZE, SIZE)
    assert patch["clear_count"].shape == (2, SIZE, SIZE)
    assert patch["label"][5, 7] == LOSS
    assert patch["label"].dtype == np.uint8


def test_cut_skips_patches_with_missing_imagery(tmp_path):
    data = blank_stack(SIZE * 2, SIZE)
    data[0, :, :SIZE] = NODATA  # one band missing across the whole first patch
    path = tmp_path / "stack.tif"
    write_raster(path, data)

    records, skipped, _ = cut(tmp_path, [path])
    assert (len(records), skipped) == (1, 1)
    assert records[0]["left"] == LEFT + SIZE_M


def test_grid_ignores_file_origin(tmp_path):
    # A file that starts 100 m into a cell still lands on the shared grid.
    path = tmp_path / "stack.tif"
    write_raster(path, blank_stack(SIZE * 3, SIZE * 2), left=LEFT + 100, top=TOP - 100)

    records, _, _ = cut(tmp_path, [path])
    assert records
    assert all(r["left"] % SIZE_M == 0 and r["top"] % SIZE_M == 0 for r in records)


def test_patch_across_a_file_seam_is_rebuilt(tmp_path):
    # Two files split one grid cell down the middle; values mark which file a pixel came from.
    half = SIZE // 2
    west, east = blank_stack(half, SIZE), blank_stack(half, SIZE)
    west[0], east[0] = 11, 22
    write_raster(tmp_path / "a.tif", west)
    write_raster(tmp_path / "b.tif", east, left=LEFT + half * 10)

    records, skipped, output = cut(tmp_path, [tmp_path / "a.tif", tmp_path / "b.tif"])
    assert (len(records), skipped) == (1, 0)
    image = np.load(output / f"{records[0]['patch_id']}.npz")["image"]
    assert (image[0, :, :half] == 11).all() and (image[0, :, half:] == 22).all()


def test_overlapping_files_produce_one_patch_per_cell(tmp_path):
    write_raster(tmp_path / "a.tif", blank_stack(SIZE * 2, SIZE))
    write_raster(tmp_path / "b.tif", blank_stack(SIZE * 2, SIZE), left=LEFT + SIZE_M)

    records, _, _ = cut(tmp_path, [tmp_path / "a.tif", tmp_path / "b.tif"])
    ids = [r["patch_id"] for r in records]
    assert len(ids) == len(set(ids)) == 3


def test_overlap_fills_missing_pixels_from_the_other_file(tmp_path):
    first, second = blank_stack(SIZE, SIZE), blank_stack(SIZE, SIZE)
    first[0, :10, :] = NODATA
    second[0] = 5
    write_raster(tmp_path / "a.tif", first)
    write_raster(tmp_path / "b.tif", second)

    records, _, output = cut(tmp_path, [tmp_path / "a.tif", tmp_path / "b.tif"])
    image = np.load(output / f"{records[0]['patch_id']}.npz")["image"]
    assert (image[0, :10, :] == 5).all() and (image[0, 10:, :] == 1).all()


def test_rejects_band_count_mismatch(tmp_path):
    path = tmp_path / "stack.tif"
    write_raster(path, np.ones((3, SIZE, SIZE), dtype=np.int16))
    with pytest.raises(ValueError, match="bands"):
        cut(tmp_path, [path])


def test_rejects_misaligned_pixels(tmp_path):
    path = tmp_path / "stack.tif"
    write_raster(path, blank_stack(SIZE, SIZE), left=LEFT + 3)
    with pytest.raises(ValueError, match="aligned"):
        cut(tmp_path, [path])


def test_full_area_combines_all_four_exports():
    prefixes = source_prefixes(CONFIG, "full")
    assert len(prefixes) == 4
    assert any("_subset_" in p for p in prefixes)
    assert source_prefixes(CONFIG, "subset") == [prefixes[0]]

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


def test_one_heavy_row_cannot_leave_a_split_empty():
    # The real failure: a loss-heavy row landed alone in val and the buffer overwrote it.
    records = grid_records(21, columns=21, loss=lambda row: 3000 if row == 15 else 200)
    assign_splits(records, CONFIG)
    summary = split_summary(records)
    for name in ("train", "val", "test"):
        assert summary[name]["patches"] > 0
        assert summary[name]["loss_pixels"] > 0


def test_too_few_rows_is_reported():
    with pytest.raises(ValueError, match="three splits"):
        assign_splits(grid_records(3), CONFIG)


def test_zero_buffer_drops_nothing():
    config = replace(CONFIG, split=replace(CONFIG.split, buffer_m=0))
    records = grid_records(40)
    assign_splits(records, config)
    assert split_summary(records)["buffer"]["patches"] == 0


def test_summary_excludes_ignored_pixels_from_share():
    records = [{"split": "train", "loss_pixels": 10, "ignored_pixels": 50, "total_pixels": 100}]
    assert split_summary(records)["train"]["loss_share_of_labeled_pixels"] == 0.2
    assert IGNORE == 255  # the value downstream losses must exclude
