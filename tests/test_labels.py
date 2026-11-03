import numpy as np
import pytest

from src.data.labels import IGNORE, LOSS, NO_LOSS, derive_label, forest_at_start

NODATA = -32768


def label_for(lossyear: int, treecover: int) -> int:
    result = derive_label(
        np.array([[lossyear]], dtype=np.int16), np.array([[treecover]], dtype=np.int16),
        before=2020, after=2022, min_cover=50, nodata=NODATA,
    )
    return int(result[0, 0])


@pytest.mark.parametrize(
    ("lossyear", "treecover", "expected"),
    [
        (21, 90, LOSS),         # between the composites, in forest
        (21, 50, LOSS),         # exactly at the threshold
        (21, 49, IGNORE),       # between the composites, below the forest threshold
        (20, 90, IGNORE),       # before year, could be either side of the composite
        (22, 90, IGNORE),       # after year, could be either side of the composite
        (20, 10, IGNORE),
        (19, 90, NO_LOSS),      # already gone before the first composite
        (23, 90, NO_LOSS),      # happened after the second composite
        (0, 90, NO_LOSS),       # forest that stayed
        (0, 0, NO_LOSS),        # never forest
        (NODATA, 90, IGNORE),   # missing Hansen data
        (0, NODATA, IGNORE),
    ],
)
def test_label_rules(lossyear, treecover, expected):
    assert label_for(lossyear, treecover) == expected


def test_forest_at_start():
    lossyear = np.array([[0, 19, 20, 21, 25, 0, NODATA]], dtype=np.int16)
    treecover = np.array([[90, 90, 90, 90, 90, 40, 90]], dtype=np.int16)
    forest = forest_at_start(lossyear, treecover, before=2020, min_cover=50, nodata=NODATA)
    # Lost in 2019 means already gone; 40% cover is not forest; missing data is not forest.
    assert forest.tolist() == [[True, False, True, True, True, False, False]]


def test_wider_year_gap_keeps_all_middle_years():
    lossyear = np.array([[19, 20, 21, 22, 23, 24]], dtype=np.int16)
    treecover = np.full(lossyear.shape, 90, dtype=np.int16)
    label = derive_label(lossyear, treecover, before=2019, after=2024, min_cover=50, nodata=NODATA)
    assert label.tolist() == [[IGNORE, LOSS, LOSS, LOSS, LOSS, IGNORE]]
