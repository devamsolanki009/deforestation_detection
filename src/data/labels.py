"""Turn raw Hansen bands into training labels."""

from __future__ import annotations

import numpy as np

NO_LOSS = 0
LOSS = 1
IGNORE = 255  # excluded from training and metrics


def derive_label(
    lossyear: np.ndarray, treecover: np.ndarray, before: int, after: int, min_cover: int, nodata: int
) -> np.ndarray:
    """Build a label array from Hansen lossyear (years since 2000) and tree cover.

    Hansen dates loss by calendar year, while composites cover one dry season:
    - Loss strictly between the two years falls between the composites: LOSS, if the
      pixel was forest. Below the forest threshold it is IGNORE, since it is uncertain.
    - Loss in the before or after year may fall on either side of a composite: IGNORE.
    - Earlier loss was gone before the before composite, and later loss came after the
      after composite, so both are NO_LOSS, as are pixels that never lost cover.
    """
    before_code, after_code = before - 2000, after - 2000
    is_middle = (lossyear > before_code) & (lossyear < after_code)
    is_edge = (lossyear == before_code) | (lossyear == after_code)
    is_missing = (lossyear == nodata) | (treecover == nodata)

    label = np.full(lossyear.shape, NO_LOSS, dtype=np.uint8)
    label[is_middle & (treecover >= min_cover)] = LOSS
    label[is_middle & (treecover < min_cover)] = IGNORE
    label[is_edge | is_missing] = IGNORE
    return label
