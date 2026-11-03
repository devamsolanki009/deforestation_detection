"""Predict a whole image of any size with overlapping, blended tiles.

The model sees 256-pixel tiles. Predictions are least reliable near a tile's edge,
where the network lacks context, so tiles overlap and each one's output is weighted
by a window that fades toward its edges. The weighted average leaves no visible seams.
"""

from __future__ import annotations

import numpy as np
import torch


def blend_window(tile: int, overlap: int) -> np.ndarray:
    """Weight 1 in the tile's centre, fading linearly to near 0 across the overlap."""
    ramp = np.ones(tile, dtype=np.float32)
    if overlap > 0:
        edge = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
        ramp[:overlap] = edge
        ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Tile offsets covering 0..length; the last tile is pushed back to end exactly at length."""
    if length <= tile:
        return [0]
    return list(range(0, length - tile, stride)) + [length - tile]


@torch.no_grad()
def predict_tiled(
    model: torch.nn.Module, image: np.ndarray, device: torch.device,
    tile: int = 256, overlap: int = 64, batch_size: int = 8,
) -> np.ndarray:
    """Return the loss probability for every pixel of a normalized (bands, H, W) image."""
    if not 0 <= overlap < tile // 2:
        raise ValueError(f"overlap must be between 0 and half the tile, got {overlap}")
    _, height, width = image.shape

    # Images smaller than a tile are padded with 0, the band mean, as for missing pixels.
    padded_h, padded_w = max(height, tile), max(width, tile)
    if (padded_h, padded_w) != (height, width):
        image = np.pad(image, ((0, 0), (0, padded_h - height), (0, padded_w - width)))

    window = blend_window(tile, overlap)
    total = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight = np.zeros((padded_h, padded_w), dtype=np.float32)
    stride = tile - overlap
    positions = [(r, c) for r in tile_starts(padded_h, tile, stride) for c in tile_starts(padded_w, tile, stride)]

    for start in range(0, len(positions), batch_size):
        batch = positions[start:start + batch_size]
        x = torch.from_numpy(np.stack([image[:, r:r + tile, c:c + tile] for r, c in batch])).to(device)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)
        probabilities = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        for (r, c), probability in zip(batch, probabilities):
            total[r:r + tile, c:c + tile] += probability * window
            weight[r:r + tile, c:c + tile] += window

    return (total / weight)[:height, :width]
