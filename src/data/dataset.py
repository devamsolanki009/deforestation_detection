"""PyTorch dataset that reads one patch from disk per item."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

STATS_FILE = "band_stats.json"


def read_metadata(patch_dir: Path, split: str) -> list[dict]:
    with (patch_dir / "metadata.csv").open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def compute_band_stats(patch_dir: Path, records: list[dict], nodata: int) -> dict:
    """Per-band mean and standard deviation, streamed one patch at a time.

    Uses training patches only, so no information leaks from val or test.
    """
    total = squares = count = None
    for record in records:
        image = np.load(patch_dir / f"{record['patch_id']}.npz")["image"].astype(np.float64)
        valid = image != nodata
        values = np.where(valid, image, 0)
        if total is None:
            total, squares, count = np.zeros(len(image)), np.zeros(len(image)), np.zeros(len(image))
        total += values.sum(axis=(1, 2))
        squares += (values ** 2).sum(axis=(1, 2))
        count += valid.sum(axis=(1, 2))

    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist(), "patches": len(records)}


def load_or_compute_band_stats(patch_dir: Path, nodata: int) -> dict:
    """Cache the stats next to the patches; delete the file after re-cutting patches."""
    path = patch_dir / STATS_FILE
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    stats = compute_band_stats(patch_dir, read_metadata(patch_dir, "train"), nodata)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def augment(image: np.ndarray, label: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random 90-degree rotation and flip, applied identically to image and label.

    Satellite views have no fixed "up", so all eight orientations are realistic.
    """
    turns = int(rng.integers(4))
    image, label = np.rot90(image, turns, axes=(1, 2)), np.rot90(label, turns)
    if rng.random() < 0.5:
        image, label = image[:, :, ::-1], label[:, ::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(label)


class PatchDataset(Dataset):
    def __init__(self, patch_dir: Path, split: str, band_stats: dict, nodata: int, train: bool = False):
        self.patch_dir = patch_dir
        self.records = read_metadata(patch_dir, split)
        self.mean = np.asarray(band_stats["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(band_stats["std"], dtype=np.float32)[:, None, None]
        self.nodata = nodata
        self.train = train
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = np.load(self.patch_dir / f"{self.records[index]['patch_id']}.npz")
        raw, label = patch["image"], patch["label"]

        image = (raw.astype(np.float32) - self.mean) / self.std
        image[raw == self.nodata] = 0  # missing pixels become the band mean
        if self.train:
            image, label = augment(image, label, self.rng)

        # Label stays 0, 1 or 255; the losses and metrics skip 255.
        return torch.from_numpy(image), torch.from_numpy(label.astype(np.float32))[None]


def worker_init(worker_id: int) -> None:
    """Give each loader worker its own augmentation stream."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)


def loss_patch_sampler(records: list[dict], min_fraction: float, weight: float, seed: int) -> WeightedRandomSampler:
    """Draw patches with meaningful loss more often, keeping epoch length unchanged."""
    weights = []
    for record in records:
        labeled = int(record["total_pixels"]) - int(record["ignored_pixels"])
        share = int(record["loss_pixels"]) / max(1, labeled)
        weights.append(weight if share >= min_fraction else 1.0)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True, generator=generator)
