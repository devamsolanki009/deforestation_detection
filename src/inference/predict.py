"""Load a trained checkpoint and predict loss probabilities for patches."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PatchDataset
from src.models.factory import build_model
from src.training.config import ModelConfig


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Return the model in eval mode and the raw checkpoint, which holds the band stats."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = ModelConfig(**{**checkpoint["model_config"], "encoder_weights": None})
    model = build_model(model_config, in_channels=len(checkpoint["band_stats"]["mean"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def predict_patches(
    model: torch.nn.Module, dataset: PatchDataset, device: torch.device, batch_size: int = 16, num_workers: int = 4,
) -> Iterator[tuple[dict, np.ndarray, np.ndarray]]:
    """Yield (record, probability, label) per patch, in dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    index = 0
    for images, labels in loader:
        with torch.amp.autocast(device.type):
            logits = model(images.to(device, non_blocking=True))
        probabilities = torch.sigmoid(logits.float())[:, 0].cpu().numpy().astype(np.float16)
        for probability, label in zip(probabilities, labels[:, 0].numpy().astype(np.uint8)):
            yield dataset.records[index], probability, label
            index += 1
