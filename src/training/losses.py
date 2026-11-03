"""Dice plus focal loss, skipping ignored pixels."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

from src.data.labels import IGNORE
from src.training.config import LossConfig


class DiceFocalLoss(nn.Module):
    """Dice rewards overlap with the rare loss class; focal down-weights easy pixels.

    Both suit heavy imbalance, where plain cross-entropy would favour predicting no loss.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.dice = smp.losses.DiceLoss("binary", from_logits=True, ignore_index=IGNORE)
        self.focal = smp.losses.FocalLoss("binary", ignore_index=IGNORE)
        self.dice_weight = config.dice_weight
        self.focal_weight = config.focal_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Losses run in float32 so mixed precision cannot underflow them.
        logits = logits.float()
        return self.dice_weight * self.dice(logits, labels) + self.focal_weight * self.focal(logits, labels)
