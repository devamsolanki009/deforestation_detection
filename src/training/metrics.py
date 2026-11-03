"""Pixel metrics for the loss class, skipping ignored pixels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.labels import IGNORE


@dataclass
class LossClassCounts:
    """Running confusion counts for the loss class."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> None:
        valid = labels != IGNORE
        predicted = (torch.sigmoid(logits.float()) >= threshold) & valid
        actual = (labels == 1) & valid
        self.tp += int((predicted & actual).sum())
        self.fp += int((predicted & ~actual).sum())
        self.fn += int((~predicted & actual).sum())

    def compute(self) -> dict[str, float]:
        """Return IoU, F1, precision and recall. A metric with no denominator is 0."""
        tp, fp, fn = self.tp, self.fp, self.fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
            "precision": precision,
            "recall": recall,
        }
