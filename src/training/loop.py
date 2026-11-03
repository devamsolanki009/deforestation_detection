"""One training epoch and one evaluation pass."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import LossClassCounts


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    max_steps: int | None = None,
) -> float:
    """Return the mean training loss. max_steps cuts the epoch short for timing runs."""
    model.train()
    total, steps = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp):
            loss = loss_fn(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return total / max(1, steps)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp: bool,
    threshold: float,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Return the mean loss plus loss-class IoU, F1, precision and recall."""
    model.eval()
    counts = LossClassCounts()
    total, steps = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp):
            logits = model(images)
        total += loss_fn(logits, labels).item()
        counts.update(logits, labels, threshold)
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return {"loss": total / max(1, steps), **counts.compute()}
