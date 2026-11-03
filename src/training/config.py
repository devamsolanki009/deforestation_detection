"""Load and validate a training config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config import ConfigError, _build


@dataclass(frozen=True)
class ModelConfig:
    arch: str
    encoder: str
    encoder_weights: str | None


@dataclass(frozen=True)
class DataConfig:
    batch_size: int
    num_workers: int
    oversample_min_loss_fraction: float
    oversample_weight: float


@dataclass(frozen=True)
class OptimConfig:
    epochs: int
    lr: float
    weight_decay: float
    early_stopping_patience: int


@dataclass(frozen=True)
class LossConfig:
    dice_weight: float
    focal_weight: float


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    area: str
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    loss: LossConfig
    amp: bool
    threshold: float
    seed: int


def load_train_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Training config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"Training config is empty or not a mapping: {path}")

    sections = {"model": ModelConfig, "data": DataConfig, "optim": OptimConfig, "loss": LossConfig}
    built = {name: _build(cls, raw.get(name), name) for name, cls in sections.items()}
    config = _build(TrainConfig, {**raw, **built}, "training config")

    if config.data.batch_size < 1 or config.optim.epochs < 1:
        raise ConfigError("batch_size and epochs must be at least 1")
    if config.data.oversample_weight < 1:
        raise ConfigError("data.oversample_weight must be at least 1")
    if not 0 < config.threshold < 1:
        raise ConfigError(f"threshold must be between 0 and 1, got {config.threshold}")
    return config
