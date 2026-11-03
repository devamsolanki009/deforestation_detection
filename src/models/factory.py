"""Build segmentation models from a training config."""

from __future__ import annotations

import segmentation_models_pytorch as smp
from torch import nn

from src.training.config import ModelConfig


def build_model(config: ModelConfig, in_channels: int) -> nn.Module:
    """Early-fusion model: both dates arrive stacked as one multi-channel image.

    With pretrained weights and more than 3 input channels, smp reuses the RGB filters
    of the first layer across the extra channels.
    """
    if config.arch != "unet":
        raise ValueError(f"Unknown model arch '{config.arch}'")
    return smp.Unet(
        encoder_name=config.encoder,
        encoder_weights=config.encoder_weights,
        in_channels=in_channels,
        classes=1,  # one logit per pixel: forest loss
    )
