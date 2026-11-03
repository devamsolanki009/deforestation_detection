"""Build segmentation models from a training config."""

from __future__ import annotations

import segmentation_models_pytorch as smp
from torch import nn

from src.models.siamese import SiameseUnet
from src.training.config import ModelConfig


def build_model(config: ModelConfig, in_channels: int) -> nn.Module:
    """Build a model whose input is both dates stacked, before date first.

    unet          early fusion: one network sees all channels from the first layer
    siamese_unet  each date passes through a shared encoder; features are fused per scale

    With pretrained weights and more than 3 input channels, smp reuses the RGB filters
    of the first layer across the extra channels.
    """
    if config.arch == "siamese_unet":
        if in_channels % 2:
            raise ValueError(f"siamese_unet needs an even channel count (two dates), got {in_channels}")
        return SiameseUnet(config.encoder, config.encoder_weights, channels_per_date=in_channels // 2)
    if config.arch != "unet":
        raise ValueError(f"Unknown model arch '{config.arch}'")
    return smp.Unet(
        encoder_name=config.encoder,
        encoder_weights=config.encoder_weights,
        in_channels=in_channels,
        classes=1,  # one logit per pixel: forest loss
    )
