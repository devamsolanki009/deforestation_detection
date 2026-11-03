"""Siamese U-Net: one shared encoder per date, fused features into a U-Net decoder."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn


class FusionBlock(nn.Module):
    """Combine one scale of before and after features back to the encoder's width.

    Keeps both dates as well as their absolute difference: the difference alone
    cannot tell forest loss from regrowth, because it drops the direction of change.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat([before, after, (after - before).abs()], dim=1))


class SiameseUnet(nn.Module):
    """Input is both dates stacked: the first half of the channels is the before date."""

    def __init__(self, encoder: str, encoder_weights: str | None, channels_per_date: int):
        super().__init__()
        # Borrow the parts of a standard U-Net so only the fusion differs from the baseline.
        base = smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=channels_per_date, classes=1)
        self.encoder, self.decoder, self.segmentation_head = base.encoder, base.decoder, base.segmentation_head
        self.channels_per_date = channels_per_date
        # Stage 0 of the encoder output is the raw input, which the U-Net decoder does not use.
        self.fusions = nn.ModuleList(FusionBlock(c) for c in self.encoder.out_channels[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        before, after = x[:, :self.channels_per_date], x[:, self.channels_per_date:]
        # One encoder pass over both dates: the same weights, and faster than two calls.
        features = self.encoder(torch.cat([before, after], dim=0))
        split = [f.chunk(2, dim=0) for f in features]
        fused = [split[0][1]] + [fuse(b, a) for fuse, (b, a) in zip(self.fusions, split[1:])]
        return self.segmentation_head(self.decoder(fused))
