"""R3 Tier 2: three temporal grayscale channels -> 64-number embedding."""
from __future__ import annotations

from vio_drift.models.dl.cnn1d import EMBEDDING_DIM, RegressionHead
import torch
from torch import nn


class CameraCNNEncoder(nn.Module):
    """Three stride-2 convolution blocks and global average pooling.

    GroupNorm has no flight-level running statistics and works with small
    batches. The three frames are temporal channels, not RGB channels.
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        layers = []
        channels = 3
        for width in (16, 32, EMBEDDING_DIM):
            layers.append(nn.Sequential(
                nn.Conv2d(channels, width, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(4, width), nn.ReLU(), nn.Dropout2d(dropout),
            ))
            channels = width
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, cam: torch.Tensor) -> torch.Tensor:
        if cam.ndim != 4 or tuple(cam.shape[1:]) != (3, 64, 64):
            raise ValueError(f"cam must be (batch,3,64,64), got {tuple(cam.shape)}")
        if not torch.is_floating_point(cam):
            raise ValueError("cam must be floating point in [0,1]; divide uint8 arrays by 255")
        return self.pool(self.blocks(cam)).flatten(1)


class CameraCNN(nn.Module):
    """E5 camera-only regressor; uses R2's unchanged shared regression head."""
    def __init__(self, dropout: float = 0.3, out_dim: int = 1):
        super().__init__()
        self.encoder = CameraCNNEncoder(dropout)
        self.head = RegressionHead(EMBEDDING_DIM, out_dim)

    def encode(self, cam: torch.Tensor) -> torch.Tensor:
        return self.encoder(cam)

    def forward(self, cam: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(cam))
