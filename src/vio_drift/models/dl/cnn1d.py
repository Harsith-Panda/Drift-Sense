"""1D-CNN encoder for one IMU window.

Three convolution blocks, each Conv → BatchNorm → ReLU → Dropout, then
global average pooling to a 64-number embedding. Kernels 7, 5 and 3 are
the small local detectors from the Phase 2 plan. With stride 1 and same
padding, the receptive field is 13 samples (65 ms at 200 Hz). Pooling,
not a larger field, is what lets a detection anywhere in the 400-step
window reach the embedding.

The regression head is the single head shared with the LSTM.
"""
from __future__ import annotations

import os

# Conda NumPy and the pip torch wheel each ship libomp. Set this before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch import nn

IMU_CHANNELS = 6
EMBEDDING_DIM = 64
IMU_RATE_HZ = 200
KERNELS = (7, 5, 3)
CHANNELS = (32, 64, 64)


def receptive_field_samples(kernels: tuple[int, ...] = KERNELS) -> int:
    """Input samples that affect one pre-pooling activation, stride 1."""
    field = 1
    for kernel in kernels:
        field += kernel - 1
    return field


def receptive_field_seconds(kernels: tuple[int, ...] = KERNELS, rate_hz: float = IMU_RATE_HZ) -> float:
    return receptive_field_samples(kernels) / rate_hz


class RegressionHead(nn.Module):
    """Map the 64-d embedding to 1 value (magnitude) or 4 (magnitude + vector)."""

    def __init__(self, in_dim: int = EMBEDDING_DIM, out_dim: int = 1):
        super().__init__()
        if in_dim != EMBEDDING_DIM:
            raise ValueError("encoder embedding must be 64")
        if out_dim not in (1, 4):
            raise ValueError("head out_dim must be 1 (magnitude) or 4 (magnitude + vector)")
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.shape[-1] != EMBEDDING_DIM:
            raise ValueError(f"expected embedding (..., 64), got {tuple(embedding.shape)}")
        return self.linear(embedding)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel size must be odd so same padding keeps the length")
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.bn(self.conv(x))))


class CNN1DEncoder(nn.Module):
    """(batch, 6, time) → (batch, 64)."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        blocks = []
        in_channels = IMU_CHANNELS
        for out_channels, kernel in zip(CHANNELS, KERNELS):
            blocks.append(ConvBlock(in_channels, out_channels, kernel, dropout))
            in_channels = out_channels
        if len(blocks) != 3 or in_channels != EMBEDDING_DIM:
            raise RuntimeError("the encoder is three blocks ending at 64 channels")
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        if imu.ndim != 3 or imu.shape[1] != IMU_CHANNELS:
            raise ValueError(f"IMU must be (batch, 6, time), got {tuple(imu.shape)}")
        x = imu
        for block in self.blocks:
            x = block(x)
        return self.pool(x).squeeze(-1)


class CNN1D(nn.Module):
    """E1 model: CNN encoder plus the shared regression head."""

    def __init__(self, dropout: float = 0.3, out_dim: int = 1):
        super().__init__()
        self.encoder = CNN1DEncoder(dropout)
        self.head = RegressionHead(EMBEDDING_DIM, out_dim)

    def encode(self, imu: torch.Tensor) -> torch.Tensor:
        return self.encoder(imu)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(imu))
