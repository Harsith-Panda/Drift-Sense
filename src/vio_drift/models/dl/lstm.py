"""LSTM encoder for one IMU window.

The network reads the 400 steps in order and uses the last hidden state
as the 64-number embedding. The regression head is the same module the
1D-CNN uses.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch import nn

from vio_drift.models.dl.cnn1d import EMBEDDING_DIM, IMU_CHANNELS, RegressionHead


class LSTMEncoder(nn.Module):
    """(batch, 6, time) → (batch, 64), equal to the last layer's final hidden state."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.lstm = nn.LSTM(
            input_size=IMU_CHANNELS,
            hidden_size=EMBEDDING_DIM,
            num_layers=1,
            batch_first=True,
        )
        # A one-layer LSTM ignores nn.LSTM's dropout argument. Apply the
        # configured rate to the 64-d state so E2 uses the same knob as E1.
        self.dropout = nn.Dropout(dropout)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        if imu.ndim != 3 or imu.shape[1] != IMU_CHANNELS:
            raise ValueError(f"IMU must be (batch, 6, time), got {tuple(imu.shape)}")
        sequence = imu.transpose(1, 2)  # (batch, time, 6)
        _output, (h_n, _c_n) = self.lstm(sequence)
        last_state = h_n[-1]
        if last_state.shape[-1] != EMBEDDING_DIM:
            raise RuntimeError("last hidden state must be the 64-d embedding")
        return self.dropout(last_state)


class LSTMRegressor(nn.Module):
    """E2 model: LSTM encoder plus the shared regression head."""

    def __init__(self, dropout: float = 0.3, out_dim: int = 1):
        super().__init__()
        self.encoder = LSTMEncoder(dropout)
        self.head = RegressionHead(EMBEDDING_DIM, out_dim)

    def encode(self, imu: torch.Tensor) -> torch.Tensor:
        return self.encoder(imu)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(imu))
