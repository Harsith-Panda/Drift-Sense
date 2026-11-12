"""R4 hybrid models: learned encoders fused with hand-made and physics knowledge.

Every branch produces a 64-number embedding (the Section 6 contract):

  * IMU branch     R2's LSTM or 1D-CNN encoder on the raw (6, 400) window
  * feature branch a small MLP on the 104 Phase 1 features + the 4 physics
                   numbers (+ optional R3 optical-flow numbers)
  * camera branch  R3's 2D CNN on the (3, 64, 64) start/middle/end frames

Active embeddings are concatenated, mixed by one fusion layer, and mapped by
R2's shared ``RegressionHead`` to 1 value (magnitude) or 4 (magnitude +
dx, dy, dz). Three presets:

  H1  IMU + features + physics, predicted directly             (E3)
  H2  physics prior + learned residual (main hybrid)           (E4)
        y_hat = a * phys + b + s * f(x)
      a, b are least squares on the training flight; the head's last layer
      starts at zero so training begins exactly at the physics answer
  H3  IMU + camera (+ optional features)                       (E6)

Network outputs are in standardised units; ``y_scale`` (from the training
split) converts them to metres, so the head never has to learn the ~1.2 m
offset of ``err_mag_m``. All scaling buffers are set by
``hybrid_train.fit_scaling`` from the training split only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn

from vio_drift.models.dl.camera_cnn import CameraCNNEncoder
from vio_drift.models.dl.cnn1d import EMBEDDING_DIM, CNN1DEncoder, RegressionHead
from vio_drift.models.dl.lstm import LSTMEncoder

N_FEATURES = 104
N_PHYS = 4  # phys_dx, phys_dy, phys_dz, phys_mag (dataset order)
PHYS_MAG, PHYS_VEC = 3, slice(0, 3)
IMU_ENCODERS = {"lstm": LSTMEncoder, "cnn1d": CNN1DEncoder}


@dataclass
class PhysicsPrior:
    """Per-output linear map of the physics feature: [mag, dx, dy, dz]."""
    a: np.ndarray
    b: np.ndarray

    def predict(self, phys: np.ndarray) -> np.ndarray:
        phys = np.asarray(phys, dtype=np.float64)
        x = np.concatenate([phys[:, [PHYS_MAG]], phys[:, PHYS_VEC]], axis=1)
        return self.a * x + self.b


def fit_physics_prior(phys, y_mag, y_vec) -> PhysicsPrior:
    """Least squares y = a * phys + b per output. Training-flight rows only."""
    phys = np.asarray(phys, dtype=np.float64)
    targets = np.column_stack([np.asarray(y_mag, dtype=np.float64), np.asarray(y_vec, dtype=np.float64)])
    inputs = np.concatenate([phys[:, [PHYS_MAG]], phys[:, PHYS_VEC]], axis=1)
    if len(inputs) < 2 or not (np.isfinite(inputs).all() and np.isfinite(targets).all()):
        raise ValueError("Physics prior needs at least two finite training rows")
    a, b = np.empty(4), np.empty(4)
    for k in range(4):
        a[k], b[k] = np.polyfit(inputs[:, k], targets[:, k], 1)
    return PhysicsPrior(a=a, b=b)


class FeatureEncoder(nn.Module):
    """(batch, 104 + 4 [+ flow]) -> (batch, 64)."""

    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, EMBEDDING_DIM), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridModel(nn.Module):
    def __init__(self, imu_encoder: str | None = "lstm", use_features: bool = True,
                 use_camera: bool = False, residual: bool = False, n_cam_feats: int = 0,
                 dropout: float = 0.3, out_dim: int = 1, zero_init_head: bool | None = None):
        super().__init__()
        if imu_encoder is not None and imu_encoder not in IMU_ENCODERS:
            raise ValueError(f"imu_encoder must be one of {sorted(IMU_ENCODERS)} or None")
        if n_cam_feats and not use_features:
            raise ValueError("camera flow numbers enter through the feature branch")
        if out_dim not in (1, 4):
            raise ValueError("out_dim must be 1 (magnitude) or 4 (magnitude + vector)")
        self.config = dict(imu_encoder=imu_encoder, use_features=use_features, use_camera=use_camera,
                           residual=residual, n_cam_feats=n_cam_feats, dropout=dropout, out_dim=out_dim)
        self.residual, self.out_dim, self.n_cam_feats = residual, out_dim, n_cam_feats
        self.zero_init_head = residual if zero_init_head is None else zero_init_head
        self.imu_encoder = IMU_ENCODERS[imu_encoder](dropout) if imu_encoder else None
        self.feature_encoder = (FeatureEncoder(N_FEATURES + N_PHYS + n_cam_feats, dropout)
                                if use_features else None)
        self.camera_encoder = CameraCNNEncoder(dropout) if use_camera else None
        self.n_branches = sum(m is not None for m in (self.imu_encoder, self.feature_encoder, self.camera_encoder))
        if self.n_branches == 0:
            raise ValueError("enable at least one branch (IMU, features or camera)")
        self.fusion = nn.Sequential(nn.Linear(self.n_branches * EMBEDDING_DIM, EMBEDDING_DIM),
                                    nn.ReLU(), nn.Dropout(dropout))
        self.head = RegressionHead(EMBEDDING_DIM, out_dim)
        # Scaling buffers; set from the training split by fit_scaling().
        self.register_buffer("phys_mean", torch.zeros(N_PHYS))
        self.register_buffer("phys_std", torch.ones(N_PHYS))
        self.register_buffer("prior_a", torch.zeros(out_dim))
        self.register_buffer("prior_b", torch.zeros(out_dim))
        self.register_buffer("prior_set", torch.tensor(False))
        self.register_buffer("y_center", torch.zeros(out_dim))
        self.register_buffer("y_scale", torch.ones(out_dim))
        self._init_head()

    def _init_head(self) -> None:
        if self.zero_init_head:
            nn.init.zeros_(self.head.linear.weight)
            nn.init.zeros_(self.head.linear.bias)

    def reset_parameters(self) -> None:
        """Re-draw all weights from the current RNG state (buffers untouched)."""
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self._init_head()

    def set_prior(self, prior: PhysicsPrior) -> None:
        self.prior_a.copy_(torch.as_tensor(prior.a[: self.out_dim], dtype=torch.float32))
        self.prior_b.copy_(torch.as_tensor(prior.b[: self.out_dim], dtype=torch.float32))
        self.prior_set.fill_(True)

    def physics_prediction(self, phys: torch.Tensor) -> torch.Tensor:
        if not bool(self.prior_set):
            raise RuntimeError("residual model needs a fitted physics prior (fit_scaling/set_prior)")
        x = torch.cat([phys[:, [PHYS_MAG]], phys[:, PHYS_VEC]], dim=1)[:, : self.out_dim]
        return self.prior_a * x + self.prior_b

    def embed(self, batch: dict) -> torch.Tensor:
        parts = []
        if self.imu_encoder is not None:
            parts.append(self.imu_encoder(batch["imu"]))
        if self.feature_encoder is not None:
            phys = (batch["phys"] - self.phys_mean) / self.phys_std
            inputs = [batch["feats"], phys]
            if self.n_cam_feats:
                inputs.append(batch["cam_feats"])
            parts.append(self.feature_encoder(torch.cat(inputs, dim=1)))
        if self.camera_encoder is not None:
            if "cam" in batch:
                parts.append(self.camera_encoder(batch["cam"]))
            else:  # missing camera: neutral embedding, the other branches still predict
                parts.append(batch["phys"].new_zeros(len(batch["phys"]), EMBEDDING_DIM))
        return self.fusion(torch.cat(parts, dim=1))

    def forward(self, batch: dict) -> torch.Tensor:
        """Prediction in metres, shape (batch, out_dim)."""
        z = self.head(self.embed(batch))
        base = self.physics_prediction(batch["phys"]) if self.residual else self.y_center
        return base + self.y_scale * z

    def loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE in standardised units, so all outputs weigh equally."""
        return (((prediction - target) / self.y_scale) ** 2).mean()


PRESETS = {
    "H1": dict(imu_encoder="lstm", use_features=True, use_camera=False, residual=False),
    "H2": dict(imu_encoder="lstm", use_features=True, use_camera=False, residual=True),
    "H3": dict(imu_encoder="lstm", use_features=False, use_camera=True, residual=False),
}


def build_hybrid(kind: str, **overrides) -> HybridModel:
    if kind not in PRESETS:
        raise ValueError(f"unknown hybrid {kind!r}; choose from {sorted(PRESETS)}")
    return HybridModel(**{**PRESETS[kind], **overrides})


__all__ = ["FeatureEncoder", "HybridModel", "PhysicsPrior", "PRESETS", "build_hybrid", "fit_physics_prior"]
