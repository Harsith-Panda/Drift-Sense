"""Fold-local normalization; no torch dependency and no fitting during transform."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vio_drift.features.build_features import feature_names


def _array(values, ndim, width):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != ndim or values.shape[-1] != width or values.size == 0:
        raise ValueError(f"Expected nonempty {ndim}D array with last dimension {width}")
    if not np.isfinite(values).all():
        raise ValueError("Normalization requires finite values")
    return values


def _moments(values, axes):
    mean = values.mean(axis=axes)
    std = values.std(axis=axes)  # population std (ddof=0)
    return mean, np.where(std <= 1e-12, 1.0, std)


@dataclass
class FoldNormalizer:
    fold_id: str
    train_keys: list
    feature_order: list
    imu_mean: np.ndarray
    imu_std: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def transform_imu(self, values):
        """Preserve (N,T,6); channel order gx,gy,gz,ax,ay,az."""
        values = _array(values, 3, 6)
        return ((values - self.imu_mean) / self.imu_std).astype(np.float32)

    def transform_features(self, frame):
        """Select named feature columns only; labels and IDs never enter the scaler."""
        values = _array(frame[self.feature_order].to_numpy(), 2, 104)
        return ((values - self.feature_mean) / self.feature_std).astype(np.float32)

    def save(self, path):
        payload = dict(schema_version=1, fold_id=self.fold_id,
                       train_keys=self.train_keys, feature_order=self.feature_order)
        for name in ("imu_mean", "imu_std", "feature_mean", "feature_std"):
            payload[name] = getattr(self, name).tolist()
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path, expected_fold_id):
        payload = json.loads(Path(path).read_text())
        if payload.pop("schema_version") != 1 or payload["fold_id"] != expected_fold_id:
            raise ValueError("Normalization schema or fold mismatch")
        if payload["feature_order"] != feature_names():
            raise ValueError("Normalization feature order mismatch")
        for name in ("imu_mean", "imu_std", "feature_mean", "feature_std"):
            width = 6 if name.startswith("imu") else 104
            payload[name] = _array(payload[name], 1, width)
            if name.endswith("std") and (payload[name] <= 0).any():
                raise ValueError("Normalization scales must be positive")
        return cls(**payload)


def fit_fold_normalizer(fold, windows, raw_imu, features):
    """Select ONLY fold['train'] by (seq_id, window_id), then fit statistics.

    Raw array rows correspond to the original windows CSV row order, never
    to window_id values. Feature tables are joined by keys, allowing reordering.
    Repeated IMU samples in overlapping training windows are weighted per
    occurrence, matching the training tensor distribution.
    """
    refs = pd.DataFrame(fold["train"])
    if refs.empty or refs.duplicated(["seq_id", "window_id"]).any():
        raise ValueError("Training references must be nonempty and unique")
    if set(refs.seq_id) != {fold["training_source"]}:
        raise ValueError("Training source mismatch")
    if fold["training_source"] == fold["test_source"]:
        raise ValueError("Test flight cannot be a training source")
    keys = ["seq_id", "window_id"]
    forbidden = {tuple(row[k] for k in keys)
                 for part in ("validation", "test", "excluded_gap", "excluded_subset") for row in fold.get(part, [])}
    if any(tuple(row) in forbidden for row in refs[keys].itertuples(index=False, name=None)):
        raise ValueError("Training references overlap validation/test/excluded windows")
    seq = fold["training_source"]
    metadata = windows[seq]
    raw = np.asarray(raw_imu[seq])
    if raw.ndim != 3 or raw.shape != (len(metadata), 400, 6):
        raise ValueError("Raw IMU must align with metadata rows and have shape (N,400,6)")
    positions = metadata[keys].copy()
    positions["row"] = np.arange(len(metadata))
    selected = refs[keys].merge(positions, on=keys, how="left", validate="one_to_one")
    if selected.row.isna().any():
        raise ValueError("Training window missing from metadata")
    imu_train = _array(raw[selected.row.to_numpy(dtype=int)], 3, 6)
    names = feature_names()
    selected_features = refs[keys].merge(features[seq][[*keys, *names]],
                                        on=keys, how="left", validate="one_to_one")
    feature_train = _array(selected_features[names].to_numpy(), 2, 104)
    imu_mean, imu_std = _moments(imu_train, (0, 1))
    feature_mean, feature_std = _moments(feature_train, 0)
    return FoldNormalizer(fold["fold_id"], refs[keys].to_dict("records"), names,
                          imu_mean, imu_std, feature_mean, feature_std)
