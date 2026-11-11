"""Section 6 Dataset and deterministic CPU DataLoaders for the two easy flights."""
from pathlib import Path
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from vio_drift.data.augment import IMUAugmentation
from vio_drift.data.dl_physics import sha256
from vio_drift.data.normalize import FoldNormalizer
from vio_drift.data.splits import validate_dl_fold
from vio_drift.features.build_features import feature_names
from vio_drift.features.physics import feature_names as physics_names

SEQUENCES = ("MH_01_easy", "MH_02_easy")
KEYS = ["seq_id", "window_id"]


def _join(metadata, table, columns):
    if table.duplicated(KEYS).any() or len(table) != len(metadata):
        raise ValueError("Feature table must have exactly one row per window")
    merged = metadata[KEYS].merge(table[[*KEYS, *columns]], on=KEYS, how="left", validate="one_to_one", sort=False)
    if not np.isfinite(merged[columns].to_numpy()).all():
        raise ValueError("Missing keys or nonfinite feature values")
    return merged


class DriftDataset(Dataset):
    def __init__(self, processed_dir, fold_id, split, *, artifact_dir=None,
                 use_camera=False, augmentation=None, seed=42):
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        self.split, self.seed, self.epoch = split, int(seed), 0
        self.augmentation = augmentation or IMUAugmentation()
        if self.augmentation.enabled and split != "train":
            raise ValueError("Augmentation cannot be enabled for validation/test")
        processed = Path(processed_dir)
        artifacts = Path(artifact_dir) if artifact_dir else processed
        manifest = json.loads((artifacts / "splits_dl.json").read_text())
        matches = [f for f in manifest["folds"] if f["fold_id"] == fold_id]
        if len(matches) != 1:
            raise ValueError("Unknown or duplicated fold")
        fold = matches[0]
        windows = {s: pd.read_csv(processed / f"windows_{s}.csv") for s in SEQUENCES}
        validate_dl_fold(fold, windows, manifest["gap_ns"])
        stats_path = artifacts / "normalization_dl" / f"{fold_id}.json"
        self.normalizer = FoldNormalizer.load(stats_path, fold_id)
        sources = json.loads(stats_path.with_suffix(".sources.json").read_text())
        if sources["statistics_sha256"] != sha256(stats_path):
            raise ValueError("Normalizer checksum mismatch")
        for relative, digest in sources["inputs"].items():
            if sha256(processed.parent.parent / relative) != digest:
                raise ValueError("Normalization source changed; regenerate R1 artifacts")
        train_keys = [{k: ref[k] for k in KEYS} for ref in fold["train"]]
        if self.normalizer.train_keys != train_keys:
            raise ValueError("Normalizer was fitted on different training keys")
        physics_path = artifacts / "physics_dl" / f"physics_{fold_id}.csv"
        if not physics_path.exists():
            raise FileNotFoundError(f"Run scripts/prepare_r1_data.py: missing {physics_path}")
        physics_meta = json.loads(physics_path.with_suffix(".json").read_text())
        provenance = physics_meta["provenance"]
        if provenance["fold_id"] != fold_id or provenance["train"] != fold["train"]:
            raise ValueError("Physics calibration uses different training windows")
        if physics_meta["table_sha256"] != sha256(physics_path):
            raise ValueError("Physics table checksum mismatch")
        # Detect stale source data; paths are repository-relative in the producer.
        root = processed.parent.parent
        for relative, digest in provenance["inputs"].items():
            if sha256(root / relative) != digest:
                raise ValueError("Physics source changed; regenerate R1 artifacts")
        physics = pd.read_csv(physics_path)
        if len(physics) != sum(map(len, windows.values())) or physics.duplicated(KEYS).any():
            raise ValueError("Physics table key coverage mismatch")
        seq = fold["test_source"] if split == "test" else fold["training_source"]
        metadata = windows[seq]
        raw = np.load(processed / f"windows_raw_{seq}.npy", allow_pickle=False)
        if raw.shape != (len(metadata), 400, 6):
            raise ValueError("Raw array must match windows CSV order and shape (N,400,6)")
        features = pd.read_csv(processed / f"features_{seq}.csv")
        features = _join(metadata, features, feature_names())
        phys = _join(metadata, physics.loc[physics.seq_id == seq], physics_names())
        wanted = {ref["window_id"] for ref in fold[split]}
        # Dataset order always equals the original CSV order, even if split refs differ.
        positions = np.flatnonzero(metadata.window_id.isin(wanted).to_numpy())
        self.meta = metadata.iloc[positions][[*KEYS, "t_start_ns"]].to_dict("records")
        self.imu = self.normalizer.transform_imu(raw[positions])
        self.feats = self.normalizer.transform_features(features.iloc[positions])
        self.phys = phys.iloc[positions][physics_names()].to_numpy(dtype=np.float32)
        labels = metadata.iloc[positions][["err_mag_m", "err_dx", "err_dy", "err_dz"]].to_numpy(dtype=np.float32)
        if not np.isfinite(labels).all():
            raise ValueError("Nonfinite labels")
        self.labels = labels
        self.cam = None
        if use_camera:
            camera_path = processed / "camera" / f"cam_{seq}.npy"
            if not camera_path.exists():
                raise FileNotFoundError(f"R3 camera preprocessing required: {camera_path}")
            camera = np.load(camera_path, mmap_mode="r", allow_pickle=False)
            if camera.shape != (len(metadata), 3, 64, 64) or camera.dtype != np.uint8:
                raise ValueError("R3 camera arrays must be uint8 (N,3,64,64) in windows CSV order")
            camera_keys = pd.read_csv(processed / "camera" / f"camera_feats_{seq}.csv", usecols=KEYS)
            if not camera_keys.equals(metadata[KEYS]):
                raise ValueError("Camera key rows must match windows CSV order exactly")
            self.cam = np.asarray(camera[positions], dtype=np.float32) / 255.0
        # File-independent deterministic stream per fold/sample/epoch.
        self.fold_seed = int.from_bytes(hashlib.sha256(fold_id.encode()).digest()[:4], "little")

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        imu = self.imu[index].copy()
        if self.augmentation.enabled:
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.seed, self.fold_seed, self.epoch, int(self.meta[index]["window_id"])]))
            imu = self.augmentation.apply(imu, self.normalizer.imu_std, rng, split=self.split)
        sample = {"imu": torch.from_numpy(imu.T.copy()),
                  "feats": torch.from_numpy(self.feats[index].copy()),
                  "phys": torch.from_numpy(self.phys[index].copy()),
                  "y_mag": torch.tensor(self.labels[index, 0], dtype=torch.float32),
                  "y_vec": torch.from_numpy(self.labels[index, 1:].copy()),
                  "meta": dict(self.meta[index])}
        if self.cam is not None:
            sample["cam"] = torch.from_numpy(self.cam[index].copy())
        return sample


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_dataloader(dataset, batch_size=32, seed=42, num_workers=0):
    """Shuffle train only; no dropped samples. Set dataset epoch before iterating.

    Workers are recreated each iteration so set_epoch is propagated. Recreate
    the loader with the same seed to replay a run; epoch-specific augmentations
    are independent of worker scheduling. No CUDA initialization is needed.
    """
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=dataset.split == "train",
                      num_workers=num_workers, generator=generator,
                      worker_init_fn=seed_worker, persistent_workers=False, drop_last=False)
