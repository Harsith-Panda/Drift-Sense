"""Verify Phase 1 artifacts, regenerate DL splits and fit fold-local scalers.

Run from anywhere: python scripts/prepare_dl_data.py
No Phase 1 files are regenerated; no models are trained.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vio_drift.data.splits import load_splits, make_dl_splits
from vio_drift.data.normalize import fit_fold_normalizer, FoldNormalizer
from vio_drift.features.build_features import feature_names


def main():
    config = yaml.safe_load((ROOT / "configs/dl_config.yaml").read_text())
    processed = ROOT / config["data"]["processed_dir"]
    windows, raw, features = {}, {}, {}
    expected = {"MH_01_easy": 720, "MH_02_easy": 592}
    keys = ["seq_id", "window_id"]
    for seq in config["data"]["sequences"]:
        windows[seq] = pd.read_csv(processed / f"windows_{seq}.csv")
        raw[seq] = np.load(processed / f"windows_raw_{seq}.npy", allow_pickle=False)
        features[seq] = pd.read_csv(processed / f"features_{seq}.csv")
        w, x, f = windows[seq], raw[seq], features[seq]
        if len(w) != expected[seq] or x.shape != (len(w), 400, 6):
            raise ValueError(f"{seq}: unexpected window count/shape")
        if list(f.columns) != [*keys, *feature_names(), "err_mag_m"]:
            raise ValueError(f"{seq}: feature schema mismatch")
        joined = w.merge(f, on=keys, validate="one_to_one", suffixes=("_window", "_feature"))
        if len(joined) != len(w) or len(f) != len(w):
            raise ValueError(f"{seq}: feature keys mismatch")
        if not np.allclose(joined.err_mag_m_window, joined.err_mag_m_feature):
            raise ValueError(f"{seq}: labels differ between artifacts")
        for values in (x, w.drop(columns="seq_id").to_numpy(), f.drop(columns="seq_id").to_numpy()):
            if not np.isfinite(values).all():
                raise ValueError(f"{seq}: nonfinite Phase 1 data")
        print(f"{seq}: {len(w)} windows, IMU {x.shape}, {len(feature_names())} features; labels verified")
    outer = load_splits(processed / "splits.json")
    all_keys = pd.concat([w[keys] for w in windows.values()], ignore_index=True)
    for fold in outer:
        physics = pd.read_csv(processed / f"physics_{fold.fold_id}.csv")
        joined = all_keys.merge(physics, on=keys, validate="one_to_one")
        if len(joined) != len(all_keys) or len(physics) != len(all_keys):
            raise ValueError("Physics window keys mismatch")
        if not np.isfinite(physics[["phys_dx", "phys_dy", "phys_dz", "phys_mag"]]).all().all():
            raise ValueError("Nonfinite physics features")
        print(f"physics_{fold.fold_id}.csv: {len(physics)} rows verified (not used for DL scaling)")
    splits = make_dl_splits(windows, outer, config["validation_fraction"],
                            config["validation_gap_seconds"], config["seed"])
    fitted = [(fold, fit_fold_normalizer(fold, windows, raw, features)) for fold in splits["folds"]]
    stats_dir = processed / "normalization_dl"
    stats_dir.mkdir(exist_ok=True)
    (processed / "splits_dl.json").write_text(json.dumps(splits, indent=2) + "\n")
    for fold, scaler in fitted:
        path = stats_dir / f"{fold['fold_id']}.json"
        scaler.save(path)
        loaded = FoldNormalizer.load(path, fold["fold_id"])
        for seq in windows:
            assert loaded.transform_imu(raw[seq]).shape == raw[seq].shape
            assert loaded.transform_features(features[seq]).shape == (len(windows[seq]), 104)
        gap = (min(w["t_start_ns"] for w in fold["validation"]) -
               max(w["t_end_ns"] for w in fold["train"])) / 1e9
        print(f"{fold['fold_id']}: " + ", ".join(f"{k}={len(fold[k])}" for k in
              ("train", "validation", "excluded_gap", "test")) + f"; actual gap={gap:g}s")


if __name__ == "__main__":
    main()
