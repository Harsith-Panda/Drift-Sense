"""DL foundation contracts, using synthetic data and the Phase 1 handoff."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vio_drift.data.splits import Fold, load_splits, make_dl_splits
from vio_drift.data.normalize import FoldNormalizer, fit_fold_normalizer
from vio_drift.features.build_features import feature_names


def toy():
    windows, raw, features = {}, {}, {}
    for seq, offset in (("A", 0), ("B", 10000)):
        n = 50
        windows[seq] = pd.DataFrame({
            "seq_id": seq, "window_id": np.arange(n) * 3 + 7,
            "t_start_ns": np.arange(n, dtype=np.int64) * 1_000_000_000,
            "t_end_ns": (np.arange(n, dtype=np.int64) + 2) * 1_000_000_000,
        })
        raw[seq] = np.broadcast_to(np.arange(n)[:, None, None] + offset, (n, 400, 6)).copy()
        features[seq] = pd.DataFrame(np.broadcast_to(np.arange(n)[:, None] + offset, (n, 104)),
                                     columns=feature_names())
        features[seq]["seq_id"] = seq
        features[seq]["window_id"] = windows[seq].window_id
    splits = make_dl_splits(windows, [Fold("leave_B", ("A",), ("B",))])
    return windows, raw, features, splits


def test_chronology_gap_and_original_ids():
    windows, _, _, splits = toy()
    fold = splits["folds"][0]
    assert [len(fold[k]) for k in ("train", "validation", "excluded_gap", "test")] == [35, 10, 5, 50]
    assert fold["train"][0]["window_id"] == 7
    assert max(r["t_end_ns"] for r in fold["train"]) + 4_000_000_000 == min(r["t_start_ns"] for r in fold["validation"])
    shuffled = {s: w.sample(frac=1, random_state=5) for s, w in windows.items()}
    assert make_dl_splits(shuffled, [Fold("leave_B", ("A",), ("B",))]) == splits


def test_actual_end_times_not_id_or_stride():
    windows, _, _, _ = toy()
    windows["A"].loc[0, "t_end_ns"] = 45_000_000_000
    fold = make_dl_splits(windows, [Fold("leave_B", ("A",), ("B",))])["folds"][0]
    assert 7 not in [r["window_id"] for r in fold["train"]]
    assert 7 in [r["window_id"] for r in fold["excluded_gap"]]


def test_train_only_moments_and_roundtrip(tmp_path):
    windows, raw, features, splits = toy()
    fold = splits["folds"][0]
    scaler = fit_fold_normalizer(fold, windows, raw, features)
    np.testing.assert_allclose(scaler.imu_mean, raw["A"][:35].mean(axis=(0, 1)))
    np.testing.assert_allclose(scaler.imu_std, raw["A"][:35].std(axis=(0, 1)))
    np.testing.assert_allclose(scaler.feature_mean, features["A"][feature_names()].iloc[:35].mean())
    # Alter validation, purged and test inputs by huge amounts; fit must be identical.
    raw["A"][35:] = 999999
    raw["B"][:] = -999999
    features["A"].loc[35:, feature_names()] = 888888
    features["B"].loc[:, feature_names()] = -888888
    features["A"] = features["A"].sample(frac=1, random_state=3)
    second = fit_fold_normalizer(fold, windows, raw, features)
    for name in ("imu_mean", "imu_std", "feature_mean", "feature_std"):
        np.testing.assert_array_equal(getattr(scaler, name), getattr(second, name))
    path = tmp_path / "stats.json"
    scaler.save(path)
    before = path.read_bytes()
    second.save(path)
    assert before == path.read_bytes()
    loaded = FoldNormalizer.load(path, "leave_B")
    for seq in windows:
        imu = loaded.transform_imu(raw[seq])
        feats = loaded.transform_features(features[seq])
        assert imu.shape == (50, 400, 6) and feats.shape == (50, 104)
        assert imu.dtype == feats.dtype == np.float32
        assert np.isfinite(imu).all() and np.isfinite(feats).all()
    with pytest.raises(ValueError, match="fold mismatch"):
        FoldNormalizer.load(path, "wrong_fold")


def test_constant_channels_and_features():
    windows, raw, features, splits = toy()
    raw["A"][:] = 3
    features["A"].loc[:, feature_names()] = 5
    scaler = fit_fold_normalizer(splits["folds"][0], windows, raw, features)
    assert np.all(scaler.imu_std == 1) and np.all(scaler.feature_std == 1)
    assert np.all(scaler.transform_imu(raw["A"]) == 0)
    assert np.all(scaler.transform_features(features["A"]) == 0)


@pytest.mark.parametrize("bad", ["duplicate", "float_time", "empty_train", "same_flight"])
def test_invalid_splits_rejected(bad):
    windows, _, _, _ = toy()
    outer = [Fold("leave_B", ("A",), ("B",))]
    gap = 4
    if bad == "duplicate":
        windows["A"].loc[1, "window_id"] = windows["A"].loc[0, "window_id"]
    elif bad == "float_time":
        windows["A"]["t_start_ns"] = windows["A"].t_start_ns.astype(float)
    elif bad == "empty_train":
        gap = 1000
    else:
        outer = [Fold("bad", ("A",), ("A",))]
    with pytest.raises(ValueError):
        make_dl_splits(windows, outer, gap_seconds=gap)


def test_normalizer_rejects_leaking_refs_and_missing_keys():
    windows, raw, features, splits = toy()
    fold = splits["folds"][0]
    fold["train"].append(fold["validation"][0])
    with pytest.raises(ValueError, match="overlap"):
        fit_fold_normalizer(fold, windows, raw, features)
    fold["train"].pop()
    features["A"] = features["A"].iloc[1:]
    with pytest.raises(ValueError, match="finite"):
        fit_fold_normalizer(fold, windows, raw, features)


def test_real_data_folds_and_reproducible_artifact():
    p = ROOT / "data/processed"
    sequences = ["MH_01_easy", "MH_02_easy"]
    if not all((p / f"windows_{s}.csv").exists() for s in sequences):
        pytest.skip("Phase 1 windows unavailable")
    windows = {s: pd.read_csv(p / f"windows_{s}.csv") for s in sequences}
    result = make_dl_splits(windows, load_splits(p / "splits.json"))
    assert json.loads((p / "splits_dl.json").read_text()) == result
    assert json.dumps(result) == json.dumps(make_dl_splits(windows, load_splits(p / "splits.json")))
    for fold, expected in zip(result["folds"], [(450, 119, 23, 720), (553, 144, 23, 592)]):
        assert tuple(len(fold[k]) for k in ("train", "validation", "excluded_gap", "test")) == expected
        assert {r["seq_id"] for r in fold["test"]} == {fold["test_source"]}
        assert {r["seq_id"] for r in fold["train"] + fold["validation"]} == {fold["training_source"]}
        assert max(r["t_end_ns"] for r in fold["train"]) + result["gap_ns"] <= min(r["t_start_ns"] for r in fold["validation"])
        source_keys = {tuple(r) for r in windows[fold["training_source"]][["seq_id", "window_id"]].itertuples(index=False, name=None)}
        parts = [{(r["seq_id"], r["window_id"]) for r in fold[k]} for k in ("train", "validation", "excluded_gap")]
        assert set.union(*parts) == source_keys
        assert sum(map(len, parts)) == len(source_keys)
