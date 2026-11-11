"""Real-data leakage, Dataset, augmentation and A2 integration checks."""
import copy
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vio_drift.data.dl_dataset import DriftDataset, make_dataloader
from vio_drift.data.augment import IMUAugmentation
from vio_drift.data.dl_physics import fit_training_bias, training_interval_mask
from vio_drift.data.normalize import fit_fold_normalizer

P = ROOT / "data/processed"
FOLDS = ["leave_MH_01_easy", "leave_MH_02_easy"]

@pytest.mark.parametrize("fold_id", FOLDS)
def test_all_partitions_contract_and_replay(fold_id):
    for part in ("train", "validation", "test"):
        ds = DriftDataset(P, fold_id, part)
        a = next(iter(make_dataloader(ds)))
        b = next(iter(make_dataloader(ds)))
        expected = {"imu": (32,6,400), "feats": (32,104), "phys": (32,4),
                    "y_mag": (32,), "y_vec": (32,3)}
        assert set(a) == {*expected, "meta"}
        for key, shape in expected.items():
            assert tuple(a[key].shape) == shape
            assert a[key].dtype == torch.float32 and a[key].device.type == "cpu"
            assert torch.isfinite(a[key]).all()
            assert torch.equal(a[key], b[key])
        original = ds[0]["imu"].clone()
        ds[0]["imu"].zero_()
        assert torch.equal(ds[0]["imu"], original)
        if part != "train":
            assert a["meta"]["window_id"].tolist() == [m["window_id"] for m in ds.meta[:32]]

@pytest.mark.parametrize("fold_id", FOLDS)
def test_training_only_statistics_and_bias(fold_id):
    fold = next(f for f in json.loads((P/"splits_dl.json").read_text())["folds"] if f["fold_id"] == fold_id)
    seqs = [fold["training_source"], fold["test_source"]]
    windows = {s: pd.read_csv(P/f"windows_{s}.csv") for s in seqs}
    raw = {s: np.load(P/f"windows_raw_{s}.npy") for s in seqs}
    feats = {s: pd.read_csv(P/f"features_{s}.csv") for s in seqs}
    first = fit_fold_normalizer(fold, windows, raw, feats)
    source = fold["training_source"]
    mask = windows[source].window_id.isin([w["window_id"] for w in fold["train"]]).to_numpy()
    np.testing.assert_allclose(first.imu_mean, raw[source][mask].mean(axis=(0,1)))
    np.testing.assert_allclose(first.feature_mean, feats[source].loc[mask, first.feature_order].mean())
    raw[source][~mask] = 1e12
    raw[fold["test_source"]][:] = -1e12
    feats[source].loc[~mask, first.feature_order] = 1e12
    feats[fold["test_source"]].loc[:, first.feature_order] = -1e12
    second = fit_fold_normalizer(fold, windows, raw, feats)
    for name in ("imu_mean", "imu_std", "feature_mean", "feature_std"):
        np.testing.assert_array_equal(getattr(first,name),getattr(second,name))
    aligned = {s: pd.read_csv(ROOT/f"data/interim/aligned_{s}.csv") for s in seqs}
    bias = fit_training_bias(fold, aligned)
    from vio_drift.features.physics import BIAS_COLS
    mask = training_interval_mask(aligned[source], fold["train"])
    aligned[source].loc[~mask, BIAS_COLS] = 1e8
    aligned[fold["test_source"]].loc[:, BIAS_COLS] = -1e8
    np.testing.assert_array_equal(bias, fit_training_bias(fold, aligned))

@pytest.mark.parametrize("part", ["validation", "test"])
def test_eval_rejects_augmentation(part):
    with pytest.raises(ValueError, match="validation/test"):
        DriftDataset(P,FOLDS[0],part,augmentation=IMUAugmentation(gaussian_noise=True))
    with pytest.raises(ValueError, match="only on train"):
        IMUAugmentation().apply(np.zeros((400,6)),np.ones(6),np.random.default_rng(0),split=part)


def test_augmentation_bounds_labels_and_epochs():
    aug = IMUAugmentation(gaussian_noise=True,random_crop=True)
    ds = DriftDataset(P,FOLDS[0],"train",augmentation=aug)
    base = DriftDataset(P,FOLDS[0],"train")
    first = ds[0]
    assert torch.equal(first["imu"],ds[0]["imu"])
    assert not torch.equal(first["imu"],base[0]["imu"])
    for key in ("feats","phys","y_mag","y_vec"):
        assert torch.equal(first[key],base[0][key])
    assert first["meta"] == base[0]["meta"]
    ds.set_epoch(1)
    assert not torch.equal(first["imu"],ds[0]["imu"])
    x=np.ones((400,6),dtype=np.float32)
    out=IMUAugmentation(random_crop=True).apply(x,np.ones(6),np.random.default_rng(0),split="train")
    assert np.count_nonzero((out==0).all(axis=1)) <= 4
    assert np.all(x==1)
    out=IMUAugmentation(gaussian_noise=True).apply(np.zeros_like(x),np.ones(6),np.random.default_rng(0),split="train")
    bound=.3*np.array([1.6968e-4]*3+[2e-3]*3)*np.sqrt(200)
    assert (np.abs(out)<=bound+1e-8).all()


def test_camera_off_missing_and_synthetic_on(tmp_path):
    assert "cam" not in DriftDataset(P,FOLDS[0],"test")[0]
    with pytest.raises(FileNotFoundError,match="R3"):
        DriftDataset(P,FOLDS[0],"test",use_camera=True)
    # Isolated synthetic camera arrays validate the interface, not R3's image matching.
    processed=tmp_path/"data/processed"
    shutil.copytree(P,processed)
    shutil.copytree(ROOT/"data/interim",tmp_path/"data/interim")
    camera=processed/"camera";camera.mkdir(exist_ok=True)
    seq="MH_01_easy"
    w=pd.read_csv(processed/f"windows_{seq}.csv")
    np.save(camera/f"cam_{seq}.npy",np.full((len(w),3,64,64),255,dtype=np.uint8))
    w[["seq_id","window_id"]].to_csv(camera/f"camera_feats_{seq}.csv",index=False)
    ds=DriftDataset(processed,FOLDS[0],"test",use_camera=True)
    assert ds[0]["cam"].shape==(3,64,64) and torch.all(ds[0]["cam"]==1)
    assert next(iter(make_dataloader(ds)))["cam"].shape==(32,3,64,64)
    w[["seq_id","window_id"]].iloc[::-1].to_csv(camera/f"camera_feats_{seq}.csv",index=False)
    with pytest.raises(ValueError,match="Camera key"):
        DriftDataset(processed,FOLDS[0],"test",use_camera=True)


def test_a2_nested_fixed_evaluation_and_refit():
    base=json.loads((P/"splits_dl.json").read_text())["folds"]
    for fold in base:
        previous=set()
        for fraction in (50,75,100):
            artifacts=P if fraction==100 else P/"ablations_dl"/f"train_{fraction:03d}"
            current=next(f for f in json.loads((artifacts/"splits_dl.json").read_text())["folds"] if f["fold_id"]==fold["fold_id"])
            assert current["validation"]==fold["validation"] and current["test"]==fold["test"]
            ids={r["window_id"] for r in current["train"]}
            assert previous<=ids
            previous=ids
            ds=DriftDataset(P,fold["fold_id"],"train",artifact_dir=artifacts)
            assert len(ds)==len(ids)
    jobs=json.loads((ROOT/"results/ablations/r1_data_manifest.json").read_text())["jobs"]
    assert len(jobs)==84 and len({j["run_id"] for j in jobs})==84
