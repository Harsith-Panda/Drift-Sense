"""R3 encoder contract, deterministic camera training and real R1 integration."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.models.dl.camera_cnn import CameraCNN, CameraCNNEncoder
from vio_drift.models.dl.camera_train import train_camera, predict_camera
from vio_drift.models.dl.train import parameter_count
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset


class CameraToy(Dataset):
    def __init__(self, split, seed):
        self.split = split
        generator = torch.Generator().manual_seed(seed)
        self.cam = torch.rand(24, 3, 64, 64, generator=generator)
        # Distinct visual texture by class; IMU decoy is never consumed.
        self.cam[::2, :, ::4] = 0
        self.y = torch.tensor([0.4, 0.8] * 12)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"cam": self.cam[i], "y_mag": self.y[i], "imu": torch.full((6, 400), float("nan")),
                "meta": {"seq_id": "toy", "window_id": i * 3 + 2}}


def test_encoder_head_shapes_and_size():
    torch.set_num_threads(1)
    cam = torch.rand(4, 3, 64, 64)
    model = CameraCNN().eval()
    assert model.encode(cam).shape == (4, 64)
    assert model(cam).shape == (4, 1)
    assert CameraCNN(out_dim=4)(cam).shape == (4, 4)
    assert sum(isinstance(m, torch.nn.Conv2d) for m in model.modules()) == 3
    assert parameter_count(model.encoder) == 23808
    assert parameter_count(model) == 23873
    with pytest.raises(ValueError, match="cam must"):
        model(torch.rand(4, 64, 64, 3))
    with pytest.raises(ValueError, match="floating"):
        model(torch.zeros(4, 3, 64, 64, dtype=torch.uint8))
    with pytest.raises(ValueError, match="dropout"):
        CameraCNNEncoder(dropout=1)


def test_loss_decreases_checkpoint_restores_and_seed_repeats(tmp_path):
    train, validation = CameraToy("train", 1), CameraToy("validation", 2)
    kwargs = dict(seed=7, dropout=0, epochs=8, batch_size=12, learning_rate=0.001,
                  early_stopping=False, min_delta=0)
    first = train_camera(train, validation, checkpoint_path=tmp_path / "camera.pt", **kwargs)
    second = train_camera(train, validation, **kwargs)
    assert first.history.train_loss.iloc[-1] < first.history.train_loss.iloc[0] * 0.5
    assert first.best_epoch == int(first.history.loc[first.history.val_loss.idxmin(), "epoch"])
    saved = torch.load(tmp_path / "camera.pt", weights_only=True)
    for key in first.model.state_dict():
        assert torch.equal(first.model.state_dict()[key], second.model.state_dict()[key])
        assert torch.equal(first.model.state_dict()[key], saved["state_dict"][key])
    predictions = predict_camera(first.model, validation)
    assert predictions.columns.tolist() == ["seq_id", "window_id", "y_true", "y_pred"]
    assert predictions.window_id.tolist() == list(range(2, 74, 3))
    assert np.isfinite(predictions.y_pred).all()


def test_test_split_cannot_select_checkpoint():
    with pytest.raises(ValueError, match="test cannot"):
        train_camera(CameraToy("train", 1), CameraToy("test", 2), epochs=1)


def test_real_camera_arrays_match_r1_for_both_folds():
    from vio_drift.data.dl_dataset import DriftDataset, make_dataloader

    processed = ROOT / "data/processed"
    if not (processed / "camera/cam_MH_01_easy.npy").exists():
        pytest.skip("Run camera preprocessing for real-data integration")
    for seq, count in (("MH_01_easy", 720), ("MH_02_easy", 592)):
        array = np.load(processed / "camera" / f"cam_{seq}.npy", allow_pickle=False)
        windows = pd.read_csv(processed / f"windows_{seq}.csv")
        camera = pd.read_csv(processed / "camera" / f"camera_feats_{seq}.csv")
        assert array.shape == (count, 3, 64, 64) and array.dtype == np.uint8
        pd.testing.assert_frame_equal(camera[["seq_id", "window_id"]], windows[["seq_id", "window_id"]])
        for split in ("train", "validation", "test"):
            dataset = DriftDataset(processed, f"leave_{seq}", split, use_camera=True)
            batch = next(iter(make_dataloader(dataset, batch_size=4)))
            assert batch["cam"].shape == (4, 3, 64, 64)
            assert batch["cam"].dtype == torch.float32
            assert 0 <= batch["cam"].min() <= batch["cam"].max() <= 1
            assert torch.isfinite(CameraCNN().eval()(batch["cam"])).all()
        assert "cam" not in DriftDataset(processed, f"leave_{seq}", "test", use_camera=False)[0]
