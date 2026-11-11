"""Shape, toy-loss, seed and parameter-count checks for the IMU models."""
import os
import sys
from pathlib import Path

# Conda NumPy and the pip torch wheel each ship libomp. Set this before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.models.dl.cnn1d import (
    CNN1D,
    CHANNELS,
    KERNELS,
    RegressionHead,
    receptive_field_samples,
    receptive_field_seconds,
)
from vio_drift.models.dl.lstm import LSTMEncoder, LSTMRegressor
from vio_drift.models.dl.train import parameter_count, train_model


class _Tensors(Dataset):
    def __init__(self, imu, target, split):
        self.imu, self.target, self.split = imu, target, split

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.imu[index], self.target[index]


def _toy(n=64, seed=0, split="train"):
    generator = torch.Generator().manual_seed(seed)
    imu = torch.randn(n, 6, 400, generator=generator)
    target = imu[:, 0, :].mean(dim=1)
    return _Tensors(imu, target, split)


def test_output_shapes_and_shared_head():
    imu = torch.randn(8, 6, 400)
    cnn = CNN1D(dropout=0.3, out_dim=1).eval()
    lstm = LSTMRegressor(dropout=0.3, out_dim=1).eval()
    assert cnn.encode(imu).shape == (8, 64)
    assert lstm.encode(imu).shape == (8, 64)
    assert cnn(imu).shape == (8, 1)
    assert lstm(imu).shape == (8, 1)
    assert CNN1D(out_dim=4)(imu).shape == (8, 4)
    assert LSTMRegressor(out_dim=4)(imu).shape == (8, 4)
    assert isinstance(cnn.head, RegressionHead) and isinstance(lstm.head, RegressionHead)
    assert type(cnn.head) is type(lstm.head)
    with pytest.raises(ValueError, match="out_dim"):
        RegressionHead(out_dim=3)
    with pytest.raises(ValueError, match="IMU"):
        cnn.encode(torch.randn(8, 400, 6))


def test_lstm_embedding_is_the_last_hidden_state():
    torch.manual_seed(0)
    encoder = LSTMEncoder(dropout=0.3).eval()
    imu = torch.randn(4, 6, 400)
    embedding = encoder(imu)
    _output, (h_n, _c_n) = encoder.lstm(imu.transpose(1, 2))
    assert torch.allclose(embedding, h_n[-1])
    assert h_n.shape == (1, 4, 64)


def test_receptive_field_and_parameter_counts():
    assert KERNELS == (7, 5, 3) and CHANNELS == (32, 64, 64)
    assert receptive_field_samples() == 13
    assert receptive_field_seconds() == pytest.approx(13 / 200)
    cnn = CNN1D(out_dim=1)
    lstm = LSTMRegressor(out_dim=1)
    # Conv bias + BatchNorm affine for blocks (6→32, k=7), (32→64, k=5), (64→64, k=3).
    conv = (32 * (6 * 7 + 1) + 64) + (64 * (32 * 5 + 1) + 128) + (64 * (64 * 3 + 1) + 128)
    assert parameter_count(cnn.encoder) == conv == 24352
    assert parameter_count(cnn) == conv + 65
    # One-layer LSTM: 4 gates × (input, recurrent, input bias, hidden bias).
    lstm_encoder = 4 * 64 * (6 + 64 + 2)
    assert parameter_count(lstm.encoder) == lstm_encoder == 18432
    assert parameter_count(lstm) == lstm_encoder + 65
    assert parameter_count(CNN1D(out_dim=4)) == conv + 64 * 4 + 4


def test_loss_falls_on_a_toy_problem():
    train_data, val_data = _toy(64, 0, "train"), _toy(32, 1, "validation")
    result = train_model(
        "cnn1d", train_data, val_data, dropout=0.0, epochs=25, batch_size=32,
        learning_rate=1e-2, weight_decay=0.0, seed=0, early_stopping=False,
        use_scheduler=False, num_workers=0,
    )
    assert result.history["train_loss"].iloc[-1] < result.history["train_loss"].iloc[0] * 0.5
    lstm = train_model(
        "lstm", train_data, val_data, dropout=0.0, epochs=15, batch_size=32,
        learning_rate=1e-2, weight_decay=0.0, seed=0, early_stopping=False,
        use_scheduler=False, num_workers=0,
    )
    assert lstm.history["train_loss"].iloc[-1] < lstm.history["train_loss"].iloc[0]


def test_batchnorm_predictions_use_the_training_distribution():
    train_data, val_data = _toy(64, 4, "train"), _toy(32, 5, "validation")
    result = train_model(
        "cnn1d", train_data, val_data, dropout=0.3, epochs=12, batch_size=32,
        learning_rate=1e-2, weight_decay=0.0, seed=1, early_stopping=False, use_scheduler=False,
    )
    result.model.train()
    for module in result.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    with torch.no_grad():
        train_mode = result.model(train_data.imu).reshape(-1)
    result.model.eval()
    with torch.no_grad():
        eval_mode = result.model(train_data.imu).reshape(-1)
    train_mae = (train_mode - train_data.target).abs().mean()
    eval_mae = (eval_mode - train_data.target).abs().mean()
    assert eval_mae < train_mae + 0.05


def test_same_seed_gives_the_same_result():
    train_data, val_data = _toy(48, 2, "train"), _toy(16, 3, "validation")
    kwargs = dict(
        dropout=0.3, epochs=3, batch_size=16, learning_rate=1e-3, weight_decay=1e-4,
        seed=7, early_stopping=False, use_scheduler=True, num_workers=0,
    )
    first = train_model("cnn1d", train_data, val_data, **kwargs)
    second = train_model("cnn1d", train_data, val_data, **kwargs)
    for key in first.model.state_dict():
        assert torch.equal(first.model.state_dict()[key], second.model.state_dict()[key])
    probe = torch.randn(5, 6, 400)
    first.model.eval()
    second.model.eval()
    assert torch.equal(first.model(probe), second.model(probe))


def test_real_batch_matches_the_data_contract():
    processed = ROOT / "data" / "processed"
    if not (processed / "splits_dl.json").exists():
        pytest.skip("Phase 2 splits unavailable")
    from vio_drift.data.dl_dataset import DriftDataset, make_dataloader

    dataset = DriftDataset(processed, "leave_MH_01_easy", "validation")
    batch = next(iter(make_dataloader(dataset, batch_size=4, seed=42)))
    model = CNN1D(dropout=0.3, out_dim=1).eval()
    prediction = model(batch["imu"])
    assert batch["imu"].shape == (4, 6, 400)
    assert prediction.shape == (4, 1)
    assert torch.isfinite(prediction).all()
