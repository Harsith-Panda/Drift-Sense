"""R4 hybrid models: physics prior, branch flags, missing camera, trainer."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.models.dl.hybrid import (
    HybridModel,
    PhysicsPrior,
    build_hybrid,
    fit_physics_prior,
)
from vio_drift.models.dl.hybrid_train import (
    CameraFlowDataset,
    fit_scaling,
    predict_hybrid,
    train_hybrid,
)

B = 5
N_FLOW = 10


def _batch(n=B, seed=0, with_cam=True, with_flow=False):
    g = torch.Generator().manual_seed(seed)
    phys = torch.rand(n, 4, generator=g) + 1.0
    batch = {
        "imu": torch.randn(n, 6, 400, generator=g),
        "feats": torch.randn(n, 104, generator=g),
        "phys": phys,
        "y_mag": phys[:, 3] + 0.01,
        "y_vec": phys[:, :3] + 0.01,
    }
    if with_cam:
        batch["cam"] = torch.rand(n, 3, 64, 64, generator=g)
    if with_flow:
        batch["cam_feats"] = torch.randn(n, N_FLOW, generator=g)
    return batch


# ---------- physics prior ----------

def test_fit_physics_prior_recovers_linear_map():
    rng = np.random.default_rng(0)
    phys = rng.uniform(1, 2, size=(200, 4))
    y_mag = 1.05 * phys[:, 3] - 0.04
    y_vec = 0.9 * phys[:, :3] + 0.02
    prior = fit_physics_prior(phys, y_mag, y_vec)
    assert prior.a[0] == pytest.approx(1.05, abs=1e-6)
    assert prior.b[0] == pytest.approx(-0.04, abs=1e-6)
    np.testing.assert_allclose(prior.a[1:], 0.9, atol=1e-6)
    np.testing.assert_allclose(prior.b[1:], 0.02, atol=1e-6)


def test_physics_prior_predict_uses_mag_then_vector_columns():
    prior = PhysicsPrior(a=np.array([2.0, 1.0, 1.0, 1.0]), b=np.array([0.5, 0.0, 0.0, 0.0]))
    phys = np.array([[0.1, 0.2, 0.3, 1.0]])
    np.testing.assert_allclose(prior.predict(phys), [[2.5, 0.1, 0.2, 0.3]])


# ---------- residual hybrid (H2) ----------

@pytest.mark.parametrize("out_dim", [1, 4])
def test_residual_model_starts_exactly_at_physics_prediction(out_dim):
    model = build_hybrid("H2", out_dim=out_dim)
    prior = PhysicsPrior(a=np.array([1.1, 0.9, 0.8, 1.2]), b=np.array([-0.03, 0.01, 0.0, -0.01]))
    model.set_prior(prior)
    model.eval()
    batch = _batch(with_cam=False)
    with torch.no_grad():
        out = model(batch)
    expected = prior.predict(batch["phys"].numpy())[:, :out_dim]
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)


def test_residual_model_without_prior_is_rejected():
    model = build_hybrid("H2")
    with pytest.raises(RuntimeError, match="prior"):
        model(_batch(with_cam=False))


# ---------- branch flags and shapes ----------

@pytest.mark.parametrize("kind", ["H1", "H2", "H3"])
@pytest.mark.parametrize("out_dim", [1, 4])
def test_presets_produce_batch_by_out_dim(kind, out_dim):
    model = build_hybrid(kind, out_dim=out_dim)
    model.set_prior(PhysicsPrior(a=np.ones(4), b=np.zeros(4)))
    model.eval()
    with torch.no_grad():
        out = model(_batch())
    assert out.shape == (B, out_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("imu,feats,cam", [
    ("lstm", False, False), ("cnn1d", False, False), (None, True, False),
    (None, False, True), ("lstm", True, True),
])
def test_each_branch_combination_gives_right_shape(imu, feats, cam):
    model = HybridModel(imu_encoder=imu, use_features=feats, use_camera=cam)
    model.eval()
    with torch.no_grad():
        assert model(_batch()).shape == (B, 1)
    assert model.n_branches == int(imu is not None) + int(feats) + int(cam)


def test_no_branch_is_rejected():
    with pytest.raises(ValueError, match="branch"):
        HybridModel(imu_encoder=None, use_features=False, use_camera=False)


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError):
        build_hybrid("H9")


def test_disabled_branches_have_no_parameters():
    imu_only = HybridModel(imu_encoder="lstm", use_features=False, use_camera=False)
    assert imu_only.camera_encoder is None and imu_only.feature_encoder is None
    names = [n for n, _ in imu_only.named_parameters()]
    assert not any(n.startswith(("camera_encoder", "feature_encoder")) for n in names)


def test_fusion_handles_missing_camera_with_zero_embedding():
    model = HybridModel(imu_encoder="lstm", use_features=False, use_camera=True)
    model.eval()
    with torch.no_grad():
        out = model(_batch(with_cam=False))
    assert out.shape == (B, 1) and torch.isfinite(out).all()


def test_camera_input_changes_prediction_when_branch_on():
    model = HybridModel(imu_encoder=None, use_features=False, use_camera=True)
    torch.nn.init.normal_(model.head.linear.weight)
    model.eval()
    batch = _batch()
    other = dict(batch, cam=torch.zeros_like(batch["cam"]))
    with torch.no_grad():
        assert not torch.allclose(model(batch), model(other))


def test_camera_flow_numbers_feed_feature_branch():
    model = HybridModel(imu_encoder=None, use_features=True, use_camera=False, n_cam_feats=N_FLOW)
    torch.nn.init.normal_(model.head.linear.weight)
    model.eval()
    batch = _batch(with_flow=True)
    other = dict(batch, cam_feats=batch["cam_feats"] + 5.0)
    with torch.no_grad():
        assert not torch.allclose(model(batch), model(other))
    with pytest.raises(KeyError):
        model(_batch(with_flow=False))


# ---------- scaling ----------

class _Toy(Dataset):
    """Synthetic windows: target = physics + a feature-driven residual."""

    def __init__(self, n, seed, split):
        g = torch.Generator().manual_seed(seed)
        self.split = split
        self.phys = torch.rand(n, 4, generator=g) + 1.0
        self.feats = torch.randn(n, 104, generator=g)
        self.imu = torch.randn(n, 6, 400, generator=g) * 0.1
        self.y = self.phys[:, 3] + 0.05 * self.feats[:, 0]
        self.vec = self.phys[:, :3] + 0.05 * self.feats[:, 1:4]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"imu": self.imu[i], "feats": self.feats[i], "phys": self.phys[i],
                "y_mag": self.y[i], "y_vec": self.vec[i],
                "meta": {"seq_id": "toy", "window_id": i, "t_start_ns": i}}


def test_fit_scaling_sets_prior_and_target_scale_from_training_data():
    train = _Toy(64, 0, "train")
    model = build_hybrid("H2", imu_encoder=None)
    fit_scaling(model, train)
    phys = train.phys.numpy()
    a, b = np.polyfit(phys[:, 3], train.y.numpy(), 1)
    assert model.prior_a[0].item() == pytest.approx(a, rel=1e-4)
    assert model.prior_b[0].item() == pytest.approx(b, abs=1e-4)
    np.testing.assert_allclose(model.phys_mean.numpy(), phys.mean(0), rtol=1e-5)
    assert model.y_scale[0].item() > 0


def test_fit_scaling_rejects_non_training_split():
    with pytest.raises(ValueError, match="train"):
        fit_scaling(build_hybrid("H1"), _Toy(16, 0, "validation"))


def test_direct_model_scaling_centres_on_training_mean():
    train = _Toy(64, 0, "train")
    model = HybridModel(imu_encoder=None, use_features=True, zero_init_head=True)
    fit_scaling(model, train)
    model.eval()
    with torch.no_grad():
        out = model(_batch(with_cam=False))
    np.testing.assert_allclose(out.numpy().ravel(), train.y.mean().item(), atol=1e-5)


# ---------- trainer ----------

def _fit(kind, seed, **kw):
    return train_hybrid(build_hybrid(kind, imu_encoder=None, **kw), _Toy(320, 0, "train"),
                        _Toy(32, 1, "validation"), epochs=25, batch_size=16,
                        learning_rate=3e-3, seed=seed, patience=25)


def test_trainer_beats_the_physics_start_on_toy_residual():
    result = _fit("H2", 0)
    first = result.history.val_loss.iloc[0]
    assert result.best_val_loss < 0.5 * first


def _frozen(kind):
    """learning_rate=0: no epoch can beat the starting point on validation."""
    return train_hybrid(build_hybrid(kind, imu_encoder=None), _Toy(64, 0, "train"),
                        _Toy(32, 1, "validation"), epochs=3, learning_rate=0.0, patience=5)


def test_residual_model_may_keep_the_physics_start():
    assert _frozen("H2").best_epoch == 0


def test_direct_model_never_returns_the_untrained_network():
    # Epoch 0 of a direct model is a random network, not a meaningful baseline.
    assert _frozen("H1").best_epoch >= 1


@pytest.mark.parametrize("kind", ["H1", "H2"])
def test_min_epochs_blocks_earlier_checkpoints(kind):
    result = train_hybrid(build_hybrid(kind, imu_encoder=None), _Toy(64, 0, "train"),
                          _Toy(32, 1, "validation"), epochs=6, learning_rate=0.0,
                          patience=1, min_epochs=4)
    assert result.best_epoch >= 4
    assert result.history.epoch.max() >= 4  # early stopping cannot fire before the minimum


def test_min_epochs_beyond_epochs_is_rejected():
    with pytest.raises(ValueError, match="min_epochs"):
        train_hybrid(build_hybrid("H1", imu_encoder=None), _Toy(16, 0, "train"),
                     _Toy(16, 1, "validation"), epochs=3, min_epochs=5)


def test_trainer_is_reproducible_for_same_seed():
    a, b = _fit("H2", 7), _fit("H2", 7)
    pd.testing.assert_frame_equal(a.history, b.history)


def test_trainer_rejects_test_split_as_validation():
    with pytest.raises(ValueError, match="validation"):
        train_hybrid(build_hybrid("H1", imu_encoder=None), _Toy(16, 0, "train"),
                     _Toy(16, 1, "test"), epochs=1)


def test_vector_head_trains_and_predicts_four_columns():
    result = _fit("H2", 0, out_dim=4)
    frame = predict_hybrid(result.model, _Toy(10, 2, "test"))
    assert list(frame.columns) == ["seq_id", "window_id", "y_true", "y_pred",
                                   "y_true_dx", "y_true_dy", "y_true_dz",
                                   "y_pred_dx", "y_pred_dy", "y_pred_dz"]
    assert len(frame) == 10 and np.isfinite(frame.select_dtypes("number")).all().all()


def test_predictions_follow_contract_and_dataset_order():
    result = _fit("H1", 0)
    data = _Toy(10, 2, "test")
    frame = predict_hybrid(result.model, data)
    assert list(frame.columns) == ["seq_id", "window_id", "y_true", "y_pred"]
    assert frame.window_id.tolist() == list(range(10))
    np.testing.assert_allclose(frame.y_true, data.y.numpy(), rtol=1e-6)


# ---------- camera flow wrapper (real R3 artifacts) ----------

PROCESSED = ROOT / "data" / "processed"


@pytest.mark.skipif(not (PROCESSED / "camera" / "camera_feats_MH_01_easy.csv").exists(),
                    reason="R3 camera features not generated")
def test_camera_flow_wrapper_aligns_rows_and_uses_training_statistics():
    from vio_drift.data.dl_dataset import DriftDataset
    train = CameraFlowDataset(DriftDataset(PROCESSED, "leave_MH_01_easy", "train"), PROCESSED)
    test = CameraFlowDataset(DriftDataset(PROCESSED, "leave_MH_01_easy", "test"), PROCESSED,
                             stats=train.stats)
    assert train.split == "train" and len(test) == 720
    item = test[17]
    raw = pd.read_csv(PROCESSED / "camera" / "camera_feats_MH_01_easy.csv")
    row = raw.loc[raw.window_id == item["meta"]["window_id"], train.columns].to_numpy()[0]
    expected = (row - train.stats[0]) / train.stats[1]
    np.testing.assert_allclose(item["cam_feats"].numpy(), expected, rtol=1e-5)
    assert np.abs(np.stack([train[i]["cam_feats"].numpy() for i in range(len(train))]).mean(0)).max() < 1e-4
    with pytest.raises(ValueError, match="training"):
        CameraFlowDataset(DriftDataset(PROCESSED, "leave_MH_01_easy", "test"), PROCESSED)
