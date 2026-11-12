"""Trainer, scaling and camera-flow input for the R4 hybrid models.

Follows R2's optimisation policy (AdamW, ReduceLROnPlateau, early stopping on
validation loss, best-checkpoint restore, batch-norm refresh) but feeds the
whole Section 6 dictionary to the model instead of the IMU tensor alone.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vio_drift.data.dl_dataset import make_dataloader
from vio_drift.models.dl.hybrid import HybridModel, fit_physics_prior
from vio_drift.models.dl.train import TrainResult, set_seed

FLOW_COLUMNS = ["flow_mag_mean", "flow_mag_std", "flow_mag_p90_mean", "flow_dx_mean", "flow_dy_mean",
                "flow_dx_std", "flow_dy_std", "blur_mean", "blur_std", "corners_mean"]
TENSOR_KEYS = ("imu", "feats", "phys", "cam", "cam_feats", "y_mag", "y_vec")


class CameraFlowDataset(Dataset):
    """Adds R3's per-window optical-flow numbers as ``cam_feats``.

    Standardised with statistics of the training split only: pass the
    training wrapper's ``stats`` when wrapping validation or test data.
    """

    def __init__(self, base, processed_dir, stats=None, columns=FLOW_COLUMNS):
        if stats is None and base.split != "train":
            raise ValueError("flow statistics must come from the training split; pass stats=")
        self.base, self.columns = base, list(columns)
        keys = pd.DataFrame([{"seq_id": m["seq_id"], "window_id": m["window_id"]} for m in base.meta])
        tables = []
        for seq in keys.seq_id.unique():
            path = Path(processed_dir) / "camera" / f"camera_feats_{seq}.csv"
            if not path.exists():
                raise FileNotFoundError(f"R3 camera features required: {path}")
            tables.append(pd.read_csv(path, usecols=["seq_id", "window_id", *self.columns]))
        merged = keys.merge(pd.concat(tables), on=["seq_id", "window_id"], how="left",
                            validate="one_to_one", sort=False)
        values = merged[self.columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("missing or nonfinite camera flow numbers")
        if stats is None:
            std = values.std(axis=0)
            stats = (values.mean(axis=0), np.where(std <= 1e-12, 1.0, std))
        self.stats = stats
        self.values = ((values - stats[0]) / stats[1]).astype(np.float32)

    @property
    def split(self):
        return self.base.split

    @property
    def meta(self):
        return self.base.meta

    def set_epoch(self, epoch):
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        item["cam_feats"] = torch.from_numpy(self.values[index].copy())
        return item


def _ordered_loader(dataset, batch_size=64):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if k in TENSOR_KEYS else v) for k, v in batch.items()}


def _target(batch: dict, out_dim: int) -> torch.Tensor:
    y = batch["y_mag"].reshape(-1, 1)
    return torch.cat([y, batch["y_vec"]], dim=1) if out_dim == 4 else y


def fit_scaling(model: HybridModel, train_data) -> HybridModel:
    """Physics prior, physics standardisation and target scale from training rows only."""
    if train_data.split != "train":
        raise ValueError("scaling must be fitted on the train split")
    phys, y_mag, y_vec = [], [], []
    for batch in _ordered_loader(train_data):
        phys.append(batch["phys"].numpy())
        y_mag.append(batch["y_mag"].numpy().reshape(-1))
        y_vec.append(batch["y_vec"].numpy())
    phys, y_mag, y_vec = np.concatenate(phys), np.concatenate(y_mag), np.concatenate(y_vec)
    prior = fit_physics_prior(phys, y_mag, y_vec)
    targets = np.column_stack([y_mag, y_vec])[:, : model.out_dim]
    std = phys.std(axis=0)
    with torch.no_grad():
        model.phys_mean.copy_(torch.as_tensor(phys.mean(axis=0), dtype=torch.float32))
        model.phys_std.copy_(torch.as_tensor(np.where(std <= 1e-12, 1.0, std), dtype=torch.float32))
        model.set_prior(prior)
        spread = targets - prior.predict(phys)[:, : model.out_dim] if model.residual else targets
        scale = spread.std(axis=0)
        model.y_center.copy_(torch.as_tensor(targets.mean(axis=0), dtype=torch.float32))
        model.y_scale.copy_(torch.as_tensor(np.where(scale <= 1e-12, 1.0, scale), dtype=torch.float32))
    return model


def _refresh_batchnorm(model, train_data, batch_size, seed, device) -> None:
    """Recompute BatchNorm statistics with dropout off (same reason as R2's trainer)."""
    layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    if not layers:
        return
    momenta = [layer.momentum for layer in layers]
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.eval()
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
    with torch.no_grad():
        for batch in make_dataloader(train_data, batch_size, seed, 0):
            model(_to_device(batch, device))
    for layer, momentum in zip(layers, momenta):
        layer.momentum = momentum


def _epoch(model, loader, device, optimizer=None, grad_clip=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_abs = 0.0
    count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = _to_device(batch, device)
            target = _target(batch, model.out_dim)
            prediction = model(batch)
            loss = model.loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite hybrid loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            n = len(target)
            total_loss += loss.item() * n
            total_abs += (prediction[:, 0] - target[:, 0]).abs().sum().item()
            count += n
    if count == 0:
        raise ValueError("loader produced no samples")
    return total_loss / count, total_abs / count


def train_hybrid(model: HybridModel, train_data, val_data, *, epochs=100, batch_size=32,
                 learning_rate=1e-3, weight_decay=1e-4, seed=42, device="cpu", num_workers=0,
                 early_stopping=True, patience=15, min_delta=1e-5, scheduler_patience=5,
                 scheduler_factor=0.5, grad_clip=1.0, min_epochs=0, checkpoint_path=None,
                 curve_path=None, log=False) -> TrainResult:
    """Seeded fit; restores the best validation epoch. Test data cannot select weights.

    ``min_epochs`` is a warm-up: no checkpoint from an earlier epoch can be
    selected and early stopping cannot fire before it. ``val_mae`` in the
    history is the magnitude MAE in metres.
    """
    if not 0 <= min_epochs <= epochs:
        raise ValueError("min_epochs must be between 0 and epochs")
    if train_data.split != "train":
        raise ValueError("train_data.split must be train")
    if val_data.split != "validation":
        raise ValueError("val_data must be the validation split; the test flight cannot select weights")
    set_seed(seed)
    model.reset_parameters()
    fit_scaling(model, train_data)
    torch_device = torch.device(device)
    model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience,
        threshold=min_delta, threshold_mode="abs")
    # Epoch 0 = the untrained model (for H2: exactly the physics prior).
    _refresh_batchnorm(model, train_data, batch_size, seed, torch_device)
    init_loss, init_mae = _epoch(model, make_dataloader(val_data, batch_size, seed, num_workers), torch_device)
    rows = [{"epoch": 0, "train_loss": np.nan, "val_loss": init_loss, "train_mae": np.nan,
             "val_mae": init_mae, "learning_rate": learning_rate}]
    # Epoch 0 may be kept only by residual models, where it means "physics alone";
    # for a direct model it is an untrained random network.
    if model.residual and min_epochs == 0:
        best_val, best_mae, best_epoch = init_loss, init_mae, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        best_val, best_mae, best_epoch, best_state = float("inf"), float("inf"), 0, None
    stale = 0
    for epoch in range(1, epochs + 1):
        if hasattr(train_data, "set_epoch"):
            train_data.set_epoch(epoch - 1)
        train_loss, train_mae = _epoch(model, make_dataloader(train_data, batch_size, seed, num_workers),
                                       torch_device, optimizer, grad_clip)
        _refresh_batchnorm(model, train_data, batch_size, seed, torch_device)
        val_loss, val_mae = _epoch(model, make_dataloader(val_data, batch_size, seed, num_workers), torch_device)
        lr_now = optimizer.param_groups[0]["lr"]
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     "train_mae": train_mae, "val_mae": val_mae, "learning_rate": lr_now})
        if log:
            print(f"  epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} "
                  f"val_mae={val_mae * 100:.2f}cm lr={lr_now:.2g}", flush=True)
        scheduler.step(val_loss)
        if epoch < min_epochs:
            continue  # warm-up: not a candidate, no early stopping yet
        if val_loss < best_val - min_delta:
            best_val, best_mae, best_epoch, stale = val_loss, val_mae, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if early_stopping and stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    history = pd.DataFrame(rows)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_name": "hybrid", "config": model.config, "seed": seed,
                    "best_epoch": best_epoch, "best_val_loss": best_val, "state_dict": best_state,
                    "optimizer": "AdamW", "scheduler": "ReduceLROnPlateau"}, checkpoint_path)
    if curve_path is not None:
        Path(curve_path).parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(curve_path, index=False)
    return TrainResult(model, history, best_epoch, best_val, best_mae, checkpoint_path)


def predict_hybrid(model: HybridModel, dataset, *, batch_size=64, device="cpu") -> pd.DataFrame:
    """Shared prediction contract (seq_id, window_id, y_true, y_pred) in dataset order.

    Vector models add y_true_d* / y_pred_d* columns for drift correction (C1).
    """
    model.to(device).eval()
    rows = []
    with torch.no_grad():
        for batch in _ordered_loader(dataset, batch_size):
            batch = _to_device(batch, device)
            pred = model(batch).cpu().numpy()
            true = _target(batch, model.out_dim).cpu().numpy()
            if not np.isfinite(pred).all():
                raise ValueError("nonfinite hybrid predictions")
            for i, (seq, wid) in enumerate(zip(batch["meta"]["seq_id"], batch["meta"]["window_id"])):
                row = {"seq_id": seq, "window_id": int(wid), "y_true": float(true[i, 0]),
                       "y_pred": float(pred[i, 0])}
                if model.out_dim == 4:
                    for j, axis in enumerate("xyz", start=1):
                        row[f"y_true_d{axis}"] = float(true[i, j])
                    for j, axis in enumerate("xyz", start=1):
                        row[f"y_pred_d{axis}"] = float(pred[i, j])
                rows.append(row)
    frame = pd.DataFrame(rows)
    if model.out_dim == 4:
        frame = frame[["seq_id", "window_id", "y_true", "y_pred", "y_true_dx", "y_true_dy", "y_true_dz",
                       "y_pred_dx", "y_pred_dy", "y_pred_dz"]]
    if len(frame) != len(dataset):
        raise RuntimeError("prediction rows do not match the dataset")
    return frame


__all__ = ["CameraFlowDataset", "FLOW_COLUMNS", "fit_scaling", "predict_hybrid", "train_hybrid"]
