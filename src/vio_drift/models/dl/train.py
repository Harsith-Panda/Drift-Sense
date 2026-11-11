"""One trainer for the IMU regressors.

AdamW applies the configured weight decay as decoupled decay. Early
stopping and the learning-rate schedule both watch validation MSE.
The best validation checkpoint is restored and written to ``models/dl/``.
Loss curves go to ``results/metrics/``.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn

from vio_drift.data.dl_dataset import make_dataloader
from vio_drift.models.dl.cnn1d import CNN1D
from vio_drift.models.dl.lstm import LSTMRegressor

EXPERIMENT_MODELS = {"E1": "cnn1d", "E2": "lstm"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_model(model_name: str, dropout: float = 0.3, out_dim: int = 1) -> nn.Module:
    if model_name == "cnn1d":
        return CNN1D(dropout=dropout, out_dim=out_dim)
    if model_name == "lstm":
        return LSTMRegressor(dropout=dropout, out_dim=out_dim)
    raise ValueError("model_name must be cnn1d or lstm")


@dataclass
class TrainResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_val_loss: float
    best_val_mae: float
    checkpoint_path: Path | None


def _unpack(batch, device: torch.device):
    if isinstance(batch, dict):
        imu = batch["imu"].to(device)
        target = batch["y_mag"].to(device).reshape(-1)
        return imu, target, batch.get("meta")
    imu, target = batch[0], batch[1]
    return imu.to(device), target.to(device).reshape(-1), None


def _refresh_batchnorm(model, loader, device) -> None:
    """Recompute batch-norm statistics on the training split with dropout off.

    Each block is Conv → BatchNorm → ReLU → Dropout, so the next block's
    normalisation sees dropped activations during training and full
    activations at prediction. The running statistics are therefore refreshed
    from the training windows in the prediction regime before validation.
    """
    batchnorm_layers = [module for module in model.modules() if isinstance(module, nn.BatchNorm1d)]
    if not batchnorm_layers:
        return
    momenta = [layer.momentum for layer in batchnorm_layers]
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    for layer in batchnorm_layers:
        layer.reset_running_stats()
        layer.momentum = None
    with torch.no_grad():
        for batch in loader:
            imu, _target, _meta = _unpack(batch, device)
            model(imu)
    for layer, momentum in zip(batchnorm_layers, momenta):
        layer.momentum = momentum


def _run_epoch(model, loader, loss_fn, optimizer, device) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_abs = 0.0
    count = 0
    for batch in loader:
        imu, target, _meta = _unpack(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(imu).reshape(-1)
        loss = loss_fn(prediction, target)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = target.shape[0]
        total_loss += loss.item() * batch_size
        total_abs += (prediction.detach() - target).abs().sum().item()
        count += batch_size
    if count == 0:
        raise ValueError("loader produced no samples")
    return total_loss / count, total_abs / count


def _clone_state(model: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_model(
    model_name: str,
    train_data,
    val_data,
    *,
    dropout: float = 0.3,
    out_dim: int = 1,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    num_workers: int = 0,
    early_stopping: bool = True,
    patience: int = 15,
    min_delta: float = 1e-5,
    use_scheduler: bool = True,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    checkpoint_path: Path | None = None,
    curve_path: Path | None = None,
    plot_path: Path | None = None,
    log: bool = False,
) -> TrainResult:
    """Seed, fit, restore the best validation epoch, and write the requested files."""
    if train_data.split != "train":
        raise ValueError("train_data.split must be train")
    if val_data.split == "train":
        raise ValueError("validation data must not be the training split")
    set_seed(seed)
    torch_device = torch.device(device)
    model = build_model(model_name, dropout=dropout, out_dim=out_dim).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience,
            threshold=min_delta, threshold_mode="abs",
        )
    loss_fn = nn.MSELoss()
    rows = []
    best_val = float("inf")
    best_epoch = 0
    best_mae = float("inf")
    best_state = None
    stale = 0
    for epoch_index in range(epochs):
        if hasattr(train_data, "set_epoch"):
            train_data.set_epoch(epoch_index)
        train_loader = make_dataloader(train_data, batch_size, seed, num_workers)
        val_loader = make_dataloader(val_data, batch_size, seed, num_workers)
        train_loss, train_mae = _run_epoch(model, train_loader, loss_fn, optimizer, torch_device)
        _refresh_batchnorm(model, make_dataloader(train_data, batch_size, seed, num_workers), torch_device)
        with torch.no_grad():
            val_loss, val_mae = _run_epoch(model, val_loader, loss_fn, None, torch_device)
        learning_rate_now = optimizer.param_groups[0]["lr"]
        rows.append({
            "epoch": epoch_index + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "learning_rate": learning_rate_now,
        })
        if log:
            print(
                f"  epoch {epoch_index + 1}: train_mse={train_loss:.6f} val_mse={val_loss:.6f} "
                f"val_mae={val_mae:.4f} lr={learning_rate_now:.3g}",
                flush=True,
            )
        if scheduler is not None:
            scheduler.step(val_loss)
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_mae = val_mae
            best_epoch = epoch_index + 1
            best_state = _clone_state(model)
            stale = 0
        else:
            stale += 1
        if early_stopping and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    history = pd.DataFrame(rows)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_name": model_name,
            "dropout": dropout,
            "out_dim": out_dim,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "state_dict": best_state,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau" if use_scheduler else "none",
        }, checkpoint_path)
    if curve_path is not None:
        curve_path = Path(curve_path)
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(curve_path, index=False)
    if plot_path is not None:
        _save_curve(history, Path(plot_path), f"{model_name} validation MSE")
    return TrainResult(model, history, best_epoch, best_val, best_mae, checkpoint_path)


def _save_curve(history: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    axis.plot(history["epoch"], history["train_loss"], label="train MSE")
    axis.plot(history["epoch"], history["val_loss"], label="validation MSE")
    axis.set_xlabel("epoch")
    axis.set_ylabel("MSE (m$^2$)")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def predict_magnitude(model, dataset, *, batch_size: int, seed: int, num_workers: int, device: str) -> pd.DataFrame:
    """Test or validation predictions. Columns are the shared scoring contract."""
    model.eval()
    loader = make_dataloader(dataset, batch_size, seed, num_workers)
    seq_ids, window_ids, y_true, y_pred = [], [], [], []
    torch_device = torch.device(device)
    with torch.no_grad():
        for batch in loader:
            imu, target, meta = _unpack(batch, torch_device)
            if meta is None:
                raise ValueError("prediction batches need meta from DriftDataset")
            prediction = model(imu).reshape(-1).cpu()
            seq_ids.extend(list(meta["seq_id"]))
            window_ids.extend(int(value) for value in meta["window_id"].tolist())
            y_true.extend(target.cpu().tolist())
            y_pred.extend(prediction.tolist())
    frame = pd.DataFrame({
        "seq_id": seq_ids,
        "window_id": window_ids,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    if len(frame) != len(dataset):
        raise RuntimeError("prediction rows do not match the dataset")
    return frame


def run_validation_grid(train_data, val_data, settings: list[dict], **common) -> pd.DataFrame:
    """Fit each setting and record validation loss. The test flight is not accepted here."""
    if val_data.split == "test":
        raise ValueError("the hyper-parameter grid reads the validation block only")
    rows = []
    for setting in settings:
        model_name = setting.get("model_name", common["model_name"])
        kwargs = {**common, **setting}
        kwargs.pop("model_name", None)
        kwargs.update(checkpoint_path=None, curve_path=None, plot_path=None)
        result = train_model(model_name, train_data, val_data, **kwargs)
        rows.append({
            "model_name": model_name,
            "learning_rate": setting.get("learning_rate", common["learning_rate"]),
            "dropout": setting.get("dropout", common["dropout"]),
            "weight_decay": setting.get("weight_decay", common.get("weight_decay")),
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "best_val_mae": result.best_val_mae,
        })
    return pd.DataFrame(rows)


__all__ = [
    "EXPERIMENT_MODELS",
    "TrainResult",
    "build_model",
    "parameter_count",
    "predict_magnitude",
    "run_validation_grid",
    "set_seed",
    "train_model",
]
