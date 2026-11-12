"""Camera-only training using R1 datasets and R2's seed/result helpers.

R2's existing trainer hard-codes the IMU input and model factory. This R3
runner keeps those files unchanged and follows the same optimisation policy.
"""
from __future__ import annotations

from pathlib import Path

from vio_drift.models.dl.train import TrainResult, set_seed
from vio_drift.models.dl.camera_cnn import CameraCNN
from vio_drift.data.dl_dataset import make_dataloader
import numpy as np
import pandas as pd
import torch


def _epoch(model, dataset, *, batch_size, seed, num_workers, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_abs, count = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in make_dataloader(dataset, batch_size, seed, num_workers):
            cam = batch["cam"].to(device)
            target = batch["y_mag"].to(device).reshape(-1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(cam).reshape(-1)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite camera loss")
            if training:
                loss.backward()
                optimizer.step()
            count += len(target)
            total_loss += loss.item() * len(target)
            total_abs += (prediction.detach() - target).abs().sum().item()
    if count == 0:
        raise ValueError("Empty camera dataset")
    return total_loss / count, total_abs / count


def train_camera(train_data, val_data, *, seed=42, dropout=0.3, epochs=100,
                 batch_size=32, learning_rate=1e-3, weight_decay=1e-4,
                 device="cpu", num_workers=0, early_stopping=True, patience=15,
                 min_delta=1e-5, checkpoint_path=None, curve_path=None, log=False):
    """Train magnitude only; test data cannot be supplied as validation."""
    if train_data.split != "train" or val_data.split != "validation":
        raise ValueError("Expected train and validation splits; test cannot select weights")
    if epochs < 1 or batch_size < 1 or patience < 1 or min_delta < 0:
        raise ValueError("Invalid epoch, batch size or stopping settings")
    set_seed(seed)
    model = CameraCNN(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=min_delta, threshold_mode="abs",
    )
    common = dict(batch_size=batch_size, seed=seed, num_workers=num_workers, device=device)
    best_loss, best_mae, best_epoch, best_state, stale = float("inf"), float("inf"), 0, None, 0
    rows = []
    for epoch in range(1, epochs + 1):
        if hasattr(train_data, "set_epoch"):
            train_data.set_epoch(epoch - 1)
        train_loss, train_mae = _epoch(model, train_data, optimizer=optimizer, **common)
        val_loss, val_mae = _epoch(model, val_data, **common)
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     "train_mae": train_mae, "val_mae": val_mae,
                     "learning_rate": optimizer.param_groups[0]["lr"]})
        scheduler.step(val_loss)
        if val_loss < best_loss - min_delta:
            best_loss, best_mae, best_epoch = val_loss, val_mae, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if log and (epoch == 1 or epoch % 5 == 0):
            print(f"  epoch {epoch}: train MSE {train_loss:.6f}, validation MAE {val_mae * 100:.2f} cm", flush=True)
        if early_stopping and stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    history = pd.DataFrame(rows)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_name": "camera_cnn", "dropout": dropout, "out_dim": 1,
                    "seed": seed, "best_epoch": best_epoch, "best_val_loss": best_loss,
                    "state_dict": best_state, "optimizer": "AdamW", "scheduler": "ReduceLROnPlateau"},
                   checkpoint_path)
    if curve_path is not None:
        Path(curve_path).parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(curve_path, index=False)
    return TrainResult(model, history, best_epoch, best_loss, best_mae, checkpoint_path)


def predict_camera(model, dataset, *, batch_size=32, seed=42, num_workers=0, device="cpu"):
    """Preserve R1's CSV order and R5's four-column prediction contract."""
    if dataset.split not in ("validation", "test"):
        raise ValueError("Predictions require an unshuffled validation/test dataset")
    model.to(device).eval()
    rows = []
    with torch.no_grad():
        for batch in make_dataloader(dataset, batch_size, seed, num_workers):
            predictions = model(batch["cam"].to(device)).reshape(-1).cpu().numpy()
            if not np.isfinite(predictions).all():
                raise ValueError("Nonfinite camera predictions")
            for seq, wid, true, pred in zip(batch["meta"]["seq_id"], batch["meta"]["window_id"], batch["y_mag"], predictions):
                rows.append({"seq_id": seq, "window_id": int(wid), "y_true": float(true), "y_pred": float(pred)})
    if len(rows) != len(dataset):
        raise ValueError("Camera prediction row count mismatch")
    return pd.DataFrame(rows)
