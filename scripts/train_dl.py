"""Train E1 (1D-CNN) and E2 (LSTM) on every leave-one-flight fold.

The run uses configs/dl_config.yaml. Prediction files follow the shared
contract: results/predictions/<exp_id>_<fold>.csv with columns
seq_id, window_id, y_true, y_pred. Checkpoints go to models/dl/ and loss
curves go to results/metrics/. The test flight is used only after the
best validation checkpoint is restored.

Usage:
    python scripts/train_dl.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Conda NumPy and the pip torch wheel each ship libomp. Set this before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.data.dl_dataset import DriftDataset
from vio_drift.evaluation.metrics import score
from vio_drift.models.dl.train import EXPERIMENT_MODELS, parameter_count, predict_magnitude, train_model


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "dl_config.yaml").read_text())
    processed = ROOT / config["data"]["processed_dir"]
    folds = json.loads((processed / "splits_dl.json").read_text())["folds"]
    if config["data"]["sequences"] != ["MH_01_easy", "MH_02_easy"]:
        raise ValueError("E1 and E2 stay on the two easy flights")
    if config["use_camera"]:
        raise ValueError("E1 and E2 are IMU-only; leave use_camera false")
    early = config["early_stopping"]
    summary = []
    for experiment, model_name in EXPERIMENT_MODELS.items():
        for fold in folds:
            fold_id = fold["fold_id"]
            print(f"\n{experiment} {model_name} {fold_id}", flush=True)
            common = dict(processed_dir=processed, fold_id=fold_id, use_camera=False, seed=config["seed"])
            train_data = DriftDataset(**common, split="train")
            val_data = DriftDataset(**common, split="validation")
            test_data = DriftDataset(**common, split="test")
            stem = f"{experiment}_{fold_id}"
            result = train_model(
                model_name,
                train_data,
                val_data,
                dropout=config["dropout"],
                out_dim=1,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                seed=config["seed"],
                device=config["device"],
                num_workers=config["num_workers"],
                early_stopping=early["enabled"],
                patience=early["patience"],
                min_delta=early["min_delta"],
                use_scheduler=True,
                checkpoint_path=ROOT / "models" / "dl" / f"{stem}.pt",
                curve_path=ROOT / "results" / "metrics" / f"{stem}_loss.csv",
                plot_path=ROOT / "figures" / "dl" / f"{stem}_loss.png",
                log=True,
            )
            predictions = predict_magnitude(
                result.model, test_data,
                batch_size=config["batch_size"], seed=config["seed"],
                num_workers=config["num_workers"], device=config["device"],
            )
            prediction_path = ROOT / "results" / "predictions" / f"{stem}.csv"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(prediction_path, index=False, float_format="%.8f")
            test_score = score(predictions["y_true"], predictions["y_pred"])
            at_best = result.history.loc[result.history["epoch"] == result.best_epoch].iloc[0]
            final = result.history.iloc[-1]
            summary.append({
                "experiment": experiment,
                "model": model_name,
                "fold_id": fold_id,
                "test_seq": fold["test_source"],
                "seed": config["seed"],
                "parameters": parameter_count(result.model),
                "epochs_ran": int(final["epoch"]),
                "best_epoch": result.best_epoch,
                "best_train_loss": float(at_best["train_loss"]),
                "best_val_loss": result.best_val_loss,
                "best_val_mae_m": result.best_val_mae,
                "final_train_loss": float(final["train_loss"]),
                "final_val_loss": float(final["val_loss"]),
                "n_train": len(train_data),
                "n_validation": len(val_data),
                "n_test": len(test_data),
                "test_mae_m": test_score["mae"],
                "test_rmse_m": test_score["rmse"],
                "test_r2": test_score["r2"],
                "test_bias_m": test_score["bias"],
            })
            print(
                f"  best epoch {result.best_epoch}: val MAE {result.best_val_mae * 100:.2f} cm; "
                f"test MAE {test_score['mae'] * 100:.2f} cm, R2 {test_score['r2']:.3f}",
                flush=True,
            )
    out = ROOT / "results" / "metrics" / "imu_e1_e2.csv"
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
