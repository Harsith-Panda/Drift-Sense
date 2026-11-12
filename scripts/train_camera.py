"""Run R3's Tier 1 Lasso comparison and/or camera-only E5, leaving E1/E2 intact.

python scripts/train_camera.py                 # both tiers, three E5 seeds
python scripts/train_camera.py --tier 1        # Phase 1 tuning, with/without flow
python scripts/train_camera.py --tier 2 --seeds 42 --epochs 2  # smoke run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.models.dl.camera_train import train_camera, predict_camera
from vio_drift.models.dl.train import parameter_count
from vio_drift.data.dl_dataset import DriftDataset
from vio_drift.data.splits import load_splits
from vio_drift.evaluation.metrics import score
from vio_drift.features.build_features import feature_names as imu_feature_names
from vio_drift.features.camera import feature_names, file_sha256
from vio_drift.models.ml.data import load_dataset, fold_frames
from vio_drift.models.ml.train import fit_model
import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import yaml


def check_camera_artifacts(processed, sequences):
    """Reject stale window associations and changed generated camera files."""
    for seq in sequences:
        manifest_path = processed / "camera" / f"manifest_{seq}.json"
        manifest = json.loads(manifest_path.read_text())
        for relative, digest in manifest["inputs"].items():
            path = ROOT / relative
            # Training needs only processed arrays; interim caches are optional.
            if path.exists() or path.name == f"windows_{seq}.csv":
                if file_sha256(path) != digest:
                    raise ValueError(f"Camera source changed: {path}; rerun preprocessing")
        for name, digest in manifest["outputs"].items():
            if file_sha256(processed / "camera" / name) != digest:
                raise ValueError(f"Camera output changed: {name}; rerun preprocessing")


def tier1(processed, sequences, results_dir):
    """Same outer folds, alpha grid, purged five-block CV and median rule as Phase 1."""
    data = load_dataset(processed, sequences)
    tables = []
    for seq in sequences:
        camera = pd.read_csv(processed / "camera" / f"camera_feats_{seq}.csv")
        windows = pd.read_csv(processed / f"windows_{seq}.csv", usecols=["seq_id", "window_id"])
        if not camera[["seq_id", "window_id"]].equals(windows):
            raise ValueError("Camera feature row order does not match windows")
        tables.append(camera[["seq_id", "window_id", *feature_names()]])
    data = data.merge(pd.concat(tables, ignore_index=True), on=["seq_id", "window_id"],
                      how="left", validate="one_to_one", sort=False)
    if not np.isfinite(data[feature_names()].to_numpy()).all():
        raise ValueError("Missing/nonfinite camera features")
    diagnostics, predictions = results_dir / "diagnostics", results_dir / "predictions"
    diagnostics.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    rows, cv_rows = [], []
    for fold in load_splits(processed / "splits.json"):
        train, test = fold_frames(data, fold)
        for use_camera in (False, True):
            columns = imu_feature_names() + (feature_names() if use_camera else [])
            model_name = "lasso_camera" if use_camera else "lasso"
            print(f"Tier 1 {fold.fold_id}: {model_name}, {len(columns)} features", flush=True)
            fitted = fit_model("lasso", train[columns], train.err_mag_m)
            pred = fitted.model.predict(test[columns])
            rows.append({"fold_id": fold.fold_id, "model": model_name, "use_camera": use_camera,
                         "n_features": len(columns), "n_train": len(train), "n_test": len(test),
                         "alpha": fitted.alpha, **score(test.err_mag_m, pred)})
            cv_rows.append(fitted.cv_table.assign(fold_id=fold.fold_id, model=model_name))
            prediction = test[["seq_id", "window_id"]].copy()
            prediction["y_true"], prediction["y_pred"] = test.err_mag_m, pred
            prediction.to_csv(predictions / f"T1_{model_name}_{fold.fold_id}.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(diagnostics / "camera_tier1_check.csv", index=False)
    pd.concat(cv_rows, ignore_index=True).to_csv(diagnostics / "camera_tier1_cv.csv", index=False)
    print(table.to_string(index=False), flush=True)
    return table


def save_loss_plot(history, destination, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(history.epoch, history.train_loss, label="train (dropout on)")
    ax.plot(history.epoch, history.val_loss, label="validation")
    ax.set(xlabel="epoch", ylabel="MSE (m²)", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def tier2(processed, config, results_dir, model_dir, figure_dir, seeds, epochs):
    folds = json.loads((processed / "splits_dl.json").read_text())["folds"]
    metrics, predictions = results_dir / "metrics", results_dir / "predictions"
    metrics.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in folds:
        fold_id = fold["fold_id"]
        common = dict(processed_dir=processed, fold_id=fold_id, use_camera=True)
        train_data = DriftDataset(**common, split="train")
        val_data = DriftDataset(**common, split="validation")
        for seed in seeds:
            exp_id = "E5" if seed == config["seed"] else f"E5_seed{seed}"
            stem = f"{exp_id}_{fold_id}"
            print(f"E5 camera-only {fold_id}, seed {seed}", flush=True)
            early = config["early_stopping"]
            result = train_camera(
                train_data, val_data, seed=seed, dropout=config["dropout"], epochs=epochs,
                batch_size=config["batch_size"], learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"], device=config["device"], num_workers=config["num_workers"],
                early_stopping=early["enabled"], patience=early["patience"], min_delta=early["min_delta"],
                checkpoint_path=model_dir / f"{stem}.pt", curve_path=metrics / f"{stem}_loss.csv", log=True,
            )
            # Test-flight tensors enter only after the best validation checkpoint exists.
            test_data = DriftDataset(**common, split="test")
            pred = predict_camera(result.model, test_data, batch_size=config["batch_size"], seed=seed,
                                  num_workers=config["num_workers"], device=config["device"])
            pred.to_csv(predictions / f"{stem}.csv", index=False)
            scores = score(pred.y_true, pred.y_pred)
            rows.append({"experiment": "E5", "fold_id": fold_id, "seed": seed,
                         "parameters": parameter_count(result.model), "n_train": len(train_data),
                         "n_validation": len(val_data), "n_test": len(test_data),
                         "epochs_ran": len(result.history), "best_epoch": result.best_epoch,
                         "best_val_loss": result.best_val_loss, "best_val_mae_m": result.best_val_mae,
                         **{f"test_{key}": value for key, value in scores.items()}})
            save_loss_plot(result.history, figure_dir / f"camera_{stem}_loss.png", f"E5 {fold_id}, seed {seed}")
            print(f"  best epoch {result.best_epoch}: test MAE {scores['mae'] * 100:.2f} cm, R² {scores['r2']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(metrics / "camera_e5.csv", index=False)
    measures = ["test_mae", "test_rmse", "test_r2", "test_bias"]
    summary = table.groupby("fold_id")[measures].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary["n_seeds"] = table.groupby("fold_id").size()
    summary.to_csv(metrics / "camera_e5_summary.csv")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("1", "2", "both"), default="both")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path, help="Separate root for smoke experiments (results/models/figures)")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/dl_config.yaml").read_text())
    sequences = config["data"]["sequences"]
    if sequences != ["MH_01_easy", "MH_02_easy"]:
        raise ValueError("R3 uses only the two easy flights")
    processed = ROOT / config["data"]["processed_dir"]
    check_camera_artifacts(processed, sequences)
    seeds = args.seeds if args.seeds is not None else config["camera"]["seeds"]
    if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("Use distinct nonnegative 32-bit seeds")
    epochs = config["epochs"] if args.epochs is None else args.epochs
    destination = args.output_dir.resolve() if args.output_dir else ROOT
    results = destination / "results"
    if args.tier in ("1", "both"):
        tier1(processed, sequences, results)
    if args.tier in ("2", "both"):
        tier2(processed, config, results, destination / "models/dl", destination / "figures/dl", seeds, epochs)
    inputs = [processed / "splits.json", processed / "splits_dl.json"]
    inputs += [processed / "camera" / f"manifest_{seq}.json" for seq in sequences]
    manifest = {"tier": args.tier, "config": config, "e5_uses_camera": True,
                "seeds": seeds, "epochs": epochs,
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__,
                             "opencv": cv2.__version__, "torch": torch.__version__},
                "inputs": {str(p.relative_to(ROOT)): file_sha256(p) for p in inputs}}
    (results / "diagnostics").mkdir(parents=True, exist_ok=True)
    (results / "diagnostics" / f"camera_run_{args.tier}.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
