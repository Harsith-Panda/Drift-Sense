"""Train the R4 hybrids (E3, E4, E6, E7) on every fold and seed.

  E3  H1: LSTM + 104 features + physics, direct prediction
  E4  H2: physics prior a*phys_mag+b + learned residual (main hybrid)
  E6  H3: LSTM + camera CNN
  E7  best of E3/E4/E6 by VALIDATION MAE (never test) + R3 optical-flow numbers
  E4vec  E4 with the 4-value head (magnitude + dx, dy, dz) for drift correction
  E4cnn  E4 with R2's 1D-CNN as the IMU encoder instead of the LSTM
  IMUcnn / IMUlstm  raw IMU only (no features, no physics), target standardised

Reference rows, fitted on the same inner-training windows and scored on the
same test flight, are written alongside, because H2 must be judged against them:
  PHYS     physics alone, a*phys_mag + b (the same a, b as H2)
  ML_FEAT  Lasso on the 104 features, alpha chosen on validation
  ML_PHYS  Lasso on the 104 features + 4 physics numbers, alpha chosen on validation

Outputs (shared contract):
  results/predictions/<exp>_<fold>.csv            seed 42
  results/predictions/<exp>_seed<s>_<fold>.csv    other seeds
  results/metrics/hybrid_runs.csv                 one row per run
  results/metrics/hybrid_summary.csv              mean and std over seeds
  results/metrics/<stem>_loss.csv, figures/dl/<stem>_loss.png, models/dl/<stem>.pt

Usage:
    python scripts/train_hybrid.py [--experiments E3 E4 E6 E7 E4vec] [--seeds 42 43 44]
    python scripts/train_hybrid.py --only-references   # refresh PHYS/ML_* rows, keep network runs
    python scripts/train_hybrid.py --experiments E3 E6 --min-epochs 10   # warm-up sensitivity, saved as E3_min10, E6_min10

Runs are merged into results/metrics/hybrid_runs.csv: only the experiments
run now are replaced; every other saved row is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.data.dl_dataset import DriftDataset
from vio_drift.evaluation.metrics import score
from vio_drift.models.dl.hybrid import build_hybrid, fit_physics_prior
from vio_drift.models.dl.ml_reference import fit_lasso_reference, predict_reference
from vio_drift.models.dl.hybrid_train import FLOW_COLUMNS, CameraFlowDataset, predict_hybrid, train_hybrid
from vio_drift.models.dl.train import _save_curve, parameter_count

BASE = {  # experiment -> (preset, extra model kwargs, needs camera images)
    "E3": ("H1", {}, False),
    "E4": ("H2", {}, False),
    "E6": ("H3", {}, True),
    "E4vec": ("H2", {"out_dim": 4}, False),
    # Encoder / branch ablations
    "E4cnn": ("H2", {"imu_encoder": "cnn1d"}, False),          # H2 with the 1D-CNN encoder
    "IMUcnn": ("H1", {"imu_encoder": "cnn1d", "use_features": False}, False),  # raw IMU only
    "IMUlstm": ("H1", {"imu_encoder": "lstm", "use_features": False}, False),  # raw IMU only
}


def stem(exp, seed, fold_id, base_seed):
    return f"{exp}_{fold_id}" if seed == base_seed else f"{exp}_seed{seed}_{fold_id}"


def datasets(processed, fold_id, camera, flow, seed):
    common = dict(processed_dir=processed, fold_id=fold_id, use_camera=camera, seed=seed)
    train, val, test = (DriftDataset(**common, split=s) for s in ("train", "validation", "test"))
    if flow:
        train = CameraFlowDataset(train, processed)
        val = CameraFlowDataset(val, processed, stats=train.stats)
        test = CameraFlowDataset(test, processed, stats=train.stats)
    return train, val, test


def reference_rows(processed, folds, out_dir):
    """PHYS, ML_FEAT and ML_PHYS rows: deterministic, so seed 0 and no spread."""
    rows = []
    for fold in folds:
        fold_id = fold["fold_id"]
        train, val, test = datasets(processed, fold_id, False, False, 42)
        prior = fit_physics_prior(train.phys, train.labels[:, 0], train.labels[:, 1:])
        frames = {"PHYS": (pd.DataFrame({"seq_id": [m["seq_id"] for m in test.meta],
                                         "window_id": [m["window_id"] for m in test.meta],
                                         "y_true": test.labels[:, 0],
                                         "y_pred": prior.predict(test.phys)[:, 0]}),
                           {"prior_a": prior.a[0], "prior_b": prior.b[0]})}
        for exp, use_physics in (("ML_FEAT", False), ("ML_PHYS", True)):
            ref = fit_lasso_reference(train, val, use_physics=use_physics)
            coef = ref.model.named_steps["model"].coef_
            frames[exp] = (predict_reference(ref, test),
                           {"alpha": ref.alpha, "best_val_mae_m": ref.val_mae[ref.alpha],
                            "n_nonzero": int((coef != 0).sum()),
                            "phys_nonzero": int((coef[104:] != 0).sum()) if use_physics else 0})
        for exp, (frame, extra) in frames.items():
            frame.to_csv(out_dir / f"{exp}_{fold_id}.csv", index=False, float_format="%.8f")
            s = score(frame.y_true, frame.y_pred)
            rows.append({"experiment": exp, "fold_id": fold_id, "seed": 0, **extra,
                         **{f"test_{k}": v for k, v in s.items()}})
            print(f"{exp}_{fold_id}: test MAE {s['mae'] * 100:.2f} cm, R2 {s['r2']:.3f} {extra}", flush=True)
    return rows


def run(exp, preset, kwargs, camera, flow, fold, seed, config, processed, min_epochs=0):
    hybrid, early = config["hybrid"], config["early_stopping"]
    fold_id = fold["fold_id"]
    train, val, test = datasets(processed, fold_id, camera, flow, seed)
    model = build_hybrid(preset, **{"imu_encoder": hybrid["imu_encoder"], "dropout": config["dropout"],
                                    "n_cam_feats": len(FLOW_COLUMNS) if flow else 0, **kwargs})
    name = stem(exp, seed, fold_id, config["seed"])
    started = time.time()
    result = train_hybrid(
        model, train, val, epochs=config["epochs"], batch_size=config["batch_size"],
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"], seed=seed,
        device=config["device"], num_workers=config["num_workers"], early_stopping=early["enabled"],
        patience=early["patience"], min_delta=early["min_delta"], grad_clip=hybrid["grad_clip"],
        min_epochs=min_epochs, checkpoint_path=ROOT / "models" / "dl" / f"{name}.pt",
        curve_path=ROOT / "results" / "metrics" / f"{name}_loss.csv")
    if seed == config["seed"]:
        _save_curve(result.history.dropna(), ROOT / "figures" / "dl" / f"{name}_loss.png",
                    f"{exp} ({preset}) {fold_id} loss (standardised)")
    predictions = predict_hybrid(result.model, test, batch_size=config["batch_size"], device=config["device"])
    predictions.to_csv(ROOT / "results" / "predictions" / f"{name}.csv", index=False, float_format="%.8f")
    s = score(predictions.y_true, predictions.y_pred)
    h = result.history
    row = {"experiment": exp, "preset": preset, "fold_id": fold_id, "seed": seed, "min_epochs": min_epochs,
           "use_camera": camera, "use_flow": flow, "parameters": parameter_count(result.model),
           "epochs_ran": int(h.epoch.iloc[-1]), "best_epoch": result.best_epoch,
           "start_val_mae_m": float(h.val_mae.iloc[0]), "best_val_mae_m": result.best_val_mae,
           "best_val_loss": result.best_val_loss, "seconds": round(time.time() - started, 1),
           **{f"test_{k}": v for k, v in s.items()}}
    if result.model.out_dim == 4:
        err = predictions[[f"y_true_d{a}" for a in "xyz"]].to_numpy() - predictions[[f"y_pred_d{a}" for a in "xyz"]].to_numpy()
        row["test_vec_mae_m"] = float(np.linalg.norm(err, axis=1).mean())
    print(f"{name}: best epoch {result.best_epoch}, val MAE {result.best_val_mae * 100:.2f} cm, "
          f"test MAE {s['mae'] * 100:.2f} cm, R2 {s['r2']:.3f} ({row['seconds']} s)", flush=True)
    return row


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["best_val_mae_m", "test_mae", "test_rmse", "test_r2", "test_bias", "best_epoch"]
    per_fold = runs.groupby(["experiment", "fold_id"])[metrics].agg(["mean", "std"])
    per_fold.columns = [f"{m}_{a}" for m, a in per_fold.columns]
    per_fold = per_fold.reset_index()
    per_fold["n_seeds"] = runs.groupby(["experiment", "fold_id"]).size().to_numpy()
    overall = runs.groupby(["experiment", "seed"])[metrics].mean().groupby("experiment").agg(["mean", "std"])
    overall.columns = [f"{m}_{a}" for m, a in overall.columns]
    overall = overall.reset_index().assign(fold_id="mean_of_folds")
    return pd.concat([per_fold, overall], ignore_index=True)


def main():
    config = yaml.safe_load((ROOT / "configs" / "dl_config.yaml").read_text())
    hybrid = config["hybrid"]
    parser = argparse.ArgumentParser()
    default = ["E3", "E4", "E6", "E7"] + (["E4vec"] if hybrid["vector_head"] else [])
    parser.add_argument("--experiments", nargs="+", default=default)
    parser.add_argument("--seeds", nargs="+", type=int, default=hybrid["seeds"])
    parser.add_argument("--only-references", action="store_true",
                        help="recompute PHYS/ML_* rows and keep the saved network runs")
    parser.add_argument("--min-epochs", type=int, default=0,
                        help="warm-up before checkpoint selection; runs are saved as <exp>_min<N>")
    args = parser.parse_args()
    processed = ROOT / config["data"]["processed_dir"]
    folds = json.loads((processed / "splits_dl.json").read_text())["folds"]
    (ROOT / "results" / "predictions").mkdir(parents=True, exist_ok=True)

    if args.only_references:
        args.experiments = []
    suffix = f"_min{args.min_epochs}" if args.min_epochs else ""
    rows = reference_rows(processed, folds, ROOT / "results" / "predictions")
    for exp in [e for e in args.experiments if e != "E7"]:
        preset, kwargs, camera = BASE[exp]
        for fold in folds:
            for seed in args.seeds:
                rows.append(run(exp + suffix, preset, kwargs, camera, False, fold, seed, config,
                                processed, args.min_epochs))

    if "E7" in args.experiments:
        done = pd.DataFrame(rows)
        candidates = done[done.experiment.isin(hybrid["e7_candidates"])]
        if candidates.empty:
            raise ValueError("E7 needs at least one of the candidate runs: " + ", ".join(hybrid["e7_candidates"]))
        val = candidates.groupby("experiment").best_val_mae_m.mean().sort_values()
        base = val.index[0]
        print(f"\nE7 base chosen on validation MAE: {base} ({dict(val.round(5))})", flush=True)
        preset, kwargs, camera = BASE[base]
        for fold in folds:
            for seed in args.seeds:
                row = run("E7" + suffix, preset, {**kwargs, "use_features": True}, camera, True, fold,
                          seed, config, processed, args.min_epochs)
                rows.append({**row, "e7_base": base})

    runs = pd.DataFrame(rows)
    saved_path = ROOT / "results" / "metrics" / "hybrid_runs.csv"
    if saved_path.exists():  # keep every saved experiment that was not re-run now
        saved = pd.read_csv(saved_path)
        runs = pd.concat([saved[~saved.experiment.isin(runs.experiment)], runs], ignore_index=True)
    runs.to_csv(ROOT / "results" / "metrics" / "hybrid_runs.csv", index=False)
    summary = summarise(runs)
    summary.to_csv(ROOT / "results" / "metrics" / "hybrid_summary.csv", index=False)
    cols = ["experiment", "fold_id", "test_mae_mean", "test_mae_std", "test_r2_mean", "test_r2_std"]
    view = summary[cols].copy()
    view[["test_mae_mean", "test_mae_std"]] *= 100
    print("\n(test MAE in cm)\n" + view.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
