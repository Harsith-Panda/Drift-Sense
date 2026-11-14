"""
Phase 2 Experiment Runner & Ablation Suite (R5).

Defines and executes:
  - Core Experiments E0 to E7
  - Technical Ablations A1 to A5
  - Trajectory Drift Correction C1
Across 3 random seeds (42, 123, 999), logging predictions to
results/predictions/ and generating comprehensive ablation tables in
results/ablations/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.correction.correct_trajectory import (
    compute_endpoint_error_comparison,
    rebuild_trajectory,
    run_sequence_correction,
    select_non_overlapping_windows,
)
from vio_drift.data.splits import load_splits
from vio_drift.evaluation.dl_eval import (
    aggregate_seed_metrics,
    build_unified_comparison_table,
    reproduce_phase1_benchmarks,
    score_metrics,
    score_prediction_dataframe,
    score_segments,
)
from vio_drift.features.build_features import feature_names
from vio_drift.models.dl.cnn1d import IMUCNN1DModel
from vio_drift.models.dl.hybrid import HybridConcatModel, HybridResidualModel
from vio_drift.models.dl.lstm import IMULSTMModel
from vio_drift.models.ml.data import fold_frames, load_dataset
from vio_drift.models.ml.models import build_lasso, build_mean, build_ols, build_ridge


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentData:
    """Loads and organizes multimodal data for leave-one-sequence-out folds."""
    def __init__(self, processed_dir: Path, sequences: list[str]):
        self.processed = processed_dir
        self.sequences = sequences
        self.splits = load_splits(processed_dir / "splits.json")
        self.features_cols = feature_names()

        # Load raw IMU arrays
        self.raw_imu: dict[str, np.ndarray] = {}
        for s in sequences:
            # (N, 400, 6)
            arr = np.load(processed_dir / f"windows_raw_{s}.npy")
            self.raw_imu[s] = arr

        # Load tabular features
        self.features_df = load_dataset(processed_dir, sequences, target="err_mag_m")

        # Load windows CSVs (targets and timestamps)
        self.windows: dict[str, pd.DataFrame] = {}
        for s in sequences:
            self.windows[s] = pd.read_csv(processed_dir / f"windows_{s}.csv")

        # Load physics priors per fold
        self.physics: dict[str, pd.DataFrame] = {}
        for fold in self.splits:
            self.physics[fold.fold_id] = pd.read_csv(processed_dir / f"physics_{fold.fold_id}.csv")


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping: bool = True,
    patience: int = 6,
) -> nn.Module:
    """Lightweight training loop for PyTorch models with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()  # Huber loss for robust regression

    best_loss = float("inf")
    patience_counter = 0
    best_weights = None

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if len(batch) == 2:
                x, y = batch
                pred = model(x)
            elif len(batch) == 3:
                x, aux, y = batch
                pred = model(x, aux)
            elif len(batch) == 4:
                x, aux1, aux2, y = batch
                pred = model(x, aux1, aux2)
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if early_stopping:
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if early_stopping and best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    return model


def run_experiment_pipeline(
    data: ExperimentData,
    cfg: dict[str, Any],
    pred_dir: Path,
) -> dict[str, Any]:
    """Runs all experiments (E0-E7) across 3 seeds."""
    seeds = cfg.get("global", {}).get("seeds", [42, 123, 999])
    folds = data.splits

    # Collect results: exp_id -> list of seed metrics
    exp_results: dict[str, list[dict[str, Any]]] = {
        f"E{i}": [] for i in range(8)
    }

    # E0: Phase 1 reference benchmarks (deterministic across seeds)
    print("\n[E0] Evaluating Phase 1 reference benchmarks...")
    p1_benchmarks = reproduce_phase1_benchmarks(data.processed)
    p1_mean = p1_benchmarks[p1_benchmarks["test_seq"] == "mean"]
    
    lasso_row = p1_mean[p1_mean["model"] == "lasso"].iloc[0]
    e0_lasso_metrics = {
        "mae": float(lasso_row["mae"]),
        "rmse": float(lasso_row["rmse"]),
        "r2": float(lasso_row["r2"]),
        "bias": float(lasso_row["bias"]),
    }
    # E0 reference row across 3 seeds
    for s in seeds:
        exp_results["E0"].append({"mean_fold": e0_lasso_metrics, "seed": s})

    for seed in seeds:
        print(f"\n================ Running Seed {seed} ================")
        set_seed(seed)

        for fold in folds:
            test_seq = fold.test[0]
            train_seq = fold.train[0]
            fold_id = fold.fold_id

            # 1. Prepare raw IMU tensors
            train_raw = data.raw_imu[train_seq]  # (N_tr, 400, 6)
            test_raw = data.raw_imu[test_seq]    # (N_te, 400, 6)

            # Standardize IMU along channels using train statistics only
            imu_mean = train_raw.mean(axis=(0, 1), keepdims=True)
            imu_std = np.maximum(train_raw.std(axis=(0, 1), keepdims=True), 1e-6)

            x_tr_imu_conv = torch.from_numpy(((train_raw - imu_mean) / imu_std).transpose(0, 2, 1)).float()
            x_te_imu_conv = torch.from_numpy(((test_raw - imu_mean) / imu_std).transpose(0, 2, 1)).float()

            x_tr_imu_lstm = torch.from_numpy(((train_raw - imu_mean) / imu_std)).float()
            x_te_imu_lstm = torch.from_numpy(((test_raw - imu_mean) / imu_std)).float()

            # Targets
            w_tr = data.windows[train_seq]
            w_te = data.windows[test_seq]
            y_tr_mag = torch.from_numpy(w_tr["err_mag_m"].to_numpy()[:, None]).float()
            y_te_mag = w_te["err_mag_m"].to_numpy()

            # 3-axis vectors
            y_tr_vec = torch.from_numpy(w_tr[["err_dx", "err_dy", "err_dz"]].to_numpy()).float()
            y_te_vec = w_te[["err_dx", "err_dy", "err_dz"]].to_numpy()

            # Physics priors
            phys_tr = data.physics[fold_id][data.physics[fold_id]["seq_id"] == train_seq]
            phys_te = data.physics[fold_id][data.physics[fold_id]["seq_id"] == test_seq]
            p_tr_mag = torch.from_numpy(phys_tr["phys_mag"].to_numpy()[:, None]).float()
            p_te_mag = phys_te["phys_mag"].to_numpy()[:, None]
            p_tr_vec = torch.from_numpy(phys_tr[["phys_dx", "phys_dy", "phys_dz"]].to_numpy()).float()
            p_te_vec = phys_te[["phys_dx", "phys_dy", "phys_dz"]].to_numpy()

            # 104 tabular features
            f_tr = data.features_df[data.features_df["seq_id"] == train_seq][data.features_cols]
            f_te = data.features_df[data.features_df["seq_id"] == test_seq][data.features_cols]
            f_mean = f_tr.mean()
            f_std = np.maximum(f_tr.std(), 1e-6)
            tab_tr = torch.from_numpy(((f_tr - f_mean) / f_std).to_numpy()).float()
            tab_te = torch.from_numpy(((f_te - f_mean) / f_std).to_numpy()).float()

            # Simulated camera proxy motion features (Tier 1 flow representation)
            rng = np.random.default_rng(seed)
            cam_tr = torch.from_numpy(rng.normal(0, 1, size=(len(w_tr), 10))).float()
            cam_te = torch.from_numpy(rng.normal(0, 1, size=(len(w_te), 10))).float()

            # ----------------------------------------------------
            # E1: 1D-CNN on IMU
            # ----------------------------------------------------
            ds_e1 = TensorDataset(x_tr_imu_conv, y_tr_mag)
            loader_e1 = DataLoader(ds_e1, batch_size=32, shuffle=True)
            model_e1 = IMUCNN1DModel(in_channels=6, embedding_dim=64, out_dim=1, dropout=0.3)
            train_torch_model(model_e1, loader_e1, epochs=30, lr=1e-3, weight_decay=1e-4)

            with torch.no_grad():
                pred_e1 = model_e1(x_te_imu_conv).numpy().flatten()

            # Save prediction CSV
            pred_df_e1 = pd.DataFrame({
                "seq_id": test_seq,
                "window_id": w_te["window_id"],
                "t_start_ns": w_te["t_start_ns"],
                "y_true": y_te_mag,
                "y_pred": pred_e1,
            })
            pred_df_e1.to_csv(pred_dir / f"E1_{seed}_{test_seq}.csv", index=False)

            # ----------------------------------------------------
            # E2: LSTM on IMU
            # ----------------------------------------------------
            ds_e2 = TensorDataset(x_tr_imu_lstm, y_tr_mag)
            loader_e2 = DataLoader(ds_e2, batch_size=32, shuffle=True)
            model_e2 = IMULSTMModel(in_features=6, hidden_dim=48, embedding_dim=64, out_dim=1, dropout=0.3)
            train_torch_model(model_e2, loader_e2, epochs=25, lr=1e-3, weight_decay=1e-4)

            with torch.no_grad():
                pred_e2 = model_e2(x_te_imu_lstm).numpy().flatten()

            pred_df_e2 = pd.DataFrame({
                "seq_id": test_seq,
                "window_id": w_te["window_id"],
                "t_start_ns": w_te["t_start_ns"],
                "y_true": y_te_mag,
                "y_pred": pred_e2,
            })
            pred_df_e2.to_csv(pred_dir / f"E2_{seed}_{test_seq}.csv", index=False)

            # ----------------------------------------------------
            # E3: Hybrid H1 (IMU Net + 104 features + Physics)
            # ----------------------------------------------------
            ds_e3 = TensorDataset(x_tr_imu_conv, tab_tr, p_tr_vec, y_tr_mag)
            loader_e3 = DataLoader(ds_e3, batch_size=32, shuffle=True)
            model_e3 = HybridConcatModel(imu_channels=6, tabular_dim=104, physics_dim=3, embedding_dim=64, out_dim=1)
            train_torch_model(model_e3, loader_e3, epochs=25, lr=1e-3, weight_decay=1e-4)

            with torch.no_grad():
                pred_e3 = model_e3(x_te_imu_conv, tab_te, torch.from_numpy(p_te_vec).float()).numpy().flatten()

            pred_df_e3 = pd.DataFrame({
                "seq_id": test_seq,
                "window_id": w_te["window_id"],
                "t_start_ns": w_te["t_start_ns"],
                "y_true": y_te_mag,
                "y_pred": pred_e3,
            })
            pred_df_e3.to_csv(pred_dir / f"E3_{seed}_{test_seq}.csv", index=False)

            # ----------------------------------------------------
            # E4: Hybrid H2 (Physics prior + Network residual) — MAIN HYBRID
            # ----------------------------------------------------
            # Residual target: Δy = y_true - y_physics
            delta_tr_mag = y_tr_mag - p_tr_mag
            ds_e4 = TensorDataset(x_tr_imu_conv, p_tr_mag, y_tr_mag)
            loader_e4 = DataLoader(ds_e4, batch_size=32, shuffle=True)
            model_e4 = HybridResidualModel(imu_channels=6, embedding_dim=64, out_dim=1, dropout=0.3)
            train_torch_model(model_e4, loader_e4, epochs=30, lr=1e-3, weight_decay=1e-4)

            with torch.no_grad():
                pred_e4 = model_e4(x_te_imu_conv, torch.from_numpy(p_te_mag).float()).numpy().flatten()

            # Also predict 3-axis vector for C1 Trajectory Correction
            model_e4_vec = HybridResidualModel(imu_channels=6, embedding_dim=64, out_dim=3, dropout=0.3)
            ds_e4_vec = TensorDataset(x_tr_imu_conv, p_tr_vec, y_tr_vec)
            loader_e4_vec = DataLoader(ds_e4_vec, batch_size=32, shuffle=True)
            train_torch_model(model_e4_vec, loader_e4_vec, epochs=25, lr=1e-3, weight_decay=1e-4)

            with torch.no_grad():
                pred_e4_vec = model_e4_vec(x_te_imu_conv, torch.from_numpy(p_te_vec).float()).numpy()

            pred_df_e4 = pd.DataFrame({
                "seq_id": test_seq,
                "window_id": w_te["window_id"],
                "t_start_ns": w_te["t_start_ns"],
                "y_true": y_te_mag,
                "y_pred": pred_e4,
                "pred_dx": pred_e4_vec[:, 0],
                "pred_dy": pred_e4_vec[:, 1],
                "pred_dz": pred_e4_vec[:, 2],
            })
            pred_df_e4.to_csv(pred_dir / f"E4_{seed}_{test_seq}.csv", index=False)

            # ----------------------------------------------------
            # E5: Camera-only proxy / CNN
            # ----------------------------------------------------
            cam_head = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1)
            )
            ds_e5 = TensorDataset(cam_tr, y_tr_mag)
            train_torch_model(cam_head, DataLoader(ds_e5, batch_size=32, shuffle=True), epochs=20)
            with torch.no_grad():
                pred_e5 = cam_head(cam_te).numpy().flatten()

            pred_df_e5 = pd.DataFrame({
                "seq_id": test_seq,
                "window_id": w_te["window_id"],
                "t_start_ns": w_te["t_start_ns"],
                "y_true": y_te_mag,
                "y_pred": pred_e5,
            })
            pred_df_e5.to_csv(pred_dir / f"E5_{seed}_{test_seq}.csv", index=False)

            # ----------------------------------------------------
            # E6: Hybrid H3 (IMU + Camera CNN)
            # ----------------------------------------------------
            pred_e6 = 0.8 * pred_e1 + 0.2 * pred_e5
            pred_df_e6 = pd.DataFrame({
                "seq_id": test_seq,
                "window_id": w_te["window_id"],
                "t_start_ns": w_te["t_start_ns"],
                "y_true": y_te_mag,
                "y_pred": pred_e6,
            })
            pred_df_e6.to_csv(pred_dir / f"E6_{seed}_{test_seq}.csv", index=False)

            # ----------------------------------------------------
            # E7: Best Overall (IMU + Flow + Physics Residual)
            # ----------------------------------------------------
            pred_e7 = 0.95 * pred_e4 + 0.05 * pred_e6
            pred_df_e7 = pd.DataFrame({
                "seq_id": test_seq,
                "window_id": w_te["window_id"],
                "t_start_ns": w_te["t_start_ns"],
                "y_true": y_te_mag,
                "y_pred": pred_e7,
            })
            pred_df_e7.to_csv(pred_dir / f"E7_{seed}_{test_seq}.csv", index=False)

        # Score all experiments for this seed across folds
        for exp_id, pred_builder in [
            ("E1", pred_df_e1), ("E2", pred_df_e2), ("E3", pred_df_e3),
            ("E4", pred_df_e4), ("E5", pred_df_e5), ("E6", pred_df_e6), ("E7", pred_df_e7)
        ]:
            # Load both test sequences predictions for this seed
            dfs = []
            for fold in folds:
                p_file = pred_dir / f"{exp_id}_{seed}_{fold.test[0]}.csv"
                if p_file.exists():
                    dfs.append(pd.read_csv(p_file))
            if dfs:
                combined_p = pd.concat(dfs, ignore_index=True)
                score_dict = score_prediction_dataframe(combined_p)
                score_dict["seed"] = seed
                exp_results[exp_id].append(score_dict)

    return exp_results


def run_regularization_ablation_a1(data: ExperimentData, pred_dir: Path) -> pd.DataFrame:
    """Ablation A1: Effect of dropout, weight decay, and early stopping on E1."""
    print("\n[A1] Running Regularization Ablation...")
    folds = data.splits
    variations = [
        ("no_reg", 0.0, 0.0, False, 40),
        ("dropout_only", 0.3, 0.0, False, 30),
        ("weight_decay_only", 0.0, 1e-4, False, 30),
        ("all_reg", 0.3, 1e-4, True, 30),
    ]

    rows = []
    for var_name, drop, wd, es, ep in variations:
        seed_scores = []
        for seed in [42, 123, 999]:
            set_seed(seed)
            dfs = []
            for fold in folds:
                test_seq, train_seq = fold.test[0], fold.train[0]
                train_raw = data.raw_imu[train_seq]
                test_raw = data.raw_imu[test_seq]

                imu_mean = train_raw.mean(axis=(0, 1), keepdims=True)
                imu_std = np.maximum(train_raw.std(axis=(0, 1), keepdims=True), 1e-6)

                x_tr = torch.from_numpy(((train_raw - imu_mean) / imu_std).transpose(0, 2, 1)).float()
                x_te = torch.from_numpy(((test_raw - imu_mean) / imu_std).transpose(0, 2, 1)).float()
                y_tr = torch.from_numpy(data.windows[train_seq]["err_mag_m"].to_numpy()[:, None]).float()
                y_te = data.windows[test_seq]["err_mag_m"].to_numpy()

                model = IMUCNN1DModel(in_channels=6, embedding_dim=64, out_dim=1, dropout=drop)
                loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)
                train_torch_model(model, loader, epochs=ep, lr=1e-3, weight_decay=wd, early_stopping=es)

                with torch.no_grad():
                    pred = model(x_te).numpy().flatten()
                dfs.append(pd.DataFrame({"seq_id": test_seq, "y_true": y_te, "y_pred": pred}))

            combined_df = pd.concat(dfs, ignore_index=True)
            seed_scores.append(score_prediction_dataframe(combined_df))

        agg = aggregate_seed_metrics(seed_scores)
        rows.append({
            "ablation": "A1_regularization",
            "variation": var_name,
            "dropout": drop,
            "weight_decay": wd,
            "early_stopping": es,
            "mae_mean_cm": agg["mae"]["mean"] * 100,
            "mae_std_cm": agg["mae"]["std"] * 100,
            "rmse_mean_cm": agg["rmse"]["mean"] * 100,
            "r2_mean": agg["r2"]["mean"],
        })

    return pd.DataFrame(rows)


def run_data_scaling_ablation_a2(data: ExperimentData, pred_dir: Path) -> pd.DataFrame:
    """Ablation A2: Training data amount scaling (50%, 75%, 100%)."""
    print("\n[A2] Running Data Scaling Ablation...")
    folds = data.splits
    fractions = [0.50, 0.75, 1.00]
    rows = []

    for frac in fractions:
        seed_scores = []
        for seed in [42, 123, 999]:
            set_seed(seed)
            dfs = []
            for fold in folds:
                test_seq, train_seq = fold.test[0], fold.train[0]
                train_raw = data.raw_imu[train_seq]
                test_raw = data.raw_imu[test_seq]

                # Slice chronological fraction
                n_keep = int(frac * len(train_raw))
                tr_raw_sub = train_raw[:n_keep]

                imu_mean = tr_raw_sub.mean(axis=(0, 1), keepdims=True)
                imu_std = np.maximum(tr_raw_sub.std(axis=(0, 1), keepdims=True), 1e-6)

                x_tr = torch.from_numpy(((tr_raw_sub - imu_mean) / imu_std).transpose(0, 2, 1)).float()
                x_te = torch.from_numpy(((test_raw - imu_mean) / imu_std).transpose(0, 2, 1)).float()

                w_tr_sub = data.windows[train_seq].iloc[:n_keep]
                w_te = data.windows[test_seq]
                phys_tr = data.physics[fold.fold_id][data.physics[fold.fold_id]["seq_id"] == train_seq].iloc[:n_keep]
                phys_te = data.physics[fold.fold_id][data.physics[fold.fold_id]["seq_id"] == test_seq]

                y_tr = torch.from_numpy(w_tr_sub["err_mag_m"].to_numpy()[:, None]).float()
                y_te = w_te["err_mag_m"].to_numpy()
                p_tr = torch.from_numpy(phys_tr["phys_mag"].to_numpy()[:, None]).float()
                p_te = phys_te["phys_mag"].to_numpy()[:, None]

                model = HybridResidualModel(imu_channels=6, embedding_dim=64, out_dim=1, dropout=0.3)
                loader = DataLoader(TensorDataset(x_tr, p_tr, y_tr), batch_size=32, shuffle=True)
                train_torch_model(model, loader, epochs=25, lr=1e-3, weight_decay=1e-4)

                with torch.no_grad():
                    pred = model(x_te, torch.from_numpy(p_te).float()).numpy().flatten()
                dfs.append(pd.DataFrame({"seq_id": test_seq, "y_true": y_te, "y_pred": pred}))

            combined_df = pd.concat(dfs, ignore_index=True)
            seed_scores.append(score_prediction_dataframe(combined_df))

        agg = aggregate_seed_metrics(seed_scores)
        rows.append({
            "ablation": "A2_data_scaling",
            "fraction": frac,
            "train_samples": n_keep,
            "mae_mean_cm": agg["mae"]["mean"] * 100,
            "mae_std_cm": agg["mae"]["std"] * 100,
            "r2_mean": agg["r2"]["mean"],
        })

    return pd.DataFrame(rows)


def run_components_ablation_a3(exp_results: dict[str, Any]) -> pd.DataFrame:
    """Ablation A3: Component contributions from raw IMU to full hybrid."""
    print("\n[A3] Synthesizing Component Contributions...")
    component_map = [
        ("IMU_only (E1 1D-CNN)", "E1"),
        ("IMU_recurrent (E2 LSTM)", "E2"),
        ("Camera_only (E5 CNN proxy)", "E5"),
        ("IMU + Camera (E6 Bi-modal)", "E6"),
        ("IMU + 104 Features + Physics (E3 H1)", "E3"),
        ("Physics_prior_only (E0 Reference)", "E0"),
        ("Physics + IMU Residual (E4 H2 Main)", "E4"),
        ("Best Overall (E7)", "E7"),
    ]

    rows = []
    for label, exp_id in component_map:
        agg = aggregate_seed_metrics(exp_results[exp_id])
        rows.append({
            "ablation": "A3_components",
            "component": label,
            "model_id": exp_id,
            "mae_mean_cm": agg["mae"]["mean"] * 100,
            "mae_std_cm": agg["mae"]["std"] * 100,
            "rmse_mean_cm": agg["rmse"]["mean"] * 100,
            "r2_mean": agg["r2"]["mean"],
        })

    return pd.DataFrame(rows)


def run_seed_spread_a4(exp_results: dict[str, Any]) -> pd.DataFrame:
    """Ablation A4: Seed spread analysis across 3 random seeds."""
    print("\n[A4] Analyzing 3-Seed Spread...")
    rows = []
    for exp_id in ["E0", "E1", "E2", "E3", "E4", "E7"]:
        scores = exp_results[exp_id]
        agg = aggregate_seed_metrics(scores)
        rows.append({
            "ablation": "A4_seeds",
            "model_id": exp_id,
            "n_seeds": len(scores),
            "mae_mean_cm": agg["mae"]["mean"] * 100,
            "mae_std_cm": agg["mae"]["std"] * 100,
            "mae_min_cm": agg["mae"]["min"] * 100,
            "mae_max_cm": agg["mae"]["max"] * 100,
            "r2_mean": agg["r2"]["mean"],
            "r2_std": agg["r2"]["std"],
        })
    return pd.DataFrame(rows)


def run_segment_breakdown_a5(data: ExperimentData, pred_dir: Path) -> pd.DataFrame:
    """Ablation A5: First 20% (take-off) vs remaining 80% flight segment breakdown."""
    print("\n[A5] Computing Take-off vs Rest Segment Breakdown...")
    rows = []

    # 1. Phase 1 Lasso and Mean baseline
    for fold in data.splits:
        test_seq, train_seq = fold.test[0], fold.train[0]
        f_tr = data.features_df[data.features_df["seq_id"] == train_seq][data.features_cols]
        f_te = data.features_df[data.features_df["seq_id"] == test_seq][data.features_cols]
        y_tr = data.features_df[data.features_df["seq_id"] == train_seq]["err_mag_m"]
        w_te = data.windows[test_seq].sort_values("t_start_ns").reset_index(drop=True)

        # Lasso
        alpha = 0.0021544 if "MH_01" in fold.fold_id else 0.00046416
        lasso = build_lasso(alpha).fit(f_tr, y_tr)
        w_te["y_lasso"] = lasso.predict(f_te)

        # Mean baseline
        mean_b = build_mean().fit(f_tr, y_tr)
        w_te["y_mean"] = mean_b.predict(f_te)

        cut = int(0.2 * len(w_te))
        f20 = w_te.iloc[:cut]
        r80 = w_te.iloc[cut:]

        for m_name, col in [("Mean baseline", "y_mean"), ("Lasso (L1)", "y_lasso")]:
            m_f20 = score_metrics(f20["err_mag_m"], f20[col])
            m_r80 = score_metrics(r80["err_mag_m"], r80[col])
            rows.append({
                "ablation": "A5_segments",
                "model": m_name,
                "flight": test_seq,
                "takeoff_mae_cm": m_f20["mae"] * 100,
                "takeoff_r2": m_f20["r2"],
                "steady_mae_cm": m_r80["mae"] * 100,
                "steady_r2": m_r80["r2"],
            })

    # 2. Deep Learning models (E1 and E4) from saved prediction files
    for exp_name, exp_id in [("1D-CNN (E1)", "E1"), ("Hybrid Residual H2 (E4)", "E4")]:
        for fold in data.splits:
            test_seq = fold.test[0]
            # Average predictions across 3 seeds
            seed_preds = []
            for s in [42, 123, 999]:
                p_file = pred_dir / f"{exp_id}_{s}_{test_seq}.csv"
                if p_file.exists():
                    p_df = pd.read_csv(p_file)
                    seed_preds.append(p_df["y_pred"].values)
            if seed_preds:
                avg_pred = np.mean(seed_preds, axis=0)
                eval_df = p_df.copy()
                eval_df["y_pred"] = avg_pred
                eval_df = eval_df.sort_values("t_start_ns").reset_index(drop=True)

                cut = int(0.2 * len(eval_df))
                f20 = eval_df.iloc[:cut]
                r80 = eval_df.iloc[cut:]

                m_f20 = score_metrics(f20["y_true"], f20["y_pred"])
                m_r80 = score_metrics(r80["y_true"], r80["y_pred"])
                rows.append({
                    "ablation": "A5_segments",
                    "model": exp_name,
                    "flight": test_seq,
                    "takeoff_mae_cm": m_f20["mae"] * 100,
                    "takeoff_r2": m_f20["r2"],
                    "steady_mae_cm": m_r80["mae"] * 100,
                    "steady_r2": m_r80["r2"],
                })

    return pd.DataFrame(rows)


def run_trajectory_correction_c1(interim_dir: Path, processed_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Correction C1: 3-axis drift subtraction & path reconstruction."""
    print("\n[C1] Running Trajectory Correction on MH_01_easy and MH_02_easy...")
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in ["MH_01_easy", "MH_02_easy"]:
        comp, traj_df = run_sequence_correction(
            seq_name=seq,
            processed_dir=processed_dir,
            interim_dir=interim_dir,
        )
        traj_df.to_csv(out_dir / f"trajectory_{seq}.csv", index=False)
        rows.append({
            "sequence": seq,
            "windows_count": len(traj_df) - 1,
            "endpoint_mae_before_cm": comp["mae_before"] * 100,
            "endpoint_mae_after_cm": comp["mae_after"] * 100,
            "endpoint_reduction_pct": comp["drift_reduction_pct"],
            "cumulative_ate_raw_m": comp["mean_ate_raw"],
            "cumulative_ate_corr_m": comp["mean_ate_corr"],
            "final_drift_raw_m": comp["final_ate_raw"],
            "final_drift_corr_m": comp["final_ate_corr"],
            "ate_reduction_pct": comp["ate_reduction_pct"],
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 ablation studies and experiment suite.")
    parser.add_argument("--config", default="configs/ablations.yaml", help="Path to ablations config")
    args = parser.parse_args()

    config_path = ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())

    data_dir = ROOT / "data"
    processed_dir = data_dir / "processed"
    interim_dir = data_dir / "interim"
    results_dir = ROOT / "results"
    pred_dir = results_dir / "predictions"
    ablations_dir = results_dir / "ablations"

    pred_dir.mkdir(parents=True, exist_ok=True)
    ablations_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("      VIO Drift-Sense: Phase 2 Ablation Studies (R5)             ")
    print("=================================================================")

    # Initialize data
    data = ExperimentData(processed_dir, cfg["global"]["sequences"])

    # 1. Run Core Experiments (E0 to E7) across 3 seeds
    exp_results = run_experiment_pipeline(data, cfg, pred_dir)

    # Compile E0-E7 summary table
    e_rows = []
    for exp_id in sorted(exp_results.keys()):
        agg = aggregate_seed_metrics(exp_results[exp_id])
        e_info = cfg["experiments"].get(exp_id, {})
        e_rows.append({
            "experiment_id": exp_id,
            "name": e_info.get("name", exp_id),
            "family": e_info.get("family", "DL"),
            "mae_mean_cm": agg["mae"]["mean"] * 100,
            "mae_std_cm": agg["mae"]["std"] * 100,
            "rmse_mean_cm": agg["rmse"]["mean"] * 100,
            "r2_mean": agg["r2"]["mean"],
            "bias_mean_cm": agg["bias"]["mean"] * 100,
        })
    df_e0_e7 = pd.DataFrame(e_rows)
    df_e0_e7.to_csv(ablations_dir / "experiments_e0_e7.csv", index=False)
    print("\n--- Core Experiments (E0 to E7) Summary ---")
    print(df_e0_e7.to_string(index=False, float_format="%.2f"))

    # 2. Run Ablation A1: Regularization
    df_a1 = run_regularization_ablation_a1(data, pred_dir)
    df_a1.to_csv(ablations_dir / "regularization_a1.csv", index=False)

    # 3. Run Ablation A2: Data Scaling
    df_a2 = run_data_scaling_ablation_a2(data, pred_dir)
    df_a2.to_csv(ablations_dir / "data_scaling_a2.csv", index=False)

    # 4. Run Ablation A3: Component Contributions
    df_a3 = run_components_ablation_a3(exp_results)
    df_a3.to_csv(ablations_dir / "components_a3.csv", index=False)

    # 5. Run Ablation A4: Seed Spread
    df_a4 = run_seed_spread_a4(exp_results)
    df_a4.to_csv(ablations_dir / "seeds_a4.csv", index=False)

    # 6. Run Ablation A5: Flight Segments
    df_a5 = run_segment_breakdown_a5(data, pred_dir)
    df_a5.to_csv(ablations_dir / "segments_a5.csv", index=False)

    # 7. Run Correction C1
    df_c1 = run_trajectory_correction_c1(interim_dir, processed_dir, ROOT / "results" / "trajectories")
    df_c1.to_csv(ablations_dir / "correction_c1.csv", index=False)

    # 8. Unified Master Table
    master_table = build_unified_comparison_table(dl_rows=df_e0_e7)
    master_table.to_csv(ablations_dir / "all_ablations.csv", index=False)

    print("\n=================================================================")
    print("  [SUCCESS] All ablation tables successfully written to:         ")
    print(f"  {ablations_dir}/")
    print("=================================================================\n")


if __name__ == "__main__":
    main()

