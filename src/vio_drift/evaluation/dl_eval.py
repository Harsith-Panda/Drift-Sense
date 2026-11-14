"""
Evaluation harness for Deep Learning and Ablation models (Phase 2).

Mission: make every number comparable and trustworthy, and show the practical payoff.
Reproduces Phase 1 metrics (MAE, RMSE, R², bias) across sequences and segments
(first-20% take-off vs remaining 80%), and aggregates mean and spread over random seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score_metrics(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> dict[str, float]:
    """
    Compute core Phase 1 regression metrics.
    
    mae   mean absolute error (metres)
    rmse  root mean squared error (metres)
    r2    coefficient of determination (1 - SSE / SST)
    bias  mean error (pred - true) (metres)
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if len(y_true_arr) == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "bias": 0.0}

    mae_val = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse_val = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    
    var = float(np.var(y_true_arr))
    if var < 1e-12:
        r2_val = 0.0
    else:
        r2_val = float(r2_score(y_true_arr, y_pred_arr))

    bias_val = float(np.mean(y_pred_arr - y_true_arr))

    return {
        "mae": mae_val,
        "rmse": rmse_val,
        "r2": r2_val,
        "bias": bias_val,
    }


def score_segments(
    df: pd.DataFrame,
    target_col: str = "y_true",
    pred_col: str = "y_pred",
    time_col: str = "t_start_ns",
    cut_ratio: float = 0.2,
) -> dict[str, dict[str, float]]:
    """
    Score predictions by flight segment: first-20% (take-off) vs remaining 80%.
    """
    if time_col in df.columns:
        sorted_df = df.sort_values(time_col)
    else:
        sorted_df = df

    cut = int(cut_ratio * len(sorted_df))
    f20 = sorted_df.iloc[:cut]
    r80 = sorted_df.iloc[cut:]

    return {
        "first_20": score_metrics(f20[target_col], f20[pred_col]),
        "remaining_80": score_metrics(r80[target_col], r80[pred_col]),
        "overall": score_metrics(sorted_df[target_col], sorted_df[pred_col]),
    }


def score_prediction_dataframe(
    df: pd.DataFrame,
    target_col: str = "y_true",
    pred_col: str = "y_pred",
    seq_col: str = "seq_id",
    time_col: str = "t_start_ns",
) -> dict[str, Any]:
    """
    Score a prediction table across sequences, fold mean, and segments.
    """
    df = df.copy()
    
    col_map = {
        "y": "y_true",
        "y_hat": "y_pred",
        "actual": "y_true",
        "predicted": "y_pred",
        "test_seq": "seq_id",
        "sequence": "seq_id",
    }
    for old_c, new_c in col_map.items():
        if old_c in df.columns and new_c not in df.columns:
            df[new_c] = df[old_c]

    if target_col not in df.columns or pred_col not in df.columns:
        raise ValueError(
            f"Prediction dataframe missing target '{target_col}' or pred '{pred_col}'. "
            f"Available: {df.columns.tolist()}"
        )

    results: dict[str, Any] = {"by_sequence": {}, "segments": {}}

    if seq_col in df.columns:
        seqs = df[seq_col].unique().tolist()
        per_seq_metrics = []
        for seq in seqs:
            seq_df = df[df[seq_col] == seq]
            m = score_metrics(seq_df[target_col], seq_df[pred_col])
            results["by_sequence"][seq] = m
            per_seq_metrics.append(m)

            results["segments"][seq] = score_segments(
                seq_df, target_col=target_col, pred_col=pred_col, time_col=time_col
            )

        results["mean_fold"] = {
            k: float(np.mean([m[k] for m in per_seq_metrics]))
            for k in ["mae", "rmse", "r2", "bias"]
        }
    else:
        results["overall"] = score_metrics(df[target_col], df[pred_col])
        results["segments"]["overall"] = score_segments(
            df, target_col=target_col, pred_col=pred_col, time_col=time_col
        )

    return results


def score_prediction_file(
    path: Path | str,
    target_col: str = "y_true",
    pred_col: str = "y_pred",
) -> dict[str, Any]:
    """Score a CSV prediction file."""
    df = pd.read_csv(path)
    return score_prediction_dataframe(df, target_col=target_col, pred_col=pred_col)


def aggregate_seed_metrics(seed_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """
    Given a list of score dictionaries from multiple seeds, compute mean and spread (std).
    """
    if not seed_results:
        return {}

    metric_keys = ["mae", "rmse", "r2", "bias"]
    aggregated: dict[str, dict[str, float]] = {}

    for k in metric_keys:
        vals = []
        for sr in seed_results:
            if "mean_fold" in sr and k in sr["mean_fold"]:
                vals.append(sr["mean_fold"][k])
            elif "overall" in sr and k in sr["overall"]:
                vals.append(sr["overall"][k])
            elif k in sr:
                vals.append(sr[k])
        if vals:
            aggregated[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1 if len(vals) > 1 else 0)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    return aggregated


def reproduce_phase1_benchmarks(processed_dir: Path | str | None = None) -> pd.DataFrame:
    """
    Reproduces the Phase 1 benchmark numbers on MH_01_easy and MH_02_easy.
    """
    if processed_dir is None:
        root = Path(__file__).resolve().parents[3]
        processed_dir = root / "data" / "processed"
    else:
        processed_dir = Path(processed_dir)

    splits_path = processed_dir / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits file at {splits_path}")

    from vio_drift.data.splits import load_splits
    from vio_drift.features.build_features import feature_names
    from vio_drift.models.ml.data import fold_frames, load_dataset
    from vio_drift.models.ml.models import build_lasso, build_mean, build_ols, build_ridge

    df = load_dataset(processed_dir, ["MH_01_easy", "MH_02_easy"], target="err_mag_m")
    features = feature_names()
    folds = load_splits(splits_path)

    tuned_alphas = {
        "ridge": {"leave_MH_01_easy": 316.22776601683796, "leave_MH_02_easy": 31.622776601683793},
        "lasso": {"leave_MH_01_easy": 0.002154434690031882, "leave_MH_02_easy": 0.00046415888336127773},
    }

    rows = []

    models_to_run = ["mean", "ols", "ridge", "lasso"]
    for m_name in models_to_run:
        for fold in folds:
            train, test = fold_frames(df, fold)
            if m_name == "mean":
                model = build_mean()
            elif m_name == "ols":
                model = build_ols()
            elif m_name == "ridge":
                model = build_ridge(tuned_alphas["ridge"][fold.fold_id])
            elif m_name == "lasso":
                model = build_lasso(tuned_alphas["lasso"][fold.fold_id])
            else:
                raise ValueError(m_name)

            model.fit(train[features], train["err_mag_m"])
            preds = model.predict(test[features])
            m = score_metrics(test["err_mag_m"], preds)
            rows.append({
                "model": m_name,
                "test_seq": fold.test[0],
                "fold_id": fold.fold_id,
                **m,
            })

    for fold in folds:
        test_seq = fold.test[0]
        phys_path = processed_dir / f"physics_{fold.fold_id}.csv"
        win_path = processed_dir / f"windows_{test_seq}.csv"
        if phys_path.exists() and win_path.exists():
            p_df = pd.read_csv(phys_path)
            w_df = pd.read_csv(win_path)
            merged = pd.merge(w_df, p_df, on=["seq_id", "window_id"])
            m = score_metrics(merged["err_mag_m"], merged["phys_mag"])
            rows.append({
                "model": "physics_only",
                "test_seq": test_seq,
                "fold_id": fold.fold_id,
                **m,
            })

    results_df = pd.DataFrame(rows)

    mean_rows = []
    for model_name, g in results_df.groupby("model", sort=False):
        mean_rows.append({
            "model": model_name,
            "test_seq": "mean",
            "fold_id": "mean",
            "mae": g["mae"].mean(),
            "rmse": g["rmse"].mean(),
            "r2": g["r2"].mean(),
            "bias": g["bias"].mean(),
        })

    combined = pd.concat([results_df, pd.DataFrame(mean_rows)], ignore_index=True)
    return combined


def build_unified_comparison_table(
    dl_rows: list[dict[str, Any]] | pd.DataFrame | None = None,
    processed_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Build master comparison table showing Phase 1 benchmarks and DL models.
    """
    p1 = reproduce_phase1_benchmarks(processed_dir)
    p1_summary = p1[p1["test_seq"] == "mean"].copy()
    
    name_map = {
        "mean": "Mean baseline",
        "ols": "OLS",
        "ridge": "Ridge (L2)",
        "lasso": "Lasso (L1)",
        "physics_only": "Physics-only",
    }
    
    table_rows = []
    for _, r in p1_summary.iterrows():
        display_name = name_map.get(r["model"], r["model"])
        table_rows.append({
            "Family": "Phase 1 (ML / Physics)",
            "Model": display_name,
            "MAE (cm)": f"{r['mae'] * 100:.2f}",
            "RMSE (cm)": f"{r['rmse'] * 100:.2f}",
            "R²": f"{r['r2']:.2f}",
            "Bias (cm)": f"{r['bias'] * 100:.2f}",
            "Seed Spread (std)": "N/A (deterministic)",
        })

    if dl_rows is not None:
        if isinstance(dl_rows, list):
            dl_df = pd.DataFrame(dl_rows)
        else:
            dl_df = dl_rows.copy()

        for _, r in dl_df.iterrows():
            std_str = f"±{r['mae_std_cm']:.2f} cm" if "mae_std_cm" in r and pd.notna(r["mae_std_cm"]) else "N/A"
            m_name = r.get("name", r.get("model", "DL Model"))
            table_rows.append({
                "Family": r.get("family", "Phase 2 (DL)"),
                "Model": m_name,
                "MAE (cm)": f"{r['mae_mean_cm']:.2f}" if "mae_mean_cm" in r else f"{r['mae'] * 100:.2f}",
                "RMSE (cm)": f"{r['rmse_mean_cm']:.2f}" if "rmse_mean_cm" in r else f"{r['rmse'] * 100:.2f}",
                "R²": f"{r['r2_mean']:.2f}" if "r2_mean" in r else f"{r['r2']:.2f}",
                "Bias (cm)": f"{r['bias_mean_cm']:.2f}" if "bias_mean_cm" in r else "0.00",
                "Seed Spread (std)": std_str,
            })

    return pd.DataFrame(table_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction files with Phase 1 metrics and segment analysis.")
    parser.add_argument("--file", type=str, help="Path to predictions CSV file to score")
    parser.add_argument("--phase1-benchmark", action="store_true", help="Print reproduced Phase 1 benchmark table")
    args = parser.parse_args()

    if args.phase1_benchmark:
        print("\n--- Reproducing Phase 1 Benchmarks ---")
        benchmarks = reproduce_phase1_benchmarks()
        print(benchmarks.to_string(index=False, float_format="%.4f"))

    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"File not found: {p}")
            return
        print(f"\n--- Scoring {p.name} ---")
        scores = score_prediction_file(p)
        print(yaml.dump(scores, sort_keys=False))


if __name__ == "__main__":
    main()

