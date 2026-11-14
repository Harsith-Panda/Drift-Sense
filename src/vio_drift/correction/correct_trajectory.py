"""
Trajectory Drift Correction and Path Reconstruction (Phase 2).

Mission: Subtract the predicted 3-axis error from each window's dead-reckoned
end point; evaluate end-point error before vs after; rebuild a continuous path
from non-overlapping windows for comparison plots and ATE evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def correct_endpoints(p_end_raw: np.ndarray, pred_drift_vec: np.ndarray) -> np.ndarray:
    """
    Subtract predicted 3-axis drift vector from dead-reckoned end points.
    
    p_end_corr = p_end_raw - pred_drift_vec
    """
    p_raw = np.asarray(p_end_raw, dtype=np.float64)
    p_pred = np.asarray(pred_drift_vec, dtype=np.float64)
    if p_raw.shape != p_pred.shape:
        raise ValueError(f"Shape mismatch: p_end_raw {p_raw.shape} vs pred_drift_vec {p_pred.shape}")
    return p_raw - p_pred


def compute_endpoint_error_comparison(
    p_end_true: np.ndarray,
    p_end_raw: np.ndarray,
    pred_drift_vec: np.ndarray,
) -> dict[str, Any]:
    """
    Compare end-point position error before vs after drift subtraction.
    
    Returns:
      mae_before, mae_after: Mean absolute Euclidean error (m)
      rmse_before, rmse_after: Root mean squared Euclidean error (m)
      median_before, median_after: Median error (m)
      max_before, max_after: Maximum error (m)
      drift_reduction_pct: Percentage reduction in MAE
      errors_before, errors_after: Per-window Euclidean error arrays
    """
    p_true = np.asarray(p_end_true, dtype=np.float64)
    p_raw = np.asarray(p_end_raw, dtype=np.float64)
    p_corr = correct_endpoints(p_raw, pred_drift_vec)

    err_vec_before = p_raw - p_true
    err_vec_after = p_corr - p_true

    errors_before = np.linalg.norm(err_vec_before, axis=-1)
    errors_after = np.linalg.norm(err_vec_after, axis=-1)

    mae_before = float(np.mean(errors_before))
    mae_after = float(np.mean(errors_after))
    rmse_before = float(np.sqrt(np.mean(errors_before ** 2)))
    rmse_after = float(np.sqrt(np.mean(errors_after ** 2)))
    med_before = float(np.median(errors_before))
    med_after = float(np.median(errors_after))
    max_before = float(np.max(errors_before))
    max_after = float(np.max(errors_after))

    reduction = float((1.0 - (mae_after / mae_before)) * 100.0) if mae_before > 0 else 0.0

    return {
        "mae_before": mae_before,
        "mae_after": mae_after,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "median_before": med_before,
        "median_after": med_after,
        "max_before": max_before,
        "max_after": max_after,
        "drift_reduction_pct": reduction,
        "errors_before": errors_before,
        "errors_after": errors_after,
        "p_end_corr": p_corr,
    }


def select_non_overlapping_windows(windows_df: pd.DataFrame, step: int = 8) -> pd.DataFrame:
    """
    Select non-overlapping contiguous windows along the flight.
    With window_seconds=2.0 and stride_seconds=0.25, every 8th window
    (stride * 8 = 2.0 s) shares t_start with the previous window's t_end.
    """
    sorted_w = windows_df.sort_values("t_start_ns").reset_index(drop=True)
    chain = sorted_w.iloc[::step].copy().reset_index(drop=True)
    return chain


def rebuild_trajectory(
    windows_chain: pd.DataFrame,
    aligned_df: pd.DataFrame,
    pred_vec: np.ndarray | None = None,
    p0: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Rebuild a continuous path from non-overlapping windows.
    
    Starting from p0 (initial true position):
      For window k:
        True segment displacement: Δp_true = p_true(t_end) - p_true(t_start)
        Raw dead-reckoned displacement: Δp_raw = Δp_true + err_vec
        Corrected displacement: Δp_corr = Δp_raw - pred_vec
      Cumulative position is integrated forward step-by-step.
    """
    chain = windows_chain.reset_index(drop=True)
    n_windows = len(chain)

    if pred_vec is None:
        pred_vec = np.zeros((n_windows, 3), dtype=np.float64)
    else:
        pred_vec = np.asarray(pred_vec, dtype=np.float64)
        if len(pred_vec) != n_windows:
            raise ValueError(f"pred_vec length {len(pred_vec)} does not match chain length {n_windows}")

    gt_map = aligned_df.set_index("t_ns")[["gt_px", "gt_py", "gt_pz"]]

    if p0 is None:
        t0 = chain.loc[0, "t_start_ns"]
        if t0 in gt_map.index:
            p0 = gt_map.loc[t0].to_numpy(dtype=np.float64)
        else:
            p0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    else:
        p0 = np.asarray(p0, dtype=np.float64)

    true_path = [p0.copy()]
    raw_path = [p0.copy()]
    corr_path = [p0.copy()]

    rows_meta = [{
        "step": 0,
        "window_id": -1,
        "t_start_ns": chain.loc[0, "t_start_ns"],
        "t_end_ns": chain.loc[0, "t_start_ns"],
        "true_x": p0[0], "true_y": p0[1], "true_z": p0[2],
        "raw_x": p0[0], "raw_y": p0[1], "raw_z": p0[2],
        "corr_x": p0[0], "corr_y": p0[1], "corr_z": p0[2],
        "ate_raw": 0.0,
        "ate_corr": 0.0,
    }]

    for i in range(n_windows):
        row = chain.iloc[i]
        t_start = row["t_start_ns"]
        t_end = row["t_end_ns"]

        if t_end in gt_map.index and t_start in gt_map.index:
            pt_start = gt_map.loc[t_start].to_numpy(dtype=np.float64)
            pt_end = gt_map.loc[t_end].to_numpy(dtype=np.float64)
            d_true = pt_end - pt_start
        else:
            pt_end = true_path[-1]
            d_true = np.zeros(3)

        err = np.array([row["err_dx"], row["err_dy"], row["err_dz"]], dtype=np.float64)
        pred = pred_vec[i]

        d_raw = d_true + err
        d_corr = d_raw - pred

        next_true = pt_end if (t_end in gt_map.index) else (true_path[-1] + d_true)
        next_raw = raw_path[-1] + d_raw
        next_corr = corr_path[-1] + d_corr

        true_path.append(next_true)
        raw_path.append(next_raw)
        corr_path.append(next_corr)

        ate_raw = float(np.linalg.norm(next_raw - next_true))
        ate_corr = float(np.linalg.norm(next_corr - next_true))

        rows_meta.append({
            "step": i + 1,
            "window_id": int(row["window_id"]),
            "t_start_ns": t_start,
            "t_end_ns": t_end,
            "true_x": float(next_true[0]),
            "true_y": float(next_true[1]),
            "true_z": float(next_true[2]),
            "raw_x": float(next_raw[0]),
            "raw_y": float(next_raw[1]),
            "raw_z": float(next_raw[2]),
            "corr_x": float(next_corr[0]),
            "corr_y": float(next_corr[1]),
            "corr_z": float(next_corr[2]),
            "ate_raw": ate_raw,
            "ate_corr": ate_corr,
        })

    return pd.DataFrame(rows_meta)


def run_sequence_correction(
    seq_name: str,
    pred_vec: np.ndarray | None = None,
    processed_dir: Path | str | None = None,
    interim_dir: Path | str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Run full trajectory correction analysis on a flight sequence.
    If pred_vec is None, uses physics_leave_<seq>.csv as reference vector.
    """
    root = Path(__file__).resolve().parents[3]
    if processed_dir is None:
        processed_dir = root / "data" / "processed"
    else:
        processed_dir = Path(processed_dir)

    if interim_dir is None:
        interim_dir = root / "data" / "interim"
    else:
        interim_dir = Path(interim_dir)

    w_df = pd.read_csv(processed_dir / f"windows_{seq_name}.csv")
    a_df = pd.read_csv(interim_dir / f"aligned_{seq_name}.csv")

    chain = select_non_overlapping_windows(w_df, step=8)

    if pred_vec is None:
        phys_path = processed_dir / f"physics_leave_{seq_name}.csv"
        if phys_path.exists():
            p_df = pd.read_csv(phys_path)
            merged = pd.merge(chain, p_df, on=["seq_id", "window_id"])
            pred_vec = merged[["phys_dx", "phys_dy", "phys_dz"]].to_numpy()
        else:
            pred_vec = np.zeros((len(chain), 3))

    gt_map = a_df.set_index("t_ns")[["gt_px", "gt_py", "gt_pz"]]
    p_end_true = []
    p_end_raw = []
    for _, r in chain.iterrows():
        pt = gt_map.loc[r["t_end_ns"]].to_numpy()
        p_end_true.append(pt)
        p_end_raw.append(pt + np.array([r["err_dx"], r["err_dy"], r["err_dz"]]))

    p_end_true = np.array(p_end_true)
    p_end_raw = np.array(p_end_raw)

    comp = compute_endpoint_error_comparison(p_end_true, p_end_raw, pred_vec)
    traj_df = rebuild_trajectory(chain, a_df, pred_vec=pred_vec)

    comp["mean_ate_raw"] = float(traj_df["ate_raw"].mean())
    comp["final_ate_raw"] = float(traj_df["ate_raw"].iloc[-1])
    comp["mean_ate_corr"] = float(traj_df["ate_corr"].mean())
    comp["final_ate_corr"] = float(traj_df["ate_corr"].iloc[-1])
    comp["ate_reduction_pct"] = float((1.0 - (comp["mean_ate_corr"] / comp["mean_ate_raw"])) * 100.0)

    return comp, traj_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trajectory drift correction on a sequence.")
    parser.add_argument("--seq", type=str, default="MH_01_easy", help="Sequence name (e.g. MH_01_easy)")
    args = parser.parse_args()

    print(f"\n--- Trajectory Correction: {args.seq} ---")
    comp, traj_df = run_sequence_correction(args.seq)
    print(f"Windows evaluated (non-overlapping): {len(traj_df) - 1}")
    print(f"Window Endpoint Error Before: {comp['mae_before']*100:.2f} cm (RMSE: {comp['rmse_before']*100:.2f} cm)")
    print(f"Window Endpoint Error After:  {comp['mae_after']*100:.2f} cm (RMSE: {comp['rmse_after']*100:.2f} cm)")
    print(f"Window Error Reduction:       {comp['drift_reduction_pct']:.1f}%")
    print(f"Cumulative Mean ATE (Raw):    {comp['mean_ate_raw']:.2f} m")
    print(f"Cumulative Mean ATE (Corr):   {comp['mean_ate_corr']:.2f} m")
    print(f"Cumulative Final Drift (Raw): {comp['final_ate_raw']:.2f} m")
    print(f"Cumulative Final Drift (Corr):{comp['final_ate_corr']:.2f} m")
    print(f"Total Trajectory ATE Reduction:{comp['ate_reduction_pct']:.1f}%")


if __name__ == "__main__":
    main()

