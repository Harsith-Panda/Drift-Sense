"""
Align IMU readings with ground-truth pose by timestamp.

EuRoC's ground truth (Vicon/Leica) is NOT sampled at the same rate or
exact timestamps as the IMU (200 Hz). For every IMU row we attach the
nearest ground-truth sample (nearest by absolute time difference), and
record how far away that match was — so we can detect and report gaps
instead of silently pairing IMU readings with a stale ground-truth pose.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def align_imu_to_groundtruth(
    imu_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    max_gap_ns: int = 20_000_000,  # 20 ms default tolerance
) -> pd.DataFrame:
    """
    For each IMU row, attach the nearest ground-truth row by timestamp.

    Returns a single DataFrame with IMU columns plus ground-truth columns
    (prefixed gt_) plus a `gt_gap_ns` column showing the time distance to
    the matched ground-truth sample (can be negative or positive).

    Rows whose nearest ground-truth match is farther than `max_gap_ns` are
    dropped (they'd be paired with a stale/unreliable pose) — the count of
    dropped rows is printed so it's visible, not silent.
    """
    imu_sorted = imu_df.sort_values("t_ns").reset_index(drop=True)
    gt_sorted = gt_df.sort_values("t_ns").reset_index(drop=True)

    merged = pd.merge_asof(
        imu_sorted,
        gt_sorted.add_prefix("gt_").rename(columns={"gt_t_ns": "gt_t_ns"}),
        left_on="t_ns",
        right_on="gt_t_ns",
        direction="nearest",
    )
    merged["gt_gap_ns"] = merged["t_ns"] - merged["gt_t_ns"]

    n_before = len(merged)
    merged = merged[merged["gt_gap_ns"].abs() <= max_gap_ns].reset_index(drop=True)
    n_dropped = n_before - len(merged)
    if n_dropped > 0:
        pct = 100.0 * n_dropped / n_before
        print(
            f"[align] dropped {n_dropped}/{n_before} IMU rows "
            f"({pct:.2f}%) with no ground-truth match within "
            f"{max_gap_ns / 1e6:.1f} ms"
        )

    return merged


def save_aligned(merged: pd.DataFrame, interim_dir: Path, seq_name: str) -> Path:
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / f"aligned_{seq_name}.csv"
    merged.to_csv(out_path, index=False)
    return out_path


def report_gaps(imu_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """
    Quick diagnostic used by verify_data.py / notebook 01: reports the
    sampling rate and any large time gaps in each stream, before alignment.
    """

    def _gap_stats(t_ns: np.ndarray) -> dict:
        dt = np.diff(t_ns) / 1e9  # seconds
        return {
            "n_samples": len(t_ns),
            "duration_s": (t_ns[-1] - t_ns[0]) / 1e9,
            "median_dt_s": float(np.median(dt)) if len(dt) else float("nan"),
            "max_dt_s": float(np.max(dt)) if len(dt) else float("nan"),
            "implied_rate_hz": (1.0 / np.median(dt)) if len(dt) else float("nan"),
        }

    return {
        "imu": _gap_stats(imu_df["t_ns"].to_numpy()),
        "ground_truth": _gap_stats(gt_df["t_ns"].to_numpy()),
    }
