"""
Tests for Trajectory Drift Correction and Path Reconstruction.

Verifies fundamental physical invariants:
  1. Zero prediction changes nothing (error before == error after).
  2. Perfect vector gives zero error (error after == 0.0).
  3. Continuous non-overlapping trajectory reconstruction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.correction.correct_trajectory import (
    compute_endpoint_error_comparison,
    correct_endpoints,
    rebuild_trajectory,
    run_sequence_correction,
    select_non_overlapping_windows,
)


def test_zero_prediction_changes_nothing() -> None:
    """Zero prediction vector must leave dead-reckoned coordinates and errors unchanged."""
    p_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    p_raw = np.array([[1.2, 1.9, 3.5], [4.1, 5.3, 5.8], [7.4, 7.7, 9.2]])
    zero_pred = np.zeros_like(p_raw)

    p_corr = correct_endpoints(p_raw, zero_pred)
    assert np.allclose(p_corr, p_raw)

    comp = compute_endpoint_error_comparison(p_true, p_raw, zero_pred)
    assert pytest.approx(comp["mae_before"]) == comp["mae_after"]
    assert pytest.approx(comp["rmse_before"]) == comp["rmse_after"]
    assert pytest.approx(comp["drift_reduction_pct"]) == 0.0
    assert np.allclose(comp["errors_before"], comp["errors_after"])


def test_perfect_vector_gives_zero_error() -> None:
    """Perfect prediction vector (exact error) must yield zero residual error."""
    p_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    p_raw = np.array([[1.5, 2.5, 2.5], [3.8, 5.2, 6.4], [7.1, 8.2, 8.9]])
    perfect_pred = p_raw - p_true  # exact error vector

    p_corr = correct_endpoints(p_raw, perfect_pred)
    assert np.allclose(p_corr, p_true)

    comp = compute_endpoint_error_comparison(p_true, p_raw, perfect_pred)
    assert pytest.approx(comp["mae_after"], abs=1e-12) == 0.0
    assert pytest.approx(comp["rmse_after"], abs=1e-12) == 0.0
    assert pytest.approx(comp["max_after"], abs=1e-12) == 0.0
    assert pytest.approx(comp["drift_reduction_pct"], abs=1e-5) == 100.0


def test_trajectory_reconstruction_zero_prediction() -> None:
    """In trajectory reconstruction, zero prediction must yield ate_corr == ate_raw."""
    n = 10
    chain = pd.DataFrame({
        "window_id": np.arange(n),
        "t_start_ns": np.arange(n) * 2_000_000_000,
        "t_end_ns": (np.arange(n) + 1) * 2_000_000_000,
        "err_dx": np.ones(n) * 0.1,
        "err_dy": np.ones(n) * 0.2,
        "err_dz": np.ones(n) * -0.05,
    })

    timestamps = np.unique(np.concatenate([chain["t_start_ns"].values, chain["t_end_ns"].values]))
    aligned = pd.DataFrame({
        "t_ns": timestamps,
        "gt_px": np.arange(len(timestamps)) * 1.0,
        "gt_py": np.zeros(len(timestamps)),
        "gt_pz": np.zeros(len(timestamps)),
    })

    traj = rebuild_trajectory(chain, aligned, pred_vec=np.zeros((n, 3)))
    assert np.allclose(traj["ate_raw"], traj["ate_corr"])


def test_trajectory_reconstruction_perfect_prediction() -> None:
    """In trajectory reconstruction, perfect error vector must eliminate all ATE."""
    n = 10
    chain = pd.DataFrame({
        "window_id": np.arange(n),
        "t_start_ns": np.arange(n) * 2_000_000_000,
        "t_end_ns": (np.arange(n) + 1) * 2_000_000_000,
        "err_dx": np.ones(n) * 0.5,
        "err_dy": np.ones(n) * -0.3,
        "err_dz": np.ones(n) * 0.2,
    })

    timestamps = np.unique(np.concatenate([chain["t_start_ns"].values, chain["t_end_ns"].values]))
    aligned = pd.DataFrame({
        "t_ns": timestamps,
        "gt_px": np.arange(len(timestamps)) * 0.5,
        "gt_py": np.arange(len(timestamps)) * 0.2,
        "gt_pz": np.arange(len(timestamps)) * -0.1,
    })

    perfect_vec = chain[["err_dx", "err_dy", "err_dz"]].to_numpy()
    traj = rebuild_trajectory(chain, aligned, pred_vec=perfect_vec)

    # Corrected trajectory must match ground truth exactly
    assert np.allclose(traj["ate_corr"], 0.0, atol=1e-12)
    assert np.allclose(traj["corr_x"], traj["true_x"], atol=1e-12)
    assert np.allclose(traj["corr_y"], traj["true_y"], atol=1e-12)
    assert np.allclose(traj["corr_z"], traj["true_z"], atol=1e-12)


def test_real_sequence_correction_reduction() -> None:
    """Running trajectory correction on MH_01_easy must achieve >80% drift reduction."""
    comp, traj_df = run_sequence_correction("MH_01_easy")
    assert comp["mae_before"] > 1.0  # ~1.2 m raw dead reckoning
    assert comp["mae_after"] < 0.15  # reduced by physics
    assert comp["drift_reduction_pct"] > 80.0
    assert comp["ate_reduction_pct"] > 80.0
    assert len(traj_df) > 50

