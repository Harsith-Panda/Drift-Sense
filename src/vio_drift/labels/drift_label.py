"""
Turn one Window (from windowing.py) into a drift label by comparing the
IMU-integrated predicted end position against the true (ground-truth) end
position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vio_drift.data.windowing import Window
from vio_drift.labels.imu_integration import integrate_window


@dataclass
class DriftLabel:
    seq_name: str
    window_id: int
    t_start_ns: int
    t_end_ns: int
    err_mag_m: float
    err_dx: float
    err_dy: float
    err_dz: float


def compute_drift_label(
    window: Window,
    gravity_magnitude: float,
    gravity_sign: int,
    rotation_convention: str,
) -> DriftLabel | None:
    """
    Returns None (and prints why) if the window's first row is missing the
    ground-truth columns needed to seed integration — this can happen for
    a handful of edge windows and is fine to drop, not fine to crash on.
    """
    rows = window.rows
    required = ["gt_px", "gt_py", "gt_pz", "gt_vx", "gt_vy", "gt_vz",
                "gt_qw", "gt_qx", "gt_qy", "gt_qz"]
    if rows[required].isna().any().any():
        print(f"[drift_label] {window.seq_name} window {window.window_id}: "
              f"missing ground-truth values, skipping")
        return None

    p0 = rows.loc[0, ["gt_px", "gt_py", "gt_pz"]].to_numpy(dtype=np.float64)
    v0 = rows.loc[0, ["gt_vx", "gt_vy", "gt_vz"]].to_numpy(dtype=np.float64)
    q0 = rows.loc[0, ["gt_qw", "gt_qx", "gt_qy", "gt_qz"]].to_numpy(dtype=np.float64)

    last = len(rows) - 1
    p_true_end = rows.loc[last, ["gt_px", "gt_py", "gt_pz"]].to_numpy(dtype=np.float64)

    result = integrate_window(
        rows, p0, v0, q0,
        gravity_magnitude=gravity_magnitude,
        gravity_sign=gravity_sign,
        rotation_convention=rotation_convention,
    )

    err_vec = result.p_end - p_true_end
    return DriftLabel(
        seq_name=window.seq_name,
        window_id=window.window_id,
        t_start_ns=window.t_start_ns,
        t_end_ns=window.t_end_ns,
        err_mag_m=float(np.linalg.norm(err_vec)),
        err_dx=float(err_vec[0]),
        err_dy=float(err_vec[1]),
        err_dz=float(err_vec[2]),
    )
