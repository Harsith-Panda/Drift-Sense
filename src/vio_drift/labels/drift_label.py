"""
Turn one Window (from windowing.py) into a drift label by comparing the
IMU-integrated predicted end position against the true (ground-truth) end
position.

Primary fields (err_mag_m, err_dx/dy/dz) always come from raw IMU.
err_*_bc fields are a documented oracle comparison: the same integration
after subtracting the dataset's ground-truth bias estimates. They do not
replace the headline target.
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
    err_mag_m_bc: float | None = None
    err_dx_bc: float | None = None
    err_dy_bc: float | None = None
    err_dz_bc: float | None = None


def compute_drift_label(
    window: Window,
    gravity_magnitude: float,
    gravity_sign: int,
    rotation_convention: str,
    subtract_bias: bool = False,
) -> DriftLabel | None:
    """
    Integrate once (raw or bias-corrected) and return magnitude + 3-axis error.

    Returns None (and prints why) if the window is missing the ground-truth
    columns needed to seed integration — this can happen for a handful of
    edge windows and is fine to drop, not fine to crash on.
    """
    rows = window.rows
    required = ["gt_px", "gt_py", "gt_pz", "gt_vx", "gt_vy", "gt_vz",
                "gt_qw", "gt_qx", "gt_qy", "gt_qz"]
    if subtract_bias:
        required = required + ["gt_bgx", "gt_bgy", "gt_bgz", "gt_bax", "gt_bay", "gt_baz"]
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
        subtract_bias=subtract_bias,
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


def compute_window_labels(
    window: Window,
    gravity_magnitude: float,
    gravity_sign: int,
    rotation_convention: str,
) -> DriftLabel | None:
    """Primary raw-IMU label plus bias-corrected comparison on the same window."""
    raw = compute_drift_label(
        window,
        gravity_magnitude=gravity_magnitude,
        gravity_sign=gravity_sign,
        rotation_convention=rotation_convention,
        subtract_bias=False,
    )
    if raw is None:
        return None

    bc = compute_drift_label(
        window,
        gravity_magnitude=gravity_magnitude,
        gravity_sign=gravity_sign,
        rotation_convention=rotation_convention,
        subtract_bias=True,
    )
    if bc is None:
        return None

    raw.err_mag_m_bc = bc.err_mag_m
    raw.err_dx_bc = bc.err_dx
    raw.err_dy_bc = bc.err_dy
    raw.err_dz_bc = bc.err_dz
    return raw
