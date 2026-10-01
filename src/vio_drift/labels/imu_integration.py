"""
Strapdown inertial integration: given a starting position/velocity/
orientation (taken from ground truth at the window's first timestamp),
integrate the window's raw IMU readings forward to predict the ending
position. The gap between this prediction and the *true* ending position
(from ground truth) is the drift label — see drift_label.py.

Two conventions genuinely vary between datasets/toolchains and aren't
worth guessing blindly:
  1. gravity_sign: whether world-frame gravity should be added as
     +9.81 or -9.81 on the z-axis when recovering true acceleration
     from the accelerometer's specific-force reading.
  2. rotation_convention: whether the ground-truth quaternion rotates
     body-frame vectors into the world frame, or the reverse.

Rather than assume, `calibrate_conventions()` tries all 4 combinations
on a short, very-early slice of real data (where ground truth is dense
enough to check a one-step prediction) and picks whichever combination
gives the smallest position error. This is empirical, not guessed —
but it depends on your actual ground-truth sampling being fine enough
to test a short step; if it isn't, the function says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


# ---------- quaternion helpers (Hamilton convention, [w, x, y, z]) ----------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 0 else q


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ v rotates a vector by q (body->world
    under the 'body_to_world' convention; world->body under the reverse
    convention — the *matrix* is the same, only how we use it differs)."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_angular_rate(omega: np.ndarray, dt: float) -> np.ndarray:
    """Small-angle quaternion for a rotation of `omega` (rad/s) over `dt`."""
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = omega / np.linalg.norm(omega)
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


# ---------------------------- integration core ----------------------------

@dataclass
class IntegrationResult:
    p_end: np.ndarray   # predicted position at window end, world frame (m)
    v_end: np.ndarray   # predicted velocity at window end (m/s)
    q_end: np.ndarray   # predicted orientation at window end


def integrate_window(
    window_rows: pd.DataFrame,
    p0: np.ndarray,
    v0: np.ndarray,
    q0: np.ndarray,
    gravity_magnitude: float = 9.81,
    gravity_sign: int = -1,
    rotation_convention: str = "body_to_world",
) -> IntegrationResult:
    """
    Integrate one window's raw gyro+accel readings forward from a known
    starting state (p0, v0, q0), using simple forward-Euler strapdown
    mechanization (adequate at 200 Hz over ~1 s windows for this project;
    not claiming research-grade INS accuracy).

    gravity_sign / rotation_convention: see module docstring. Get these
    from calibrate_conventions() — don't hand-pick them.
    """
    p = p0.copy().astype(np.float64)
    v = v0.copy().astype(np.float64)
    q = q0.copy().astype(np.float64)

    g_world = np.array([0.0, 0.0, gravity_sign * gravity_magnitude])

    t = window_rows["t_ns"].to_numpy()
    gyro = window_rows[["gx", "gy", "gz"]].to_numpy()
    accel = window_rows[["ax", "ay", "az"]].to_numpy()

    for i in range(len(window_rows) - 1):
        dt = (t[i + 1] - t[i]) / 1e9
        if dt <= 0:
            continue

        R = quat_to_rotmat(q)
        if rotation_convention == "body_to_world":
            a_world = R @ accel[i] + g_world
        elif rotation_convention == "world_to_body":
            a_world = R.T @ accel[i] + g_world
        else:
            raise ValueError(f"Unknown rotation_convention: {rotation_convention}")

        # integrate position & velocity (forward Euler)
        p = p + v * dt + 0.5 * a_world * dt * dt
        v = v + a_world * dt

        # integrate orientation
        dq = quat_from_angular_rate(gyro[i], dt)
        q = quat_normalize(quat_multiply(q, dq))

    return IntegrationResult(p_end=p, v_end=v, q_end=q)


# ----------------------- empirical convention calibration -----------------

def calibrate_conventions(
    aligned_df: pd.DataFrame,
    gravity_magnitude: float = 9.81,
    probe_seconds: float = 0.25,
) -> dict:
    """
    Try all 4 (gravity_sign, rotation_convention) combinations on a short,
    early slice of real aligned data, and return whichever minimizes the
    position error against ground truth over that short probe window.

    This does NOT replace validating on a full 1 s window later
    (test_label_sanity.py does that) — it's just a fast, automatic way to
    pick the two conventions instead of guessing them by hand.
    """
    t0 = aligned_df["t_ns"].iloc[0]
    probe_end = t0 + int(probe_seconds * 1e9)
    probe = aligned_df[aligned_df["t_ns"] <= probe_end].reset_index(drop=True)

    if len(probe) < 5:
        raise ValueError(
            "Not enough rows in the probe window to calibrate conventions — "
            "check that alignment worked and the sequence isn't empty."
        )

    p0 = probe.loc[0, ["gt_px", "gt_py", "gt_pz"]].to_numpy(dtype=np.float64)
    v0 = probe.loc[0, ["gt_vx", "gt_vy", "gt_vz"]].to_numpy(dtype=np.float64)
    q0 = probe.loc[0, ["gt_qw", "gt_qx", "gt_qy", "gt_qz"]].to_numpy(dtype=np.float64)
    p_true_end = probe.loc[len(probe) - 1, ["gt_px", "gt_py", "gt_pz"]].to_numpy(dtype=np.float64)

    best = None
    results = []
    for sign, conv in product([1, -1], ["body_to_world", "world_to_body"]):
        result = integrate_window(
            probe, p0, v0, q0,
            gravity_magnitude=gravity_magnitude,
            gravity_sign=sign,
            rotation_convention=conv,
        )
        err = float(np.linalg.norm(result.p_end - p_true_end))
        results.append({"gravity_sign": sign, "rotation_convention": conv, "probe_error_m": err})
        if best is None or err < best["probe_error_m"]:
            best = {"gravity_sign": sign, "rotation_convention": conv, "probe_error_m": err}

    print(f"[calibrate_conventions] tried {len(results)} combinations over "
          f"a {probe_seconds:.2f}s probe:")
    for r in sorted(results, key=lambda r: r["probe_error_m"]):
        print(f"    sign={r['gravity_sign']:+d}  conv={r['rotation_convention']:<14s} "
              f"probe_error={r['probe_error_m']:.4f} m")
    print(f"[calibrate_conventions] chosen: sign={best['gravity_sign']:+d}, "
          f"conv={best['rotation_convention']} "
          f"(probe_error={best['probe_error_m']:.4f} m)")

    if best["probe_error_m"] > 1.0:
        print(
            "[calibrate_conventions] WARNING: even the best combination has "
            f">{best['probe_error_m']:.2f} m error over a "
            f"{probe_seconds:.2f}s probe. That's higher than expected for "
            "such a short horizon — double-check that loader.py's column "
            "mapping actually matches your CSV headers (run "
            "scripts/verify_data.py) before trusting the drift labels."
        )

    return best
