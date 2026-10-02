"""
Toy-data check that subtract_bias actually removes a known body-frame
accel bias. Does not need EuRoC downloaded.

Run with: pytest tests/test_bias_correction.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.labels.imu_integration import integrate_window


def _rest_window(accel_bias_x: float = 0.4, n: int = 200, rate_hz: int = 200) -> pd.DataFrame:
    dt_ns = int(1e9 / rate_hz)
    return pd.DataFrame({
        "t_ns": [i * dt_ns for i in range(n)],
        "gx": np.zeros(n), "gy": np.zeros(n), "gz": np.zeros(n),
        "ax": np.full(n, accel_bias_x), "ay": np.zeros(n), "az": np.full(n, 9.81),
        "gt_bgx": np.zeros(n), "gt_bgy": np.zeros(n), "gt_bgz": np.zeros(n),
        "gt_bax": np.full(n, accel_bias_x), "gt_bay": np.zeros(n), "gt_baz": np.zeros(n),
    })


def test_subtract_bias_cancels_known_accel_bias():
    rows = _rest_window()
    p0 = np.zeros(3)
    v0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    raw = integrate_window(rows, p0, v0, q0, subtract_bias=False)
    bc = integrate_window(rows, p0, v0, q0, subtract_bias=True)

    # 1 s of 0.4 m/s^2 uncompensated accel bias ≈ 0.5 * 0.4 * 1^2 = 0.2 m
    assert np.linalg.norm(raw.p_end) > 0.15
    assert np.linalg.norm(bc.p_end) < 1e-6
