"""
Run with: pytest tests/test_windowing.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.windowing import make_windows, stack_raw_imu


def _toy_aligned(n_seconds=3, rate_hz=200):
    n = n_seconds * rate_hz
    t0 = 0
    dt_ns = int(1e9 / rate_hz)
    return pd.DataFrame({
        "t_ns": [t0 + i * dt_ns for i in range(n)],
        "gx": np.random.randn(n), "gy": np.random.randn(n), "gz": np.random.randn(n),
        "ax": np.random.randn(n), "ay": np.random.randn(n), "az": np.random.randn(n) + 9.81,
        "gt_px": np.zeros(n), "gt_py": np.zeros(n), "gt_pz": np.zeros(n),
    })


def test_every_window_has_expected_row_count():
    df = _toy_aligned(n_seconds=3, rate_hz=200)
    windows = make_windows(df, "toy", window_seconds=1.0, stride_seconds=0.25, expected_rows=200)
    assert len(windows) > 0
    for w in windows:
        assert len(w.rows) == 200, f"window {w.window_id} has {len(w.rows)} rows, expected 200"


def test_window_count_matches_stride_math():
    df = _toy_aligned(n_seconds=3, rate_hz=200)
    windows = make_windows(df, "toy", window_seconds=1.0, stride_seconds=0.25, expected_rows=200)
    # duration 3s, window 1s -> valid starts span 2s, stride 0.25s -> ~9 windows (0,0.25,...,2.0)
    assert 8 <= len(windows) <= 9


def test_stack_raw_imu_shape():
    df = _toy_aligned(n_seconds=3, rate_hz=200)
    windows = make_windows(df, "toy", window_seconds=1.0, stride_seconds=0.25, expected_rows=200)
    arr = stack_raw_imu(windows, expected_rows=200)
    assert arr.shape == (len(windows), 200, 6)
