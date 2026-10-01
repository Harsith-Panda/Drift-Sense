"""
Run with: pytest tests/test_alignment.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.align import align_imu_to_groundtruth


def _toy_imu(n=10, rate_hz=200):
    t0 = 1_000_000_000
    dt_ns = int(1e9 / rate_hz)
    return pd.DataFrame({
        "t_ns": [t0 + i * dt_ns for i in range(n)],
        "gx": np.zeros(n), "gy": np.zeros(n), "gz": np.zeros(n),
        "ax": np.zeros(n), "ay": np.zeros(n), "az": np.ones(n) * 9.81,
    })


def _toy_gt(n=3, rate_hz=20):
    t0 = 1_000_000_000
    dt_ns = int(1e9 / rate_hz)
    return pd.DataFrame({
        "t_ns": [t0 + i * dt_ns for i in range(n)],
        "px": np.zeros(n), "py": np.zeros(n), "pz": np.zeros(n),
        "qw": np.ones(n), "qx": np.zeros(n), "qy": np.zeros(n), "qz": np.zeros(n),
        "vx": np.zeros(n), "vy": np.zeros(n), "vz": np.zeros(n),
        "bgx": np.zeros(n), "bgy": np.zeros(n), "bgz": np.zeros(n),
        "bax": np.zeros(n), "bay": np.zeros(n), "baz": np.zeros(n),
    })


def test_every_imu_row_gets_a_match():
    imu = _toy_imu(n=10)
    gt = _toy_gt(n=3)
    merged = align_imu_to_groundtruth(imu, gt, max_gap_ns=100_000_000)
    assert len(merged) == len(imu), "every IMU row should find a ground-truth match here"


def test_gap_column_is_reasonable():
    imu = _toy_imu(n=10)
    gt = _toy_gt(n=3)
    merged = align_imu_to_groundtruth(imu, gt, max_gap_ns=100_000_000)
    assert (merged["gt_gap_ns"].abs() < 100_000_000).all()


def test_rows_dropped_beyond_max_gap():
    imu = _toy_imu(n=10)
    # ground truth far away in time -> everything should be dropped
    gt = _toy_gt(n=3)
    gt["t_ns"] = gt["t_ns"] + 10_000_000_000
    merged = align_imu_to_groundtruth(imu, gt, max_gap_ns=1_000_000)
    assert len(merged) == 0
