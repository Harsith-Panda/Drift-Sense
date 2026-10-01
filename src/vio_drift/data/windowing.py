"""
Cut an aligned sequence into fixed-length sliding windows.

Each window is a contiguous slice of IMU rows (with their matched
ground-truth rows attached) of a fixed *time* length (default 1.0 s),
stepping forward by a fixed stride (default 0.25 s). We window by time,
not by a fixed row count, because IMU sampling isn't perfectly uniform —
this keeps windows honest about how much real time they actually cover.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Window:
    seq_name: str
    window_id: int
    t_start_ns: int
    t_end_ns: int
    rows: pd.DataFrame  # the aligned rows inside [t_start_ns, t_end_ns)


def make_windows(
    aligned_df: pd.DataFrame,
    seq_name: str,
    window_seconds: float = 1.0,
    stride_seconds: float = 0.25,
    expected_rows: int | None = 200,
    row_count_tolerance: int = 10,
) -> list[Window]:
    """
    Slide a window of `window_seconds` across `aligned_df`, stepping by
    `stride_seconds`, starting at the first timestamp.

    If `expected_rows` is given, windows whose row count differs from it
    by more than `row_count_tolerance` are skipped (they likely sit over
    a data gap) — skipped windows are counted and reported, not silent.
    """
    t = aligned_df["t_ns"].to_numpy()
    t0, t1 = t[0], t[-1]

    window_ns = int(window_seconds * 1e9)
    stride_ns = int(stride_seconds * 1e9)

    windows: list[Window] = []
    n_skipped = 0
    window_id = 0

    start = t0
    while start + window_ns <= t1:
        end = start + window_ns
        mask = (t >= start) & (t < end)
        rows = aligned_df.loc[mask]

        if expected_rows is not None and abs(len(rows) - expected_rows) > row_count_tolerance:
            n_skipped += 1
        else:
            windows.append(
                Window(
                    seq_name=seq_name,
                    window_id=window_id,
                    t_start_ns=int(start),
                    t_end_ns=int(end),
                    rows=rows.reset_index(drop=True),
                )
            )
            window_id += 1

        start += stride_ns

    if n_skipped > 0:
        print(
            f"[windowing] {seq_name}: skipped {n_skipped} windows with an "
            f"unexpected row count (likely over a data gap)"
        )
    print(f"[windowing] {seq_name}: produced {len(windows)} windows")

    return windows


def stack_raw_imu(windows: list[Window], expected_rows: int = 200) -> np.ndarray:
    """
    Stack windows' [gx,gy,gz,ax,ay,az] into one array of shape
    (n_windows, expected_rows, 6), padding/truncating each window to
    `expected_rows` if it's off by a row or two (common at stream edges).
    """
    cols = ["gx", "gy", "gz", "ax", "ay", "az"]
    out = np.zeros((len(windows), expected_rows, 6), dtype=np.float64)

    for i, w in enumerate(windows):
        arr = w.rows[cols].to_numpy()
        n = min(len(arr), expected_rows)
        out[i, :n, :] = arr[:n]
        if n < expected_rows:
            out[i, n:, :] = arr[-1]  # hold last value to pad

    return out
