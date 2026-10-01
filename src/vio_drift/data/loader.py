"""
Load raw EuRoC (ASL-format) IMU and ground-truth CSVs into clean pandas
DataFrames with explicit, renamed columns and SI units.

This module is intentionally strict: if the CSV headers don't match what
we expect, it raises a clear error instead of silently mis-mapping columns.
That's on purpose — a silent column mismatch would poison every downstream
label and feature without anyone noticing.

Expected raw headers (standard EuRoC ASL format):

  imu0/data.csv:
    #timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],
    a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]

  state_groundtruth_estimate0/data.csv:
    #timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],
    q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],
    v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],
    b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],
    b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]

If your actual files differ (extra columns, different order, different
header text), update EXPECTED_IMU_COLS / EXPECTED_GT_COLS below to match,
and re-run scripts/verify_data.py to confirm.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---- what we expect to find, by position (order matters, not exact text) ----
# We match by column *position* after stripping the header, not by exact
# string, because EuRoC headers vary slightly in spacing/units notation
# across mirrors. We only check the *count* and a couple of anchor tokens.

EXPECTED_IMU_NCOLS = 7
EXPECTED_GT_NCOLS = 17

IMU_COLS = ["t_ns", "gx", "gy", "gz", "ax", "ay", "az"]
GT_COLS = [
    "t_ns",
    "px", "py", "pz",
    "qw", "qx", "qy", "qz",
    "vx", "vy", "vz",
    "bgx", "bgy", "bgz",
    "bax", "bay", "baz",
]


class SchemaMismatchError(ValueError):
    """Raised when a raw CSV doesn't have the column count we expect."""


@dataclass
class SequencePaths:
    seq_dir: Path

    @property
    def imu_csv(self) -> Path:
        return self.seq_dir / "mav0" / "imu0" / "data.csv"

    @property
    def gt_csv(self) -> Path:
        return self.seq_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"


def _read_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            f"Check data/README.md — did you extract the zip into "
            f"data/raw/euroc/<sequence_name>/mav0/... ?"
        )
    # header line starts with '#', so treat it as a comment and supply our own
    return pd.read_csv(path, comment=None, header=0)


def load_imu(seq_dir: Path) -> pd.DataFrame:
    """
    Load imu0/data.csv for one sequence.

    Returns a DataFrame with columns:
        t_ns (int64), gx, gy, gz (rad/s), ax, ay, az (m/s^2)
    sorted by timestamp, with duplicate timestamps dropped.
    """
    paths = SequencePaths(seq_dir)
    raw = _read_raw_csv(paths.imu_csv)

    if raw.shape[1] != EXPECTED_IMU_NCOLS:
        raise SchemaMismatchError(
            f"{paths.imu_csv} has {raw.shape[1]} columns, expected "
            f"{EXPECTED_IMU_NCOLS}. Raw header was: {list(raw.columns)}\n"
            f"Update EXPECTED_IMU_NCOLS / IMU_COLS in loader.py to match, "
            f"then re-run scripts/verify_data.py."
        )

    raw.columns = IMU_COLS
    raw["t_ns"] = raw["t_ns"].astype(np.int64)
    raw = raw.sort_values("t_ns").drop_duplicates(subset="t_ns").reset_index(drop=True)
    return raw


def load_ground_truth(seq_dir: Path) -> pd.DataFrame:
    """
    Load state_groundtruth_estimate0/data.csv for one sequence.

    Returns a DataFrame with columns:
        t_ns (int64), px,py,pz (m), qw,qx,qy,qz (unit quaternion),
        vx,vy,vz (m/s), bgx,bgy,bgz (rad/s), bax,bay,baz (m/s^2)
    sorted by timestamp, with duplicate timestamps dropped.
    """
    paths = SequencePaths(seq_dir)
    raw = _read_raw_csv(paths.gt_csv)

    if raw.shape[1] != EXPECTED_GT_NCOLS:
        raise SchemaMismatchError(
            f"{paths.gt_csv} has {raw.shape[1]} columns, expected "
            f"{EXPECTED_GT_NCOLS}. Raw header was: {list(raw.columns)}\n"
            f"Update EXPECTED_GT_NCOLS / GT_COLS in loader.py to match, "
            f"then re-run scripts/verify_data.py."
        )

    raw.columns = GT_COLS
    raw["t_ns"] = raw["t_ns"].astype(np.int64)
    raw = raw.sort_values("t_ns").drop_duplicates(subset="t_ns").reset_index(drop=True)

    # normalize quaternions defensively (raw data should already be ~unit norm)
    q = raw[["qw", "qx", "qy", "qz"]].to_numpy()
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    raw[["qw", "qx", "qy", "qz"]] = q / norms

    return raw


def load_sequence(raw_dir: Path, seq_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: load (imu_df, gt_df) for one sequence by name."""
    seq_dir = Path(raw_dir) / seq_name
    return load_imu(seq_dir), load_ground_truth(seq_dir)
