"""
Run this FIRST, before anything else.

Checks that both required CSVs exist for every sequence in
configs/config.yaml, prints their raw headers, and confirms the column
count matches what loader.py expects. If it doesn't match, this script
tells you exactly what to fix — it does not try to guess and proceed.

Usage:
    python scripts/verify_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.loader import (
    SequencePaths, EXPECTED_IMU_NCOLS, EXPECTED_GT_NCOLS,
)


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())

    raw_dir = Path(config["data"]["raw_dir"])
    sequences = config["data"]["sequences"]

    all_ok = True

    for seq_name in sequences:
        print(f"\n=== {seq_name} ===")
        seq_dir = raw_dir / seq_name
        paths = SequencePaths(seq_dir)

        for label, path, expected_ncols in [
            ("imu0/data.csv", paths.imu_csv, EXPECTED_IMU_NCOLS),
            ("state_groundtruth_estimate0/data.csv", paths.gt_csv, EXPECTED_GT_NCOLS),
        ]:
            if not path.exists():
                print(f"  [MISSING] {label}: expected at {path}")
                all_ok = False
                continue

            with open(path) as f:
                header = f.readline().strip()
            ncols = len(header.split(","))

            status = "OK" if ncols == expected_ncols else "MISMATCH"
            if status == "MISMATCH":
                all_ok = False
            print(f"  [{status}] {label}")
            print(f"      path: {path}")
            print(f"      columns found: {ncols} (expected {expected_ncols})")
            print(f"      header: {header}")

    print("\n" + "=" * 60)
    if all_ok:
        print("All sequences look correct. Safe to run scripts/make_windows.py.")
    else:
        print("Fix the issues above before running make_windows.py.")
        print("If the column COUNT is right but names look different, that's")
        print("fine — loader.py maps by position, not exact text.")
        print("If the column COUNT is wrong, update EXPECTED_IMU_NCOLS / ")
        print("EXPECTED_GT_NCOLS and IMU_COLS / GT_COLS in "
              "src/vio_drift/data/loader.py to match, then re-run this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
