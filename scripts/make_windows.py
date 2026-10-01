"""
End-to-end Part A pipeline for one or all sequences:
  load -> align -> window -> calibrate conventions (once) -> label -> save

Produces, per sequence, in data/processed/:
  windows_<seq>.csv       (seq_id, window_id, t_start_ns, t_end_ns,
                            err_mag_m, err_dx, err_dy, err_dz)
  windows_raw_<seq>.npy   (N_windows, 200, 6) — gx,gy,gz,ax,ay,az

Usage:
    python scripts/make_windows.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.loader import load_sequence
from vio_drift.data.align import align_imu_to_groundtruth, save_aligned, report_gaps
from vio_drift.data.windowing import make_windows, stack_raw_imu
from vio_drift.labels.imu_integration import calibrate_conventions
from vio_drift.labels.drift_label import compute_drift_label


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())

    raw_dir = root / config["data"]["raw_dir"]
    interim_dir = root / config["data"]["interim_dir"]
    processed_dir = root / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    sequences = config["data"]["sequences"]
    window_seconds = config["windowing"]["window_seconds"]
    stride_seconds = config["windowing"]["stride_seconds"]
    expected_rows = int(
        config["windowing"]["window_seconds"] * config["windowing"]["imu_rate_hz"]
    )

    gravity_magnitude = config["integration"]["gravity_magnitude"]
    gravity_sign = config["integration"]["gravity_sign"]
    rotation_convention = config["integration"]["rotation_convention"]

    calibrated_here = False

    for seq_name in sequences:
        print(f"\n{'=' * 60}\n{seq_name}\n{'=' * 60}")

        imu_df, gt_df = load_sequence(raw_dir, seq_name)
        print(f"[load] {seq_name}: {len(imu_df)} IMU rows, {len(gt_df)} ground-truth rows")
        print(f"[load] gap report: {report_gaps(imu_df, gt_df)}")

        aligned = align_imu_to_groundtruth(imu_df, gt_df)
        save_aligned(aligned, interim_dir, seq_name)

        # Calibrate gravity_sign / rotation_convention ONCE, on the first
        # sequence, if not already pinned in config.yaml. Reuse for the rest
        # so every sequence uses the same physical convention.
        if gravity_sign is None or rotation_convention is None:
            if not calibrated_here:
                best = calibrate_conventions(aligned, gravity_magnitude=gravity_magnitude)
                gravity_sign = best["gravity_sign"]
                rotation_convention = best["rotation_convention"]
                calibrated_here = True
                print(
                    f"\n[make_windows] Pin these in configs/config.yaml for "
                    f"reproducibility:\n"
                    f"    gravity_sign: {gravity_sign}\n"
                    f"    rotation_convention: \"{rotation_convention}\"\n"
                )

        windows = make_windows(
            aligned, seq_name,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            expected_rows=expected_rows,
        )

        kept_windows = []
        kept_labels = []
        for w in windows:
            lab = compute_drift_label(
                w, gravity_magnitude=gravity_magnitude,
                gravity_sign=gravity_sign, rotation_convention=rotation_convention,
            )
            if lab is not None:
                kept_windows.append(w)
                kept_labels.append(lab)

        labels_df = pd.DataFrame([
            {
                "seq_id": lab.seq_name,
                "window_id": lab.window_id,
                "t_start_ns": lab.t_start_ns,
                "t_end_ns": lab.t_end_ns,
                "err_mag_m": lab.err_mag_m,
                "err_dx": lab.err_dx,
                "err_dy": lab.err_dy,
                "err_dz": lab.err_dz,
            }
            for lab in kept_labels
        ])
        raw_array = stack_raw_imu(kept_windows, expected_rows=expected_rows)

        csv_path = processed_dir / f"windows_{seq_name}.csv"
        npy_path = processed_dir / f"windows_raw_{seq_name}.npy"
        labels_df.to_csv(csv_path, index=False)
        np.save(npy_path, raw_array)

        print(f"[save] {csv_path}  ({len(labels_df)} rows)")
        print(f"[save] {npy_path}  shape={raw_array.shape}")
        print(
            f"[summary] {seq_name}: err_mag_m mean={labels_df['err_mag_m'].mean():.4f}, "
            f"median={labels_df['err_mag_m'].median():.4f}, "
            f"max={labels_df['err_mag_m'].max():.4f}"
        )

    print("\nDone. Handoff files are in data/processed/ — see data/README.md.")


if __name__ == "__main__":
    main()
