"""
Phase 1 diagnostic / Phase 2 D5 groundwork: mean drift vs window length
for raw IMU and bias-corrected labels. Does not change the primary 1 s
window hand-off.

Writes results/diagnostics/window_length.csv

Usage (from repo root, after conda activate drift-sense):
    python scripts/window_length_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.align import align_imu_to_groundtruth
from vio_drift.data.loader import load_sequence
from vio_drift.data.windowing import make_windows
from vio_drift.labels.drift_label import compute_window_labels

WINDOW_LENGTHS = [0.25, 0.5, 1.0, 1.5, 2.0]
MAX_WINDOWS = 80


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    raw_dir = root / config["data"]["raw_dir"]
    out_dir = root / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    gravity_magnitude = config["integration"]["gravity_magnitude"]
    gravity_sign = config["integration"]["gravity_sign"]
    rotation_convention = config["integration"]["rotation_convention"]

    rows = []
    for seq_name in config["data"]["sequences"]:
        print(f"\n{'=' * 60}\n{seq_name}\n{'=' * 60}")
        imu_df, gt_df = load_sequence(raw_dir, seq_name)
        aligned = align_imu_to_groundtruth(imu_df, gt_df)

        for seconds in WINDOW_LENGTHS:
            windows = make_windows(
                aligned, seq_name,
                window_seconds=seconds,
                stride_seconds=seconds,
                expected_rows=None,
            )
            raw_errs, bc_errs = [], []
            for w in windows[:MAX_WINDOWS]:
                lab = compute_window_labels(
                    w,
                    gravity_magnitude=gravity_magnitude,
                    gravity_sign=gravity_sign,
                    rotation_convention=rotation_convention,
                )
                if lab is None or lab.err_mag_m_bc is None:
                    continue
                raw_errs.append(lab.err_mag_m)
                bc_errs.append(lab.err_mag_m_bc)

            n = len(raw_errs)
            raw_mean = float(sum(raw_errs) / n) if n else float("nan")
            bc_mean = float(sum(bc_errs) / n) if n else float("nan")
            print(f"  {seconds:.2f}s  n={n}  raw={raw_mean:.4f} m  bc={bc_mean:.4f} m")
            rows.append({
                "seq_id": seq_name,
                "window_seconds": seconds,
                "n_windows": n,
                "mean_err_mag_m": raw_mean,
                "mean_err_mag_m_bc": bc_mean,
            })

    out_path = out_dir / "window_length.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
