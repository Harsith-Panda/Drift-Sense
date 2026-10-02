"""
Part B: generate EDA figures and summary tables.

Usage (repo root, drift-sense env):
    python scripts/run_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.align import align_imu_to_groundtruth, report_gaps
from vio_drift.data.loader import load_sequence

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "eda"
RESULTS = ROOT / "results" / "eda"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })


def load_all(config: dict):
    raw_dir = ROOT / config["data"]["raw_dir"]
    loaded = {}
    for seq in config["data"]["sequences"]:
        imu, gt = load_sequence(raw_dir, seq)
        loaded[seq] = (imu, gt)
    return loaded


def quality_table(config: dict, loaded: dict) -> pd.DataFrame:
    rows = []
    for seq, (imu, gt) in loaded.items():
        gaps = report_gaps(imu, gt)
        aligned = align_imu_to_groundtruth(imu, gt)
        n_imu = len(imu)
        n_drop = n_imu - len(aligned)
        imu_null = int(imu.isna().sum().sum())
        gt_null = int(gt.isna().sum().sum())
        imu_dt = np.diff(imu["t_ns"].to_numpy()) / 1e9
        rows.append({
            "seq_id": seq,
            "imu_rows": n_imu,
            "gt_rows": len(gt),
            "duration_s": gaps["imu"]["duration_s"],
            "imu_rate_hz": gaps["imu"]["implied_rate_hz"],
            "gt_rate_hz": gaps["ground_truth"]["implied_rate_hz"],
            "imu_median_dt_s": gaps["imu"]["median_dt_s"],
            "imu_max_dt_s": gaps["imu"]["max_dt_s"],
            "imu_missing": imu_null,
            "gt_missing": gt_null,
            "aligned_rows": len(aligned),
            "align_dropped": n_drop,
            "align_dropped_pct": 100.0 * n_drop / n_imu,
            "imu_dt_jitter_s": float(np.std(imu_dt)),
        })
    return pd.DataFrame(rows)


def plot_raw_imu(loaded: dict, seq: str, seconds: float = 10.0) -> None:
    imu, _ = loaded[seq]
    t = (imu["t_ns"] - imu["t_ns"].iloc[0]) / 1e9
    mask = t < seconds
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t[mask], imu.loc[mask, "gx"], label="gx", lw=0.8)
    axes[0].plot(t[mask], imu.loc[mask, "gy"], label="gy", lw=0.8)
    axes[0].plot(t[mask], imu.loc[mask, "gz"], label="gz", lw=0.8)
    axes[0].set_ylabel("gyro (rad/s)")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].set_title(f"{seq} — first {seconds:.0f}s IMU")
    axes[1].plot(t[mask], imu.loc[mask, "ax"], label="ax", lw=0.8)
    axes[1].plot(t[mask], imu.loc[mask, "ay"], label="ay", lw=0.8)
    axes[1].plot(t[mask], imu.loc[mask, "az"], label="az", lw=0.8)
    axes[1].set_ylabel("accel (m/s$^2$)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "raw_imu_signals.png")
    plt.close(fig)


def plot_trajectories(loaded: dict) -> None:
    fig = plt.figure(figsize=(12, 4.2))
    for i, (seq, (_, gt)) in enumerate(loaded.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot(gt["px"], gt["py"], gt["pz"], lw=0.7)
        ax.set_title(seq)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
    fig.suptitle("Ground-truth trajectories", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "true_trajectories.png")
    plt.close(fig)


def plot_speed(loaded: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for seq, (_, gt) in loaded.items():
        t = (gt["t_ns"] - gt["t_ns"].iloc[0]) / 1e9
        speed = np.linalg.norm(gt[["vx", "vy", "vz"]].to_numpy(), axis=1)
        ax.plot(t, speed, lw=0.8, label=seq)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    ax.set_title("Ground-truth speed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "speed_comparison.png")
    plt.close(fig)


def load_labels(config: dict) -> pd.DataFrame:
    processed = ROOT / config["data"]["processed_dir"]
    frames = [pd.read_csv(processed / f"windows_{seq}.csv") for seq in config["data"]["sequences"]]
    return pd.concat(frames, ignore_index=True)


def label_summary(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seq, g in labels.groupby("seq_id"):
        for col, kind in [("err_mag_m", "raw"), ("err_mag_m_bc", "bias_corrected")]:
            s = g[col]
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            rows.append({
                "seq_id": seq,
                "label": kind,
                "n": len(s),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "skew": float(s.skew()),
                "iqr_outliers": n_out,
            })
    return pd.DataFrame(rows)


def plot_label_distributions(labels: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for seq, g in labels.groupby("seq_id"):
        axes[0].hist(g["err_mag_m"], bins=30, alpha=0.45, label=seq)
        axes[1].hist(g["err_mag_m_bc"], bins=30, alpha=0.45, label=seq)
    axes[0].set_title("Primary target: raw IMU drift")
    axes[0].set_xlabel("err_mag_m (m)")
    axes[1].set_title("Oracle comparison: bias-corrected")
    axes[1].set_xlabel("err_mag_m_bc (m)")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "label_distribution.png")
    plt.close(fig)


def plot_label_over_time(labels: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)
    for ax, (seq, g) in zip(axes, labels.groupby("seq_id")):
        t = (g["t_start_ns"] - g["t_start_ns"].iloc[0]) / 1e9
        ax.plot(t, g["err_mag_m"], lw=0.7, label="raw")
        ax.plot(t, g["err_mag_m_bc"], lw=0.7, label="bias-corrected")
        ax.set_ylabel("error (m)")
        ax.set_title(seq)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("window start (s)")
    fig.suptitle("Drift label over the flight")
    fig.tight_layout()
    fig.savefig(FIG / "label_over_time.png")
    plt.close(fig)


def plot_axis_errors(labels: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    raw = labels.melt(id_vars="seq_id", value_vars=["err_dx", "err_dy", "err_dz"],
                      var_name="axis", value_name="error")
    bc = labels.melt(id_vars="seq_id", value_vars=["err_dx_bc", "err_dy_bc", "err_dz_bc"],
                     var_name="axis", value_name="error")
    raw.boxplot(column="error", by=["seq_id", "axis"], ax=axes[0], rot=30)
    bc.boxplot(column="error", by=["seq_id", "axis"], ax=axes[1], rot=30)
    axes[0].set_title("Raw 3-axis error")
    axes[1].set_title("Bias-corrected 3-axis error")
    axes[0].set_ylabel("error (m)")
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(FIG / "axis_error_boxplots.png")
    plt.close(fig)


def plot_window_length() -> None:
    path = ROOT / "results" / "diagnostics" / "window_length.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for seq, g in df.groupby("seq_id"):
        axes[0].plot(g["window_seconds"], g["mean_err_mag_m"], marker="o", label=seq)
        axes[1].plot(g["window_seconds"], g["mean_err_mag_m_bc"], marker="o", label=seq)
    axes[0].set_title("Raw IMU — error vs window length")
    axes[1].set_title("Bias-corrected — error vs window length")
    for ax in axes:
        ax.set_xlabel("window length (s)")
        ax.set_ylabel("mean error (m)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "error_vs_window_length.png")
    plt.close(fig)


def plot_easy_vs_medium(loaded: dict, labels: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    speeds = []
    names = []
    for seq, (_, gt) in loaded.items():
        speed = np.linalg.norm(gt[["vx", "vy", "vz"]].to_numpy(), axis=1)
        speeds.append(speed)
        names.append(seq)
    axes[0].boxplot(speeds, tick_labels=names)
    axes[0].set_ylabel("speed (m/s)")
    axes[0].set_title("Speed by sequence")
    axes[0].tick_params(axis="x", rotation=15)

    gyro_norms = []
    for seq, (imu, _) in loaded.items():
        gyro_norms.append(np.linalg.norm(imu[["gx", "gy", "gz"]].to_numpy(), axis=1))
    axes[1].boxplot(gyro_norms, tick_labels=names)
    axes[1].set_ylabel("|gyro| (rad/s)")
    axes[1].set_title("Rotation rate by sequence")
    axes[1].tick_params(axis="x", rotation=15)

    labels.boxplot(column="err_mag_m", by="seq_id", ax=axes[2])
    axes[2].set_title("Raw 1 s drift by sequence")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("err_mag_m (m)")
    fig.suptitle("Easy vs medium")
    fig.tight_layout()
    fig.savefig(FIG / "easy_vs_medium.png")
    plt.close(fig)


def main() -> None:
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())

    print("[eda] loading sequences")
    loaded = load_all(config)
    q = quality_table(config, loaded)
    q.to_csv(RESULTS / "quality_summary.csv", index=False)
    print(q.to_string(index=False))

    print("[eda] plots: IMU / trajectory / speed")
    plot_raw_imu(loaded, config["data"]["sequences"][0])
    plot_trajectories(loaded)
    plot_speed(loaded)

    print("[eda] labels")
    labels = load_labels(config)
    assert labels[["err_mag_m", "err_mag_m_bc"]].isna().sum().sum() == 0
    summary = label_summary(labels)
    summary.to_csv(RESULTS / "label_summary.csv", index=False)
    print(summary.to_string(index=False))

    plot_label_distributions(labels)
    plot_label_over_time(labels)
    plot_axis_errors(labels)
    plot_window_length()
    plot_easy_vs_medium(loaded, labels)

    print(f"[eda] figures → {FIG}")
    print(f"[eda] tables  → {RESULTS}")
    print("[eda] log(err_mag_m): not recommended — raw magnitude is tight, not a long tail.")
    print("[eda] splits: leave-one-sequence-out (data/processed/splits.json)")


if __name__ == "__main__":
    main()
