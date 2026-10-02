"""
The single most important sanity check in Part A: run this against your
REAL downloaded data, not synthetic data, since it's checking whether the
integration math actually agrees with reality on your sequences.

Run with:
    pytest tests/test_label_sanity.py -v

It will SKIP (not fail) if data/raw/euroc/MH_01_easy isn't present, so it's
safe to run in any environment.
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.data.loader import load_sequence
from vio_drift.data.align import align_imu_to_groundtruth
from vio_drift.data.windowing import make_windows
from vio_drift.labels.imu_integration import calibrate_conventions
from vio_drift.labels.drift_label import compute_drift_label, compute_window_labels

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())


def _seq_available(seq_name: str) -> bool:
    cfg = _config()
    seq_dir = ROOT / cfg["data"]["raw_dir"] / seq_name
    return (seq_dir / "mav0" / "imu0" / "data.csv").exists()


@pytest.mark.skipif(not _seq_available("MH_01_easy"), reason="MH_01_easy not downloaded")
def test_error_grows_with_window_length():
    """
    Physical sanity check: integrating IMU over a LONGER window should
    accumulate MORE error (or at least not dramatically less) than a
    SHORTER window, on average. If this fails, something is wrong with
    the integration (wrong sign, wrong convention, or a units mismatch) —
    don't trust the drift labels until this passes.
    """
    cfg = _config()
    imu_df, gt_df = load_sequence(ROOT / cfg["data"]["raw_dir"], "MH_01_easy")
    aligned = align_imu_to_groundtruth(imu_df, gt_df)

    best = calibrate_conventions(aligned, gravity_magnitude=cfg["integration"]["gravity_magnitude"])

    def mean_error_for_window(seconds: float) -> float:
        windows = make_windows(aligned, "MH_01_easy", window_seconds=seconds,
                                stride_seconds=seconds, expected_rows=None)
        errs = []
        for w in windows[:50]:  # first 50 windows is enough for a sanity check
            lab = compute_drift_label(
                w,
                gravity_magnitude=cfg["integration"]["gravity_magnitude"],
                gravity_sign=best["gravity_sign"],
                rotation_convention=best["rotation_convention"],
            )
            if lab is not None:
                errs.append(lab.err_mag_m)
        assert errs, f"no valid windows produced for {seconds}s — check ground-truth coverage"
        return sum(errs) / len(errs)

    err_short = mean_error_for_window(0.25)
    err_long = mean_error_for_window(1.0)

    print(f"\nmean error @0.25s windows: {err_short:.4f} m")
    print(f"mean error @1.0s windows:  {err_long:.4f} m")

    assert err_long >= err_short * 0.5, (
        f"1.0s window error ({err_long:.4f} m) is much smaller than 0.25s "
        f"window error ({err_short:.4f} m) — this is physically backwards "
        f"and suggests a bug in imu_integration.py or a column mapping "
        f"issue in loader.py. Do not proceed to Part C/D with these labels."
    )


@pytest.mark.skipif(not _seq_available("MH_01_easy"), reason="MH_01_easy not downloaded")
def test_calibration_probe_error_is_reasonable():
    """
    calibrate_conventions() already warns on a bad probe error; this test
    turns that warning into a hard failure so it can't be missed.
    """
    cfg = _config()
    imu_df, gt_df = load_sequence(ROOT / cfg["data"]["raw_dir"], "MH_01_easy")
    aligned = align_imu_to_groundtruth(imu_df, gt_df)
    best = calibrate_conventions(aligned, gravity_magnitude=cfg["integration"]["gravity_magnitude"])
    assert best["probe_error_m"] < 1.0, (
        f"Best-case probe error is {best['probe_error_m']:.2f} m over a very "
        f"short window — too high to trust. Re-check loader.py's column "
        f"mapping against your actual CSV headers (scripts/verify_data.py)."
    )


@pytest.mark.skipif(not _seq_available("MH_01_easy"), reason="MH_01_easy not downloaded")
def test_bias_corrected_error_is_smaller_than_raw():
    """
    Oracle check: subtracting the dataset's own gyro/accel bias estimates
    should reduce mean 1 s position error. If it does not, subtract_bias
    is using the wrong columns or the wrong frame.
    """
    cfg = _config()
    imu_df, gt_df = load_sequence(ROOT / cfg["data"]["raw_dir"], "MH_01_easy")
    aligned = align_imu_to_groundtruth(imu_df, gt_df)
    windows = make_windows(
        aligned, "MH_01_easy",
        window_seconds=1.0, stride_seconds=1.0, expected_rows=None,
    )

    raw_errs, bc_errs = [], []
    for w in windows[:50]:
        lab = compute_window_labels(
            w,
            gravity_magnitude=cfg["integration"]["gravity_magnitude"],
            gravity_sign=cfg["integration"]["gravity_sign"],
            rotation_convention=cfg["integration"]["rotation_convention"],
        )
        if lab is None or lab.err_mag_m_bc is None:
            continue
        raw_errs.append(lab.err_mag_m)
        bc_errs.append(lab.err_mag_m_bc)

    assert raw_errs, "no valid windows for bias-correction check"
    mean_raw = sum(raw_errs) / len(raw_errs)
    mean_bc = sum(bc_errs) / len(bc_errs)
    print(f"\nmean raw 1s error: {mean_raw:.4f} m")
    print(f"mean bias-corrected 1s error: {mean_bc:.4f} m")
    assert mean_bc < mean_raw, (
        f"bias-corrected mean ({mean_bc:.4f} m) is not smaller than raw "
        f"({mean_raw:.4f} m) — check subtract_bias column mapping."
    )
