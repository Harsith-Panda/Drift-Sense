"""R3 timestamp boundaries, row alignment, image-only features and offline cache."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.features.camera import (
    build_camera_windows, dense_flow, feature_names, prepare_frame_cache,
    read_frame_index, select_window_frames, verify_camera_sequence,
)

BASE = 1_403_636_579_763_555_584


def synthetic():
    times = BASE + np.arange(11, dtype=np.int64) * 50_000_000
    index = pd.DataFrame({"frame_id": np.arange(11), "timestamp_ns": times,
                          "filename": [f"{t}.png" for t in times]})
    windows = pd.DataFrame({"seq_id": ["toy", "toy"], "window_id": [19, 3],
                            "t_start_ns": [times[4], times[0]], "t_end_ns": [times[10], times[6]]})
    rng = np.random.default_rng(10)
    frame = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    frames = np.array([np.roll(frame, i, axis=1) for i in range(11)])
    return frames, index, windows


def raw_camera(tmp_path):
    frames, index, windows = synthetic()
    cam0 = tmp_path / "cam0"
    (cam0 / "data").mkdir(parents=True)
    index[["timestamp_ns", "filename"]].to_csv(cam0 / "data.csv", index=False,
                                                header=["#timestamp [ns]", "filename"])
    (cam0 / "sensor.yaml").write_text("sensor_type: camera\nresolution: [64, 64]\nrate_hz: 20\n")
    for frame, filename in zip(frames, index.filename):
        assert cv2.imwrite(str(cam0 / "data" / filename), frame)
    return cam0, frames, index, windows


def test_exact_half_open_boundaries_and_original_row_order():
    frames, index, windows = synthetic()
    selected = select_window_frames(index, windows)
    assert selected.window_id.tolist() == [19, 3]
    assert selected[["frame_start", "frame_middle", "frame_end"]].values.tolist() == [[4, 7, 9], [0, 3, 5]]
    for row in selected.itertuples(index=False):
        for frame_id in (row.frame_start, row.frame_middle, row.frame_end):
            assert row.t_start_ns <= int(index.iloc[frame_id].timestamp_ns) < row.t_end_ns
    # A 1 ns offset at EuRoC scale must not round back onto the boundary.
    windows.loc[1, "t_start_ns"] += 1
    assert select_window_frames(index, windows).iloc[1].frame_start == 1


def test_shapes_labels_ignored_and_outside_pairs_excluded():
    frames, index, windows = synthetic()
    images, table = build_camera_windows(frames, index, windows)
    assert images.shape == (2, 3, 64, 64) and images.dtype == np.uint8
    np.testing.assert_array_equal(images[0], frames[[4, 7, 9]])
    assert np.isfinite(table[feature_names()]).all().all()
    assert len(feature_names()) == 10
    poisoned = windows.assign(err_mag_m=np.nan, err_dx=-1e30, phys_mag=1e30)
    images2, table2 = build_camera_windows(frames, index, poisoned)
    np.testing.assert_array_equal(images, images2)
    pd.testing.assert_frame_equal(table, table2)
    # Changing a frame at the exclusive end cannot affect this window's flow.
    frames[10] = 0
    _, table3 = build_camera_windows(frames, index, windows)
    pd.testing.assert_frame_equal(table, table3)


def test_blank_frames_are_finite_with_no_motion_or_corners():
    frames, index, windows = synthetic()
    _, table = build_camera_windows(np.zeros_like(frames), index, windows)
    assert np.all(table[feature_names()].to_numpy() == 0)


def test_flow_direction_and_time_units():
    frames, _, _ = synthetic()
    flow = dense_flow(frames[0], frames[1], 0.05)[8:-8, 8:-8]
    assert np.median(flow[..., 0]) == pytest.approx(20, abs=2)
    assert abs(np.median(flow[..., 1])) < 1
    np.testing.assert_allclose(dense_flow(frames[0], frames[1], 0.1), dense_flow(frames[0], frames[1], 0.05) / 2)


@pytest.mark.parametrize("bad", ["empty", "gap", "duplicate", "float", "missing_start", "missing_end"])
def test_invalid_coverage_is_rejected(bad):
    _, index, windows = synthetic()
    if bad == "empty":
        windows.loc[0, "t_end_ns"] = windows.loc[0, "t_start_ns"] + 1
    elif bad == "gap":
        index = index.drop([5, 6, 7]).reset_index(drop=True)
        index["frame_id"] = np.arange(len(index))
    elif bad == "duplicate":
        index.loc[1, "timestamp_ns"] = index.loc[0, "timestamp_ns"]
    elif bad == "float":
        windows["t_start_ns"] = windows.t_start_ns.astype(float)
    elif bad == "missing_start":
        windows.loc[1, "t_start_ns"] -= 1_000_000_000
    else:
        windows.loc[0, "t_end_ns"] += 1_000_000_000
    with pytest.raises(ValueError):
        select_window_frames(index, windows)


def test_raw_verification_and_cache_reuse_without_images(tmp_path):
    cam0, frames, expected_index, windows = raw_camera(tmp_path)
    pd.testing.assert_frame_equal(read_frame_index(cam0), expected_index)
    report = verify_camera_sequence(cam0, windows)
    assert report["images_opened"] == 11
    assert report["median_rate_hz"] == 20
    cache = tmp_path / "interim"
    saved, saved_index = prepare_frame_cache(cam0, cache, "toy")
    np.testing.assert_array_equal(saved, frames)
    (cam0 / "data").rename(cam0 / "images_unavailable")
    reloaded, reloaded_index = prepare_frame_cache(cam0, cache, "toy")
    np.testing.assert_array_equal(reloaded, saved)
    pd.testing.assert_frame_equal(reloaded_index, saved_index)
    with pytest.raises(ValueError, match="unreadable"):
        verify_camera_sequence(cam0, windows)
    with (cache / "frames64_toy.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="cache changed"):
        prepare_frame_cache(cam0, cache, "toy")


def test_corrupt_image_and_unsafe_filename_fail(tmp_path):
    cam0, _, index, windows = raw_camera(tmp_path)
    (cam0 / "data" / index.iloc[2].filename).write_text("not an image")
    with pytest.raises(ValueError, match="unreadable"):
        verify_camera_sequence(cam0, windows)
    source = (cam0 / "data.csv").read_text()
    (cam0 / "data.csv").write_text(source.replace(index.iloc[0].filename, "../outside.png"))
    with pytest.raises(ValueError, match="plain filenames"):
        read_frame_index(cam0)
