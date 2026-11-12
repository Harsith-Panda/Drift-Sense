"""R3 offline camera inputs. No IMU, physics, labels or ground truth are read.

Windows use Phase 1's half-open intervals [t_start_ns, t_end_ns). All
timestamp arithmetic is integer nanoseconds. Flow is computed once between
adjacent 64x64 frames; a window only aggregates pairs wholly inside it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

IMAGE_SIZE = 64
WINDOW_COLUMNS = ["seq_id", "window_id", "t_start_ns", "t_end_ns"]
FEATURE_COLUMNS = [
    "flow_mag_mean", "flow_mag_std", "flow_mag_p90_mean",
    "flow_dx_mean", "flow_dy_mean", "flow_dx_std", "flow_dy_std",
    "blur_mean", "blur_std", "corners_mean",
]


def feature_names() -> list[str]:
    """Only these ten columns are predictors; IDs/counts/timestamps are metadata."""
    return list(FEATURE_COLUMNS)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frame_index(cam0_dir) -> pd.DataFrame:
    """Read EuRoC's '#timestamp [ns],filename' without losing nanoseconds."""
    path = Path(cam0_dir) / "data.csv"
    source = pd.read_csv(path, dtype=str)
    if len(source.columns) != 2 or source.empty or source.isna().any().any():
        raise ValueError(f"Invalid camera index: {path}")
    if source.columns[0].strip().lstrip("#") != "timestamp [ns]" or source.columns[1].strip() != "filename":
        raise ValueError(f"Expected EuRoC timestamp/filename header: {path}")
    try:
        timestamps = np.array([int(value) for value in source.iloc[:, 0]], dtype=np.int64)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Camera timestamps must be integer nanoseconds") from exc
    filenames = source.iloc[:, 1].str.strip()
    if any(not name or Path(name).name != name or "\\" in name or name in (".", "..") for name in filenames):
        raise ValueError("Camera filenames must be plain filenames inside cam0/data")
    if filenames.duplicated().any():
        raise ValueError("Duplicate camera filenames")
    result = pd.DataFrame({"frame_id": np.arange(len(source)), "timestamp_ns": timestamps, "filename": filenames})
    _validate_index(result)
    return result


def _validate_index(index):
    if len(index) < 3:
        raise ValueError("At least three camera frames are required")
    for column in ("frame_id", "timestamp_ns"):
        if not pd.api.types.is_integer_dtype(index[column]):
            raise ValueError(f"{column} must contain integers")
    timestamps = index.timestamp_ns.to_numpy()
    if not np.array_equal(index.frame_id.to_numpy(), np.arange(len(index))):
        raise ValueError("Frame IDs must match array row positions")
    if np.any(timestamps < 0) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Camera timestamps must be strictly increasing and unique")


def read_windows(path) -> pd.DataFrame:
    # Explicit usecols prevents accidental dependence on err_*, GT, or physics.
    return pd.read_csv(path, usecols=WINDOW_COLUMNS)[WINDOW_COLUMNS]


def select_window_frames(index, windows, max_gap_factor=2.5) -> pd.DataFrame:
    """First, nearest temporal midpoint, last frame, all inside each window.

    Preserve the CSV row order (even for shuffled/non-contiguous window IDs).
    Fail on missing coverage instead of padding, borrowing or dropping windows.
    """
    _validate_index(index)
    if not np.isfinite(max_gap_factor) or max_gap_factor < 1:
        raise ValueError("max_gap_factor must be finite and >= 1")
    metadata = windows[WINDOW_COLUMNS]
    if metadata.empty or metadata.isna().any().any() or metadata.duplicated(["seq_id", "window_id"]).any():
        raise ValueError("Empty, missing or duplicate window metadata")
    if metadata.seq_id.nunique() != 1:
        raise ValueError("Process one flight at a time")
    for column in WINDOW_COLUMNS[1:]:
        if not pd.api.types.is_integer_dtype(metadata[column]):
            raise ValueError(f"{column} must contain integers")
    times = index.timestamp_ns.to_numpy(dtype=np.int64)
    max_gap = int(np.median(np.diff(times)) * max_gap_factor)
    rows = []
    for seq, window_id, start, end in metadata.itertuples(index=False, name=None):
        if end <= start:
            raise ValueError(f"Window {window_id}: invalid interval")
        lo, hi = np.searchsorted(times, [start, end], side="left")
        if hi - lo < 3:
            raise ValueError(f"Window {window_id}: fewer than three frames inside interval")
        inside = times[lo:hi]
        if (inside[0] - start > max_gap or end - inside[-1] > max_gap
                or np.any(np.diff(inside) > max_gap)):
            raise ValueError(f"Window {window_id}: camera timestamp coverage has a gap")
        midpoint = start + (end - start) // 2
        middle = int(lo + np.argmin(np.abs(inside - midpoint)))
        if middle in (lo, hi - 1):
            raise ValueError(f"Window {window_id}: cannot choose three distinct temporal frames")
        rows.append({"seq_id": seq, "window_id": window_id,
                     "t_start_ns": start, "t_end_ns": end,
                     "frame_start": int(lo), "frame_middle": middle,
                     "frame_end": int(hi - 1), "frame_count": int(hi - lo)})
    return pd.DataFrame(rows)


def read_sensor(cam0_dir):
    sensor = yaml.safe_load((Path(cam0_dir) / "sensor.yaml").read_text())
    if sensor.get("sensor_type") != "camera" or len(sensor.get("resolution", [])) != 2:
        raise ValueError("sensor.yaml must describe a camera and its resolution")
    width, height = sensor["resolution"]
    if width <= 0 or height <= 0 or sensor.get("rate_hz", 0) <= 0:
        raise ValueError("Invalid camera resolution or frame rate")
    return sensor


def read_gray(path, resolution=None):
    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise ValueError(f"Missing or unreadable camera image: {path}")
    if resolution is not None and frame.shape != (resolution[1], resolution[0]):
        raise ValueError(f"Camera image resolution differs from sensor.yaml: {path}")
    return frame


def verify_camera_sequence(cam0_dir, windows, max_gap_factor=2.5):
    """Open every indexed raw image and check coverage of every window."""
    cam0_dir = Path(cam0_dir)
    index = read_frame_index(cam0_dir)
    sensor = read_sensor(cam0_dir)
    selection = select_window_frames(index, windows, max_gap_factor)
    for row in index.itertuples(index=False):
        read_gray(cam0_dir / "data" / row.filename, sensor["resolution"])
    deltas = np.diff(index.timestamp_ns.to_numpy()) / 1e9
    return {"status": "ok", "n_frames": len(index), "n_windows": len(windows),
            "images_opened": len(index), "nominal_rate_hz": sensor["rate_hz"],
            "median_rate_hz": float(1 / np.median(deltas)),
            "mean_rate_hz": float(1 / deltas.mean()), "max_frame_gap_ms": float(deltas.max() * 1000),
            "min_frames_per_window": int(selection.frame_count.min()),
            "max_frames_per_window": int(selection.frame_count.max()),
            "width": sensor["resolution"][0], "height": sensor["resolution"][1]}


def prepare_frame_cache(cam0_dir, interim_dir, seq, *, rebuild=False):
    """Convert once, reuse checked small arrays even without the raw images.

    --rebuild-cache is required after changing raw image contents. Metadata
    and cached-array changes are detected through SHA-256 on every reuse.
    """
    cam0_dir, interim_dir = Path(cam0_dir), Path(interim_dir)
    array_path = interim_dir / f"frames64_{seq}.npy"
    index_path = interim_dir / f"frame_index_{seq}.csv"
    manifest_path = interim_dir / f"cache_{seq}.json"
    if not rebuild and any(p.exists() for p in (array_path, index_path, manifest_path)):
        if not all(p.exists() for p in (array_path, index_path, manifest_path)):
            raise ValueError("Incomplete frame cache; use --rebuild-cache")
        manifest = json.loads(manifest_path.read_text())
        for path in (array_path, index_path):
            if manifest["outputs"][path.name] != file_sha256(path):
                raise ValueError("Frame cache changed; use --rebuild-cache")
        for name, digest in manifest["inputs"].items():
            if (cam0_dir / name).exists() and file_sha256(cam0_dir / name) != digest:
                raise ValueError("Camera metadata changed; use --rebuild-cache")
        index = pd.read_csv(index_path)
        _validate_index(index)
        frames = np.load(array_path, mmap_mode="r", allow_pickle=False)
        _validate_frames(frames, index)
        return frames, index
    index = read_frame_index(cam0_dir)
    sensor = read_sensor(cam0_dir)
    frames = np.empty((len(index), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for row in index.itertuples(index=False):
        frame = read_gray(cam0_dir / "data" / row.filename, sensor["resolution"])
        frames[row.frame_id] = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    interim_dir.mkdir(parents=True, exist_ok=True)
    np.save(array_path, frames, allow_pickle=False)
    index.to_csv(index_path, index=False)
    manifest = {"schema_version": 1, "seq_id": seq, "image_size": IMAGE_SIZE,
                "resize": "INTER_AREA", "opencv": cv2.__version__,
                "inputs": {name: file_sha256(cam0_dir / name) for name in ("data.csv", "sensor.yaml")},
                "outputs": {p.name: file_sha256(p) for p in (array_path, index_path)}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return frames, index


def _validate_frames(frames, index):
    if frames.shape != (len(index), IMAGE_SIZE, IMAGE_SIZE) or frames.dtype != np.uint8:
        raise ValueError("Frame cache must be uint8 (F,64,64)")


def dense_flow(previous, current, seconds):
    """Forward flow in resized-image pixels/second, x right and y down."""
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Frame interval must be positive")
    return cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0) / seconds


def build_camera_windows(frames, index, windows, *, flow_method="farneback", max_gap_factor=2.5):
    """Return uint8 (N,3,64,64) and metadata plus ten label-free summaries."""
    if flow_method != "farneback":
        raise ValueError("Supported flow_method: farneback")
    _validate_frames(frames, index)
    selection = select_window_frames(index, windows, max_gap_factor)
    cv2.setNumThreads(1)
    # dx, dy, magnitude: first/second moments plus per-pair magnitude p90.
    pair_stats = np.empty((len(frames) - 1, 7), dtype=np.float64)
    blur, corners = np.empty(len(frames)), np.empty(len(frames))
    timestamps = index.timestamp_ns.to_numpy()
    for i, frame in enumerate(frames):
        blur[i] = cv2.Laplacian(frame, cv2.CV_64F).var()
        points = cv2.goodFeaturesToTrack(frame, maxCorners=100, qualityLevel=0.01, minDistance=3, blockSize=3)
        corners[i] = 0 if points is None else len(points)
        if i:
            flow = dense_flow(frames[i - 1], frame, int(timestamps[i] - timestamps[i - 1]) / 1e9).astype(np.float64)
            dx, dy = flow[..., 0], flow[..., 1]
            mag = np.hypot(dx, dy)
            pair_stats[i - 1] = [dx.mean(), dy.mean(), mag.mean(), (dx**2).mean(),
                                 (dy**2).mean(), (mag**2).mean(), np.percentile(mag, 90)]
    summaries = []
    for row in selection.itertuples(index=False):
        start, end = row.frame_start, row.frame_end
        dx, dy, mag, dx2, dy2, mag2, p90 = pair_stats[start:end].mean(axis=0)
        summaries.append([mag, np.sqrt(max(0, mag2 - mag**2)), p90, dx, dy,
                          np.sqrt(max(0, dx2 - dx**2)), np.sqrt(max(0, dy2 - dy**2)),
                          blur[start:end + 1].mean(), blur[start:end + 1].std(),
                          corners[start:end + 1].mean()])
    values = np.asarray(summaries)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite camera summaries")
    table = pd.concat([selection, pd.DataFrame(values, columns=FEATURE_COLUMNS)], axis=1)
    camera = frames[selection[["frame_start", "frame_middle", "frame_end"]].to_numpy()]
    return np.asarray(camera, dtype=np.uint8), table
