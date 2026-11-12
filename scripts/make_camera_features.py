"""Build R3 cached frames, aligned camera arrays, flow features and figures.

Run verify_camera_data.py first. A subsequent run uses only the small cache;
pass --rebuild-cache after replacing raw images.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.features.camera import (
    build_camera_windows, dense_flow, feature_names, file_sha256,
    prepare_frame_cache, read_windows,
)


def save_figures(examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = ROOT / "figures/dl"
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(examples), 3, figsize=(8, 5), squeeze=False)
    for row, (seq, frames, index, selection) in enumerate(examples):
        for col, label in enumerate(("start", "middle", "end")):
            frame_id = int(selection[f"frame_{label}"])
            axes[row, col].imshow(frames[frame_id], cmap="gray", vmin=0, vmax=255)
            delta = (int(index.iloc[frame_id].timestamp_ns) - int(selection.t_start_ns)) / 1e9
            axes[row, col].set_title(f"{seq} · {label}\nwindow {selection.window_id}, +{delta:.3f} s", fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle("R3 inputs: three 64 × 64 grayscale frames per window")
    fig.tight_layout()
    fig.savefig(destination / "camera_samples.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(examples), figsize=(8, 4), squeeze=False)
    for col, (seq, frames, index, selection) in enumerate(examples):
        frame_id = int(selection.frame_middle)
        seconds = (int(index.iloc[frame_id + 1].timestamp_ns) - int(index.iloc[frame_id].timestamp_ns)) / 1e9
        flow = dense_flow(frames[frame_id], frames[frame_id + 1], seconds)
        axis = axes[0, col]
        axis.imshow(frames[frame_id], cmap="gray", vmin=0, vmax=255)
        y, x = np.mgrid[4:64:8, 4:64:8]
        axis.quiver(x, y, flow[y, x, 0] * seconds, flow[y, x, 1] * seconds,
                    color="#ffbe55", angles="xy", scale_units="xy", scale=1)
        axis.set_title(f"{seq}: adjacent frames\narrows = displacement at 64 × 64", fontsize=10)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(destination / "camera_flow_examples.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    phase1 = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
    config = yaml.safe_load((ROOT / "configs/dl_config.yaml").read_text())
    camera = config["camera"]
    if config["data"]["sequences"] != ["MH_01_easy", "MH_02_easy"]:
        raise ValueError("R3 uses only MH_01_easy and MH_02_easy")
    if camera["image_size"] != 64 or camera["frames_per_window"] != 3:
        raise ValueError("R1/R3 contract requires 3 x 64 x 64 camera arrays")
    processed = ROOT / config["data"]["processed_dir"]
    output = ROOT / camera["processed_dir"]
    if output != processed / "camera":
        raise ValueError("camera.processed_dir must match R1's data/processed/camera location")
    interim = ROOT / camera["interim_dir"]
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    for seq in config["data"]["sequences"]:
        print(f"Preparing {seq}", flush=True)
        windows_path = processed / f"windows_{seq}.csv"
        windows = read_windows(windows_path)
        if set(windows.seq_id) != {seq}:
            raise ValueError(f"Wrong flight IDs in {windows_path}")
        frames, index = prepare_frame_cache(
            ROOT / phase1["data"]["raw_dir"] / seq / "mav0/cam0", interim, seq,
            rebuild=args.rebuild_cache,
        )
        images, table = build_camera_windows(
            frames, index, windows, flow_method=camera["flow_method"],
            max_gap_factor=camera["max_gap_factor"],
        )
        array_path, table_path = output / f"cam_{seq}.npy", output / f"camera_feats_{seq}.csv"
        np.save(array_path, images, allow_pickle=False)
        table.to_csv(table_path, index=False)
        inputs = [windows_path, interim / f"frames64_{seq}.npy", interim / f"frame_index_{seq}.csv"]
        manifest = {"schema_version": 1, "seq_id": seq, "shape": list(images.shape),
                    "dtype": "uint8", "window_interval": "[t_start_ns, t_end_ns)",
                    "feature_columns": feature_names(), "camera_config": camera,
                    "inputs": {str(p.relative_to(ROOT)): file_sha256(p) for p in inputs},
                    "outputs": {p.name: file_sha256(p) for p in (array_path, table_path)}}
        (output / f"manifest_{seq}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        examples.append((seq, frames, index, table.iloc[len(table) // 2]))
        print(f"  {images.shape}, {len(table)} aligned feature rows", flush=True)
    save_figures(examples)


if __name__ == "__main__":
    main()
