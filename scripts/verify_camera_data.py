"""Check both cam0 streams before preprocessing; open every indexed image."""
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.features.camera import read_windows, verify_camera_sequence


def main():
    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
    camera = yaml.safe_load((ROOT / "configs/dl_config.yaml").read_text())["camera"]
    rows = []
    for seq in ("MH_01_easy", "MH_02_easy"):
        print(f"Checking {seq}: all images and window timestamps", flush=True)
        try:
            windows = read_windows(ROOT / config["data"]["processed_dir"] / f"windows_{seq}.csv")
            report = verify_camera_sequence(
                ROOT / config["data"]["raw_dir"] / seq / "mav0/cam0", windows,
                max_gap_factor=camera["max_gap_factor"],
            )
        except (OSError, ValueError, KeyError) as exc:
            report = {"status": "error", "error": str(exc)}
        rows.append({"seq_id": seq, **report})
        print(report, flush=True)
    output = ROOT / "results/diagnostics/camera_data_check.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit("Camera verification failed; fix/extract the listed files and rerun.")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
