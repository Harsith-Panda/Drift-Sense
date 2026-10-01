"""
EuRoC's Machine Hall zip is gated behind a page on the ETH Research
Collection (no direct, script-friendly download URL) — so this script
does NOT auto-download anything. It exists so a teammate re-running the
pipeline from scratch has the exact steps written down in one place,
and it double-checks the result once you've done the manual part.

Usage:
    python scripts/download_data.py            # prints instructions
    python scripts/download_data.py --check     # checks what's present
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

INSTRUCTIONS = """
EuRoC MAV Dataset — manual download steps
===========================================

1. Open: https://doi.org/10.3929/ethz-b-000690084
   (this is the current, official home for the dataset)

2. Download "Machine Hall Datasets (ZIP)" — it contains MH_01–MH_05.
   We only need MH_01_easy, MH_02_easy, MH_03_medium for Phase 1.

3. Extract it so the result looks like:

     data/raw/euroc/MH_01_easy/mav0/imu0/data.csv
     data/raw/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv
     data/raw/euroc/MH_02_easy/mav0/...
     data/raw/euroc/MH_03_medium/mav0/...

   You can delete mav0/cam0 and mav0/cam1 for these three — we never
   read camera images in Phase 1.

4. Run this script again with --check, or just run:
     python scripts/verify_data.py
"""


def check(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    raw_dir = Path(config["data"]["raw_dir"])
    sequences = config["data"]["sequences"]

    print("Checking data/raw/euroc/ ...\n")
    all_present = True
    for seq_name in sequences:
        imu_path = raw_dir / seq_name / "mav0" / "imu0" / "data.csv"
        gt_path = raw_dir / seq_name / "mav0" / "state_groundtruth_estimate0" / "data.csv"
        ok = imu_path.exists() and gt_path.exists()
        all_present &= ok
        print(f"  [{'OK' if ok else 'MISSING'}] {seq_name}")

    print()
    if all_present:
        print("All expected sequences found. Run scripts/verify_data.py next.")
    else:
        print("Some sequences missing — follow the instructions above.")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "config.yaml"

    if "--check" in sys.argv:
        check(config_path)
    else:
        print(INSTRUCTIONS)


if __name__ == "__main__":
    main()
