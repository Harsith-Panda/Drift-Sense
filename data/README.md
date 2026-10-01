# Data setup (Part A)

## 1. Download
From the ETH Research Collection page for EuRoC (DOI: 10.3929/ethz-b-000690084),
download **"Machine Hall Datasets (ZIP)"**. It contains MH_01–MH_05; we only use
MH_01_easy, MH_02_easy, MH_03_medium for Phase 1.

## 2. Extract only what we need
Each sequence's images are large and unused. If the zip lets you extract selectively:

```bash
mkdir -p data/raw/euroc
unzip MachineHall.zip -d data/raw/euroc
```

Then, per sequence, delete or ignore `mav0/cam0/` and `mav0/cam1/` — we never read
image files in this project.

## 3. Expected layout
```
data/raw/euroc/
├── MH_01_easy/
│   └── mav0/
│       ├── imu0/data.csv
│       └── state_groundtruth_estimate0/data.csv
├── MH_02_easy/
│   └── mav0/...
└── MH_03_medium/
    └── mav0/...
```

## 4. Verify before doing anything else
```bash
python scripts/verify_data.py
```
This checks that both CSVs exist for every sequence in `configs/config.yaml`,
prints their column headers, and tells you plainly if a column name doesn't
match what `loader.py` expects. **Run this first — every other script assumes
it passed.**

## 5. Then build the windows + labels
```bash
python scripts/make_windows.py
```
Produces, per sequence, in `data/processed/`:
- `windows_<seq>.csv` — one row per window: ids, timestamps, drift labels
- `windows_raw_<seq>.npy` — raw IMU arrays, shape (N_windows, 200, 6)

These two files are the hand-off to Part B (EDA/splits) and Part C (features).
