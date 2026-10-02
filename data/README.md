# Data setup (Part A)

The team tracks:
- `data/raw/euroc/` — IMU + ground-truth CSVs for MH_01, MH_02, MH_03
- `data/interim/aligned_*.csv` — timestamp-aligned tables
- `data/processed/` — window labels + raw window tensors

Camera folders (`cam0/`, `cam1/`) stay out of git (see `.gitignore`).

## 1. Download (only if the CSVs are missing)
From the ETH Research Collection page for EuRoC (DOI: 10.3929/ethz-b-000690084),
download **"Machine Hall Datasets (ZIP)"**. It contains MH_01–MH_05; we only use
MH_01_easy, MH_02_easy, MH_03_medium for Phase 1.

## 2. Extract only what we need
```bash
mkdir -p data/raw/euroc
unzip MachineHall.zip -d data/raw/euroc
```

Delete or ignore `mav0/cam0/` and `mav0/cam1/` — we never read image files
in Phase 1.

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

## 5. Then build the windows + labels
```bash
python scripts/make_windows.py
```
Produces, per sequence, in `data/processed/`:
- `windows_<seq>.csv` — ids, timestamps, raw drift (`err_mag_m`, `err_dx/dy/dz`)
  and oracle bias-corrected copies (`err_*_bc`)
- `windows_raw_<seq>.npy` — raw IMU arrays, shape (N_windows, 200, 6)

```bash
python scripts/window_length_diagnostic.py
```
writes `results/diagnostics/window_length.csv`.

Part B adds `data/processed/splits.json` (leave-one-sequence-out folds)
and EDA figures under `figures/eda/` (`python scripts/run_eda.py`).
