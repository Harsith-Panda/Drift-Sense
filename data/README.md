# Data setup (Part A)

The team tracks:
- `data/raw/euroc/` — IMU + ground-truth CSVs for MH_01 and MH_02
- `data/interim/aligned_*.csv` — timestamp-aligned tables
- `data/processed/` — window labels + raw window tensors

Raw images (`cam0/data/`) and all of `cam1/` stay out of git. Phase 2 tracks
`cam0/data.csv`, `cam0/sensor.yaml` and the small camera arrays described below.

## 1. Download (only if the CSVs are missing)
From the ETH Research Collection page for EuRoC (DOI: 10.3929/ethz-b-000690084),
download **"Machine Hall Datasets (ZIP)"**. It contains MH_01–MH_05; we only use
MH_01_easy and MH_02_easy for Phase 1.

## 2. Extract only what we need
```bash
mkdir -p data/raw/euroc
unzip MachineHall.zip -d data/raw/euroc
```

Phase 1 does not read images. Keep `mav0/cam0/` locally for the Phase 2 camera
branch; `mav0/cam1/` is not needed.

## 3. Expected layout
```
data/raw/euroc/
├── MH_01_easy/
│   └── mav0/
│       ├── imu0/data.csv
│       └── state_groundtruth_estimate0/data.csv
└── MH_02_easy/
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
- `windows_raw_<seq>.npy` — raw IMU arrays, shape (N_windows, 400, 6) for the 2.0 s windows (720 on MH_01_easy, 592 on MH_02_easy)

```bash
python scripts/window_length_diagnostic.py
```
writes `results/diagnostics/window_length.csv`.

Part B adds `data/processed/splits.json` (leave-one-sequence-out folds)
and EDA figures under `figures/eda/` (`python scripts/run_eda.py`).

## 6. Phase 2 camera data (R3)

For **MH_01_easy and MH_02_easy only**, extract `mav0/cam0/data/`,
`mav0/cam0/data.csv` and `mav0/cam0/sensor.yaml` from the flight archives.
Images remain local and ignored. Existing windows, labels and features are reused.

```bash
python scripts/verify_camera_data.py
python scripts/make_camera_features.py
python scripts/train_camera.py
```

Preprocessing writes `data/interim/camera/frames64_<seq>.npy` (every grayscale
frame at 64 × 64) and `frame_index_<seq>.csv`, then
`data/processed/camera/cam_<seq>.npy` (`uint8`, N × 3 × 64 × 64) and
`camera_feats_<seq>.csv` (window keys, frame IDs and ten image-only features).
N is 720 for MH_01 and 592 for MH_02. Array rows match the original windows
CSV exactly. Cache and processed manifests record input/output SHA-256 hashes.

Teammates can train from the processed arrays and feature CSVs without raw
images or interim caches. R1's `DriftDataset(..., use_camera=True)` converts
the arrays to float32 in [0, 1]. The top-level `use_camera: false` is retained
for R2's E1/E2 runner; E5 explicitly enables the camera.

See [camera_data.md](../docs/camera_data.md) for feature definitions, cache
rebuilding, experiment outputs, limitations and R4 integration.
