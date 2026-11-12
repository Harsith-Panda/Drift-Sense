# R3 camera branch: data and reproduction

R3 adds two image-based representations for the existing 1,312 windows. It
does not regenerate or change Phase 1 windows, labels, features, physics,
splits, normalizers, R1 datasets or R2 models/training. Only `.gitignore`, the
`camera:` configuration block, the camera section in `data/README.md` and
one R1 test fixture are updated among existing files. The fixture checks
missing camera files in its temporary copy, since the real files now exist;
its assertions and production behavior are preserved.

## Inputs and extraction

Extract these paths for **MH_01_easy** and **MH_02_easy** from their EuRoC
Machine Hall archives, preserving the directory structure:

```
data/raw/euroc/<seq>/mav0/cam0/data.csv
data/raw/euroc/<seq>/mav0/cam0/sensor.yaml
data/raw/euroc/<seq>/mav0/cam0/data/<timestamp>.png
```

`data.csv` has the EuRoC header `#timestamp [ns],filename`. Timestamps are
signed 64-bit integer nanoseconds, never float seconds during alignment.
The verifier checks both metadata files, strict timestamp order, every
indexed image's readability/resolution and coverage of every window. Its
CSV records measured mean/median rates, largest timestamp gap and frame
counts. A failure produces an error row and a nonzero exit status.

Only `cam0/data/` and `cam1/` are ignored. `cam0/data.csv`, `sensor.yaml`,
interim/processed arrays, CSVs, manifests, figures and results are shareable.
No raw images are written into generated artifacts or checkpoints.

## Run order

Use the project's Python environment with the dependencies in
`requirements.txt` (including PyTorch and OpenCV). From the repository root:

```bash
python scripts/verify_camera_data.py
python scripts/make_camera_features.py
python -m pytest tests/test_camera_features.py tests/test_camera_models.py -q
python scripts/train_camera.py
```

The last command runs Tier 1 and E5 (seeds 42, 43, 44). Independent runs:

```bash
python scripts/train_camera.py --tier 1
python scripts/train_camera.py --tier 2
python scripts/train_camera.py --tier 2 --seeds 42 --epochs 2 --output-dir /tmp/drift-sense-camera-smoke
```

`--output-dir` keeps short smoke runs separate from reported experiments.
The seed/epoch/config/library versions used in a run are recorded in
`results/diagnostics/camera_run_<tier>.json`. CPU, batch size 32, learning
rate 0.001, dropout 0.3, weight decay 0.0001, epoch limit 100 and early
stopping settings come from the existing DL configuration. No test-flight
tuning is performed. Re-running a tier replaces that tier's summary and the
requested seeds' files; run the full configured seed list for final tables.

## One-time frame conversion and alignment

`prepare_frame_cache` converts every indexed image to grayscale and resizes
directly to 64 × 64 with OpenCV `INTER_AREA`. It keeps no raw-resolution
copies. The original 752 × 480 aspect ratio is intentionally resized to the
plan's square tensor; image-space flow units are those of this resized grid.
No calibration/undistortion or metric-motion conversion is applied.

For each half-open interval **[t_start_ns, t_end_ns)**:

1. Select the first frame at or after the start and the last strictly before
   the end, using integer timestamp search.
2. Select the in-window frame nearest the temporal midpoint, preferring
   the earlier frame on a tie.
3. Require three distinct frames and reject uncovered boundaries or any
   internal gap greater than 2.5 times the measured median frame period.

Never use a frame outside the window, pad a missing frame, silently drop a
window or sort the output by window ID. These properties preserve R1's
array/CSV contract even with non-contiguous or shuffled window IDs.

| Artifact | Format |
|---|---|
| `data/interim/camera/frames64_<seq>.npy` | uint8, (F, 64, 64), all frames in camera timestamp order |
| `data/interim/camera/frame_index_<seq>.csv` | `frame_id` (zero-based array row), `timestamp_ns`, `filename` |
| `data/interim/camera/cache_<seq>.json` | metadata/cache checksums and conversion settings |
| `data/processed/camera/cam_<seq>.npy` | uint8, (N, 3, 64, 64), channels start/middle/end |
| `data/processed/camera/camera_feats_<seq>.csv` | window keys/times, `frame_start/middle/end`, `frame_count`, ten predictors |
| `data/processed/camera/manifest_<seq>.json` | source checksums, output checksums, shape and feature/config contract |

At 3,682 and 3,040 frames, the interim arrays total about 27.5 MB (decimal).
The two window arrays total about 16.1 MB. All arrays load with
`allow_pickle=False`. A repeated preprocessing run checks cache checksums
and can operate without raw images. After **changing raw image contents**,
run `python scripts/make_camera_features.py --rebuild-cache`; raw image
contents are not re-read during normal cache reuse. Changed camera metadata,
incomplete caches or cache corruption fail with a rebuild instruction.

## Tier 1 features

Farneback dense forward flow is computed between each pair of consecutive
cached frames once, divided by the actual timestamp difference. Each window
aggregates **only pairs with both endpoints inside that window** (39 pairs
for the measured 40-frame windows). Features are independent of labels,
ground truth, physics, IMU and neighboring windows.

| Predictor | Definition / units |
|---|---|
| `flow_mag_mean` | Mean flow magnitude over pixels and in-window pairs, pixels/s |
| `flow_mag_std` | Population std of those magnitudes, pixels/s |
| `flow_mag_p90_mean` | Mean of per-pair pixel-magnitude 90th percentiles, pixels/s |
| `flow_dx_mean`, `flow_dy_mean` | Mean signed flow; x right, y down, pixels/s |
| `flow_dx_std`, `flow_dy_std` | Population std over pixels and pairs, pixels/s |
| `blur_mean`, `blur_std` | Mean/std of frame Laplacian variance; lower usually means less sharpness |
| `corners_mean` | Mean Shi–Tomasi corner count per frame, capped at 100 |

Farneback settings are `pyr_scale=0.5, levels=3, winsize=15, iterations=3,
poly_n=5, poly_sigma=1.2, flags=0`. Corners use `qualityLevel=0.01,
minDistance=3, blockSize=3`. Blank frames yield finite zeros. Laplacian
variance also depends on texture/contrast, so it is not a calibrated blur
measurement. Flow cannot separate translation, rotation, scene depth or
dynamic objects and cannot directly measure IMU bias.

`feature_names()` is the sole predictor allowlist. IDs, frame counts,
timestamps and selected frame numbers are **not** passed to Lasso.
`scripts/train_camera.py --tier 1` joins only those ten numbers by
`seq_id, window_id` to Phase 1's 104 features. It calls the existing
`fit_model('lasso', ...)`: the same alpha grid, five time blocks, 16-row
purge, median-MAE selection and scaler fitted inside each training block.
Then it refits on the whole training flight and evaluates the other flight,
exactly as Phase 1. Each variant selects its own alpha on training data.

Tier 1 intentionally uses Phase 1's whole-flight refit for a comparable
baseline; E5 uses R1's smaller inner training partition. Their training
sample counts therefore differ. Deterministic Lasso is not rerun for three
identical seeds. Tables contain actual measurements, without a forced
claim that camera features help.

## Tier 2 and R4 handoff

`CameraCNNEncoder(dropout=0.3)` accepts float (B, 3, 64, 64) in [0, 1] and
returns (B, 64). Three blocks use 3 × 3, stride-2 convolutions with channels
16/32/64, four-group GroupNorm, ReLU and Dropout2d, followed by global
average pooling. GroupNorm avoids flight-level running statistics and
remains defined for small batches. A pre-pooling activation sees 15 × 15
input pixels; pooling covers the whole spatial map. The encoder has 23,808
parameters; R2's shared linear head adds 65 for a 23,873-parameter E5 model.

```python
from vio_drift.models.dl.camera_cnn import CameraCNNEncoder, CameraCNN
from vio_drift.features.camera import feature_names

camera_embedding = CameraCNNEncoder()(batch["cam"])  # (B, 64), for H3
magnitude = CameraCNN()(batch["cam"])                # (B, 1), for E5
four_outputs = CameraCNN(out_dim=4)(batch["cam"])    # architecture support only
camera_columns = feature_names()                    # ten Tier 1 inputs for E7
```

E5 trains magnitude only. A four-output head is available for later vector
training; this work does not claim to train drift correction. R4 can join
the image embedding with its IMU embedding and can join the ten flow
columns by window keys, fitting any feature scaler on the eligible training
rows only. The global `use_camera` switch remains R1's existing switch;
the E5 runner explicitly requests `use_camera=True`. R2's runner still
requires the existing false setting for E1/E2.

`camera_train.py` follows R2's AdamW/MSE, ReduceLROnPlateau and early-stopping
policy, reusing its seed helper and `TrainResult`. R2's model factory and
batch unpacker are currently IMU-specific, so camera training is isolated
in an R3 module instead of changing them. GroupNorm requires no BatchNorm
refresh. Validation must be the `validation` split; a test split is rejected.
The best validation checkpoint is restored before test prediction. No
camera augmentations are introduced because their label safety is unproven.

## Outputs and interpretation

- Verification: `results/diagnostics/camera_data_check.csv`.
- Tier 1 metrics/CV: `camera_tier1_check.csv`, `camera_tier1_cv.csv` in
  `results/diagnostics/`; predictions `T1_lasso[_camera]_<fold>.csv`.
- E5 metrics: `results/metrics/camera_e5.csv` (fold × seed),
  `camera_e5_summary.csv` (per-fold mean/sample std), and individual loss CSVs.
- E5 predictions: `results/predictions/E5_<fold>.csv` for configured primary
  seed 42; `E5_seed43_<fold>.csv` and `E5_seed44_<fold>.csv` for other seeds.
  Every prediction file has `seq_id, window_id, y_true, y_pred` in original order.
- Checkpoints: `models/dl/E5*.pt`, ignored by git.
- Figures: `figures/dl/camera_samples.png`, `camera_flow_examples.png` and
  `camera_E5*_loss.png`. Frames are selected by the middle window, not by
  labels or favorable examples; flow arrows show one adjacent pair.
- Walkthrough/results: `notebooks/09_camera_branch.ipynb`.

MAE, RMSE and bias are in metres in the CSVs. Positive bias means
overprediction. R² is dimensionless. An overall score is an equal-fold
average, not a pooled-window R². Seed spread is optimisation variability,
not uncertainty over unseen flights: only two flights exist. Whether the
camera improves a fused IMU model is R4/R5's E6/A3 experiment, not E5 alone.
The Phase 1 caveat remains: labels and the physics prior start windows from
ground-truth state, even though R3 preprocessing does not read ground truth.

## Measured results and report integration

The completed default run reproduces Lasso's 2.50 cm equal-fold MAE.
Appending the ten camera predictors gives 3.63 cm. E5 gives 4.42 ± 0.12 cm
(sample SD over the three per-seed equal-fold averages), with MH_01 at
3.79 ± 0.11 cm and MH_02 at 5.05 ± 0.21 cm. Neither camera variant improves
on Lasso. On the MH_02 test fold, Tier 1 CV selects alpha 0.1, yielding a
constant predictor. These outcomes are preserved rather than retuned on
test data. Full per-fold/seed values are in the result CSVs and notebook.

The R3 report source is `reports/endsem/r3_camera.tex`, with citations in
`r3_references.bib`, two-slide content in `reports/endsem/slides/r3_content.md`,
literature review in `literature/r3_visual_motion.md` and viva notes in
`docs/r3_viva.md`. R1's report entry point is left unchanged: its integrator
can add `\input{r3_camera}` and append `r3_references` to the bibliography
list when assembling the team report.
