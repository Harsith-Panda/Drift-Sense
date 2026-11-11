# R1 data foundation

Run from the repository root in the project environment:

```sh
python scripts/prepare_dl_data.py
python -m pytest tests/test_dl_data.py -q
python -m pytest tests -q
python -c "import torch, cv2; print(torch.__version__, cv2.__version__)"
python -m pip check
```

The preparation script verifies existing Phase 1 counts, raw shapes, feature
schema, matching labels and physics keys before writing `splits_dl.json` and
`normalization_dl/leave_<test-flight>.json`. It does not regenerate Phase 1
artifacts. Running it again with unchanged inputs gives identical JSON bytes.

## Split contract

Reuse Phase 1's outer folds from `splits.json`: leave MH_01_easy out first,
then MH_02_easy. Within the other flight, sort by actual `t_start_ns` and
reserve the final `ceil(0.2 * N)` windows for validation, before purging.
Keep earlier windows only when their actual `t_end_ns` is at most the first
validation start minus 4,000,000,000 ns. Thus the gap is four seconds of
empty time **between window intervals**, not four seconds between starts.
Phase 1 windows are half-open `[t_start_ns, t_end_ns)`; equality is allowed.
All other earlier windows are explicitly recorded in `excluded_gap` and
must not be used for fitting. Original IDs and integer timestamps are retained.
Raw array positions refer to original CSV row order, not numeric window IDs.

| Held-out flight | Training | Validation | Purged | Test |
| --- | ---: | ---: | ---: | ---: |
| MH_01_easy | 450 | 119 | 23 | 720 |
| MH_02_easy | 553 | 144 | 23 | 592 |

The actual gap is 4 seconds in both folds. The validation fraction is measured
against the source flight before purging; 592 * 0.2 rounds up to 119.
Only validation may drive early stopping or hyperparameter selection. Test
flights remain untouched by fitting and selection. The two folds must have
independent models and normalizers. The seed is recorded for later stochastic
training; split generation itself is deterministic and uses no RNG.

## Normalization contract

`fit_fold_normalizer` selects only explicit inner-training keys. It computes
six IMU channel means and population standard deviations over axes `(0,1)`
of the selected `(N,400,6)` tensor. Channel order is `gx,gy,gz,ax,ay,az`.
Overlapping training samples count once per occurrence in that tensor;
no attempt is made to reconstruct unique raw timestamps.

For engineered features it reuses `feature_names()` and joins the 104 columns
by `(seq_id, window_id)`. IDs, timestamps, labels and physics features are not
scaled. Moments use float64; transforms return float32 `(N,T,6)` and `(N,104)`.
Standard deviations <= 1e-12 use scale 1. Nonfinite inputs fail explicitly.
Transforms never fit or update statistics. Each JSON stores the fold ID,
training keys, feature order and statistics; loading requires an expected
fold ID to guard against accidental cross-fold reuse.

Example after loading the source dictionaries as in the preparation script:

```python
scaler = fit_fold_normalizer(fold, windows, raw, features)
validation_imu = scaler.transform_imu(validation_raw)
test_features = scaler.transform_features(test_feature_frame)
scaler.save(stats_path)
scaler = FoldNormalizer.load(stats_path, fold["fold_id"])
```

## Completed R1 integration

Run `python scripts/prepare_r1_data.py` for full preparation and
`python scripts/demo_dl_data.py` for both-fold CPU DataLoader examples.
The original first-stage preparation command remains supported, but the full
command is required before Dataset use because it supplies physics provenance.
`notebooks/07_dl_data_prep.ipynb` is the executable walkthrough.

`DriftDataset(processed_dir, fold_id, split, artifact_dir=None, use_camera=False,
augmentation=None, seed=42)` returns the Section 6 contract. Dataset order follows
the original windows CSV; training DataLoaders shuffle with a seeded generator.
Call `dataset.set_epoch(epoch)` before each epoch. Workers are recreated each epoch.
All returned arrays are independent copies; validation/test reject enabled augmentation.

Physics files under `physics_dl/` use the original Phase 1 integrator with a bias
fitted only to the union of inner-training intervals. Matched GT timestamps are
also restricted. Metadata binds training references and source/table checksums.
Normalizer sidecars bind source and statistics checksums. Dataset loading rejects
stale source artifacts, mismatched training keys, missing joins and invalid folds.
Physics is unscaled in world units, ordered dx,dy,dz,magnitude. GT initial states
remain a limitation. Original Phase 1 tables are never overwritten.

Optional camera arrays must be uint8 (N,3,64,64), with `camera_feats_<seq>.csv`
key rows matching windows CSV order. Values become float32 in [0,1]. Real arrays
are absent; synthetic camera tests verify the interface and missing-data errors.
R3 must validate image timestamps; key checks alone cannot prove image alignment.

Augmentation defaults off. Noise uses the verified sensor densities 1.6968e-4 gyro
and 2e-3 accel, multiplied by sqrt(200 Hz), then 0.1 and clipped at three sigma.
Thus the largest added value is 0.3 nominal sensor sigma. Crop masks at most four
edge samples in normalized space without shortening or resampling time. It models
missing observations, not changed motion; exact drift invariance is not claimed.
Labels, camera, physics and engineered features remain original references.

## A1/A2 handoff

`configs/r1_ablations.yaml` and `results/ablations/r1_data_manifest.json` describe
84 jobs: 66 A1 (eight factorial regularization cases plus three augmentation probes,
two folds, three seeds) and 18 A2. All use CPU. Early-stopping-off selects the final
epoch; on selects best validation loss. No validation-driven scheduler is used.

A2 percentages apply AFTER reserving validation and purging, not to the entire
flight. Chronological prefixes use ceiling: 225/338/450 and 277/415/553 training
windows. Validation/test never change. Every subset refits scalers AND calibration;
artifacts live in `ablations_dl/train_050` and `train_075`; 100% reuses base artifacts.
This design confounds flight phase with quantity and must be reported as such.

R2/R5 model/trainer/evaluation components are absent; jobs have not been trained.
R2 should consume job overrides and define optimizer decay semantics. R4 should
use DL-specific physics and fit residual-prior coefficients on inner training only.
R3 provides camera artifacts. R5 runs and scores jobs with three-seed spread.

## Writing and integration

See `literature/r1_regularization.md`, `reports/endsem/r1_data.tex`,
`r1_references.bib`, `slides/r1_content.md`, and `docs/r1_viva.md`.
The IEEE main.tex supplies R1 integration and explicit teammate insertion slots.
No LaTeX engine is installed here; compilation and full-team integration remain
pending. Buddy review awaits a teammate PR. These are drafts, with no invented results.

## Final validation

Full project suite: **62 passed**, no skips or warnings (including 19 R1 cases).
Notebook validated and executed all code cells. Every batch, including partial
final batches, was checked across train/validation/test for both folds.
CPU tensor shapes are `(B,6,400)`, `(B,104)`, `(B,4)`, `(B,)`, `(B,3)`.
Synthetic camera batch is `(32,3,64,64)` float32; real camera data remain absent.
`pip check` and `git diff --check` passed. Existing Phase 1 tests pass; Phase 1
algorithms and data were not regenerated. No model training was performed.

## Completion checklist

| Requirement | Status | Evidence / dependency |
| --- | --- | --- |
| Full plan reviewed; two-flight scope | DONE | Sections 0–12 read |
| Configuration and CPU dependencies | DONE | Existing pins preserved; imports checked |
| Purged chronological flight folds | DONE | 450/119/720 and 553/144/592 |
| Training-only normalization | DONE | Perturbation tests and fold-key checks |
| Shared Dataset contract and joins | DONE | Both-fold tests and all-batch demonstration |
| Fold/subset physics calibration | DONE | Inner-training interval union; leakage tests |
| Optional camera interface | DONE | Synthetic tests and missing-file rejection |
| Real camera data validation | BLOCKED | R3 arrays and image timestamp checks absent |
| Label-safe observation augmentation | DONE | Train-only; crop-and-mask interpretation documented |
| Dataset/normalization tests | DONE | 19 R1 cases; 62 full-project cases |
| Notebook and reproducible loaders | DONE | Executed notebook and demonstrations |
| A1 data/config contribution | DONE | 66 jobs, no fabricated metrics |
| A2 data/config contribution | DONE | 18 jobs; subset-specific preprocessing |
| Complete A1/A2 model results | BLOCKED | R2 models/trainer and R5 harness absent |
| R1 report/literature/slides/viva drafts | DONE | Source files included |
| LaTeX integrator scaffold | DONE | IEEE main.tex, R1 section and references |
| Compiled complete team report | BLOCKED | TeX toolchain and other-role sections absent |
| Teammate buddy PR review | PENDING | Requires a teammate PR |
