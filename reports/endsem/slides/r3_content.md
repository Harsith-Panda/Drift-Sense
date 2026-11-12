# R3 slide content (two slides)

## Slide 1 — Turning camera images into compact model inputs

- Two cam0 streams, approximately 20 Hz; all 6,722 raw images verified.
- Same 1,312 windows, 40 frames each. Integer timestamps and half-open
  windows prevent out-of-window frame selection.
- Offline cache: grayscale 64 × 64. Three temporal channels = first,
  nearest midpoint, last. Window tensors total about 16 MB.
- Tier 1: ten dense-flow, sharpness and corner-count summaries; no labels
  used in feature extraction.
- Tier 2: three Conv2d/GroupNorm/ReLU/Dropout2d blocks → global average
  pooling → 64-number embedding → shared magnitude head (23,873 parameters).

Visual: `figures/dl/camera_samples.png`; use
`figures/dl/camera_flow_examples.png` as a small supporting image.

Speaker notes: The CNN channels are times, not colors. Flow is in resized
pixels/second and cannot by itself separate depth, translation and rotation.
R1 loads the arrays directly; R4 can concatenate the 64-number embedding
with its IMU embedding. R1/R2 production code was preserved.

## Slide 2 — Honest cross-flight results

| Representation | Equal-fold MAE |
|---|---:|
| Phase 1 Lasso (reproduced) | 2.50 cm |
| Lasso + ten camera features | 3.63 cm |
| Camera-only E5, three seeds | 4.42 ± 0.12 cm |
| Phase 1 mean baseline | 4.53 cm |

- Tier 1 uses the existing training-flight CV/alpha rule. Camera features
  did not improve this baseline; one fold selected a constant predictor.
- E5 uses the fixed 4 s purged validation split and restores the best
  validation checkpoint. Seeds 42/43/44; no seed picked by test score.
- Per-flight E5 MAE: MH_01 3.79 ± 0.11 cm; MH_02 5.05 ± 0.21 cm.
- Verdict: image-only prediction has weak explanatory power. Whether visual
  features complement IMU remains the E6/A3 fusion experiment.

Speaker notes: ± is sample standard deviation over seeds. For the overall
number, average the two folds within each seed first. These are not
confidence intervals over unseen flights. Lasso uses whole-flight refits;
E5 holds out an inner validation block. The ground-truth initialisation
caveat from Phase 1 still applies to the labels.
