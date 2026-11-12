# R4 slides: Hybrid approach (3 slides)

## Slide 1: Build on physics, don't replace it
- Figure: `figures/dl/hybrid_architecture.png`
- Phase 1: our physics feature alone reaches R² 0.82; the measured ceiling for any bias-based predictor is 0.83
- Every branch gives 64 numbers: deep IMU encoder (LSTM or 1D-CNN), feature MLP (104 features + 4 physics), camera CNN
- **H1 (E3):** fuse everything, predict directly
- **H2 (E4, main):** `physics answer (a·phys_mag + b)` + learned correction; the correction starts at 0, so training *begins* at physics
- **H3 (E6):** IMU + camera

## Slide 2: Results (test flight, mean of 2 folds, 3 seeds)
- Figure: `figures/dl/hybrid_comparison.png`

| Model | Mean MAE (cm) | Mean R² |
|---|---|---|
| Phase 1 Lasso (ML) | 2.50 | 0.76 |
| Physics only | 2.11 | 0.82 |
| Raw IMU, 1D-CNN | 2.60 ± 0.12 | 0.72 |
| H1: IMU + features + physics* | 2.64 ± 0.03 | 0.73 |
| **H2: physics + LSTM correction** | **2.15 ± 0.01** | **0.82** |
| **H2: physics + CNN correction** | **2.13 ± 0.02** | **0.82** |
| H3: IMU + camera* | 2.85 ± 0.05 | 0.64 |

\* with 10-epoch warm-up (see slide 3)

- **H2 is the best DL model**, better than Phase 1 ML on both flights, and works with either encoder
- It ties with physics alone, which is already at the ceiling
- Same inputs, different design: H1 learns physics from scratch (0.73); H2 gets it for free (0.82)

## Slide 3: What we learned
- Figure: `figures/dl/hybrid_h2_curves.png` (validation MAE; dashed = physics)
- The validation block is the landing phase: label std 2–3.6 cm vs 6–9 cm in training, and physics R² ≈ 0 there
- Direct models were stopped at epoch 1, almost untrained. A 10-epoch warm-up (chosen from training curves) fixed it: H1 0.53 → 0.73, H3 0.38 → 0.64. Both versions are reported
- H2 is safe by design: if the network can't help, it falls back to physics
- The camera adds variance, not accuracy (agrees with R3)
- **Next:** a validation split that covers take-off, cruise and landing
