# R5 slides: Technical Validation, Ablations and Drift Correction (3 slides)

## Slide 1: Comparability & Benchmark Anchoring
- Figure: `figures/dl/ablation_table_e0_e7.png`
- **Rule 1:** Reproduce Phase 1 benchmarks before evaluating any deep network
  - Mean baseline: 4.53 cm
  - Tuned Lasso (E0): **2.50 cm MAE**, R² = 0.76
  - Calibrated physics prior: R² = 0.82
- Enforced by automated regression tests (`test_dl_eval.py`)
- All models evaluated across 3 fixed random seeds (42, 123, 999) under identical leave-one-sequence-out folds

## Slide 2: Ablation Findings & Best Architecture
- Figure: `figures/dl/seed_spread_a4.png` and `figures/dl/data_scaling_a2.png`

| Model | Inputs | MAE (cm) | R² | Note |
|---|---|---|---|---|
| E0 Lasso | 104 Hand-crafted | 2.50 ± 0.00 | 0.76 | Phase 1 baseline |
| E1 1D-CNN | Raw IMU | 19.87 ± 3.12 | -10.10 | Underfits small data |
| E2 LSTM | Raw IMU | 4.36 ± 2.46 | 0.33 | High seed variance |
| E3 Hybrid H1 | IMU + Feat + Phys | 17.05 ± 3.15 | -6.73 | Needs large data |
| **E4 Hybrid H2** | **Phys + Residual** | **2.30 ± 0.09** | **0.79** | **Beats Lasso ✓** |
| E7 Best Hybrid | Multi-modal | 2.41 ± 0.29 | 0.78 | Low variance |

- **Key Insight:** Inductive bias is critical on small datasets. Pure deep learning struggles; grounding the network in analytical physics (H2) achieves the best performance with minimal seed spread (std 0.09 cm).

## Slide 3: Practical Payoff — Trajectory Drift Correction
- Figure: `figures/trajectories/trajectory_MH_01_easy.png` and `figures/trajectories/ate_over_time.png`
- Reconstruct continuous paths using non-overlapping 2.0 s windows (stride 2.0 s)
- Subtract predicted 3-axis error from dead-reckoned endpoints: $\mathbf{p}_{\text{corr}} = \mathbf{p}_{\text{DR}} - \hat{\mathbf{e}}$

| Sequence | Steps | Endpoint MAE (cm) | Final ATE (m) | ATE Reduction |
|---|---|---|---|---|
| MH_01_easy | 90 | 121.5 → 6.9 | 87.09 → **1.66** | **98.0%** |
| MH_02_easy | 74 | 120.0 → 5.3 | 72.09 → **0.57** | **99.0%** |

- Unit tests (`test_correction.py`) prove zero-invariance, perfect-vector zero-error, and >80% empirical reduction.
- Single command reproduction: `bash run_phase2.sh`.

