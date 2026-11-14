# Drift-Sense Phase 2 — Team Decisions Log

> **Purpose:** Record every significant design decision made during Phase 2 (R5: Evaluation, Ablations and Drift Correction), so that the viva examiner can follow our reasoning and so future team members can understand the rationale.

---

## D1 — Dataset Scope (2026-11-01)

**Decision:** Keep only MH_01_easy and MH_02_easy for all Phase 2 experiments.

**Rationale:**
- Phase 1 used these two sequences exclusively; changing the dataset would break comparability.
- Together they provide 1,440 windows (720 per sequence), enough to show trends but small enough to train quickly on CPU.
- Leave-one-sequence-out cross-validation gives 2 folds, matching the Phase 1 split in `data/processed/splits.json`.

**Alternatives rejected:** Adding MH_03_medium — rejected because we do not have Phase 1 baselines for it, so the comparison table would be incomplete.

---

## D2 — Camera as Tier 0 (IMU-only Default) (2026-11-01)

**Decision:** The primary pipeline is IMU-only (Tier 0). Camera input is an optional ablation controlled by the `use_camera` flag in `configs/ablations.yaml`.

**Rationale:**
- Real-world VIO systems lose camera signal in dark or texture-less environments; IMU-only robustness matters.
- Camera data (EuRoC `/cam0`) requires heavy preprocessing; using a proxy prevents fair comparison.
- Experiments E5 and E6 explore camera at ablation time only.

**Alternatives rejected:** Making camera the default — rejected because it would obscure the IMU contribution and add preprocessing complexity.

---

## D3 — Primary Target: err_mag_m + 3-axis Head (2026-11-01)

**Decision:** Train models to predict the scalar drift magnitude `err_mag_m` as the primary loss target, and add a 3-axis head (err_x_m, err_y_m, err_z_m) for trajectory correction.

**Rationale:**
- Phase 1 evaluated on `err_mag_m`; using the same target makes Phase 2 results directly comparable.
- The 3-axis head enables `correct_trajectory.py` to subtract the predicted vector from each dead-reckoned endpoint.
- Multi-task loss: L = L_magnitude + 0.3 × L_3axis.

---

## D4 — Framework and Seeds (2026-11-02)

**Decision:** Use PyTorch (CPU-first). Report every DL number as mean ± std over 3 seeds: 42, 123, 999.

**Rationale:**
- PyTorch is the team standard and is available system-wide at Python 3.14.
- 3 seeds give a 95% CI width ≈ 2 std/√3; acceptable for a course project.
- Seed set {42, 123, 999} chosen before seeing results (registered in this log on 2026-11-02).

---

## D5 — Same Folds as Phase 1 (2026-11-02)

**Decision:** All experiments use the same leave-one-sequence-out folds defined in `data/processed/splits.json` as Phase 1.

**Rationale:**
- Guarantees that train/test splits are identical, so any MAE difference is due to the model, not the split.
- The evaluation harness (`dl_eval.py`) loads `splits.json` directly and raises an error if it is missing.

---

## D6 — Hybrid H2 as Main Architecture (E4) (2026-11-03)

**Decision:** The "best DL model" for reporting is Hybrid H2: a physics-prior branch concatenated with a residual neural network that predicts the correction to the physics estimate.

**Rationale:**
- Physics prior alone achieves R² = 0.82 (Phase 1); the residual net reduces bias while keeping low variance.
- E4 achieved MAE = 2.30 cm, which beats Lasso (2.50 cm) — the first time a neural model surpasses the Phase 1 ML baseline on this dataset.
- Architecture: physics_output + MLP_residual(physics_features ‖ IMU_embedding).

**Alternatives evaluated:** H1 (feature concatenation, E3) — worse than Lasso; pure LSTM (E2) — 4.36 cm.

---

## D7 — Non-overlapping Windows for Trajectory (2026-11-03)

**Decision:** Select every 8th window (2.0 s stride at 0.25 s step) for trajectory reconstruction.

**Rationale:**
- Overlapping windows cannot be chained into a path without double-counting 1.75 s of displacement.
- The 8th-window selection is deterministic and reproducible.
- This yields 90 and 74 steps for MH_01_easy and MH_02_easy respectively.

---

## D8 — Evaluation Harness Proves Comparability (2026-11-04)

**Decision:** `dl_eval.py:reproduce_phase1_benchmarks()` must return Lasso MAE = 2.504 cm (tolerance ±0.01 cm) and physics-only R² = 0.820 (±0.002). These checks are enforced by `tests/test_dl_eval.py`.

**Rationale:**
- Without bit-exact reproduction of Phase 1 numbers, any "improvement" over Lasso could be explained by a different data split or preprocessing.
- Tests run in CI before every ablation report is generated.

---

## D9 — run_ablations.py as Single Entry Point (2026-11-05)

**Decision:** All ablation experiments (E0–E7, A1–A5, C1) are defined in `configs/ablations.yaml` and executed by `scripts/run_ablations.py` in a single pass.

**Rationale:**
- Reproducibility: one script, deterministic order, fixed seeds.
- Output CSVs in `results/ablations/` are version-controlled (small text files).
- `run_phase2.sh` calls this script as the first step.

---

## D10 — Correction Metric: Endpoint MAE and ATE (2026-11-05)

**Decision:** Report two correction metrics: (1) per-window endpoint MAE before/after, (2) cumulative ATE along the reconstructed path.

**Rationale:**
- Endpoint MAE shows the per-window correction effect independently of path reconstruction.
- ATE is the standard VIO/SLAM metric; it rewards globally consistent trajectories.
- Both metrics are computed by `correct_trajectory.py:compute_endpoint_error_comparison()`.

---

## References

- EuRoC MAV dataset: Burri et al., "The EuRoC micro aerial vehicle datasets", IJRR 2016.
- Drift evaluation methodology: Sturm et al., "A benchmark for the evaluation of RGB-D SLAM systems", IROS 2012.
- Ablation study design: Melis & Dyer, "On the state of the art of evaluation in neural language models", ICLR 2018.
- VIO drift correction: Forster et al., "IMU preintegration on manifold for efficient visual-inertial maximum-a-posteriori estimation", RSS 2015.
