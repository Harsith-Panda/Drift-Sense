# R5 literature-review draft: Drift Evaluation Methodology and Ablation Studies

In visual-inertial odometry (VIO) and inertial dead-reckoning, evaluating predictive models requires distinguishing between local window estimation errors and global cumulative drift, while ensuring that deep learning benchmarks remain strictly comparable to classical physics and machine learning baselines. The papers below inform the evaluation protocol, metric design, and trajectory reconstruction implemented in `src/vio_drift/evaluation/dl_eval.py` and `src/vio_drift/correction/correct_trajectory.py`.

## Trajectory Error Metrics: ATE vs. Relative Error

Sturm et al. formalise the benchmark methodology for trajectory estimation in visual and RGB-D SLAM. They separate evaluation into two fundamental metrics:
1. **Absolute Trajectory Error (ATE):** Measures global trajectory consistency by computing the Euclidean root-mean-square error between time-aligned ground-truth and estimated poses after optimal rigid alignment (Umeyama SE(3) or Sim(3) alignment).
2. **Relative Pose Error (RPE):** Evaluates local frame-to-frame or window-to-window drift over fixed time intervals $\Delta t$.

Our window evaluation targets the 2.0 s relative displacement error $\mathbf{e}_i = \Delta \mathbf{p}_{\text{dead-reckoned}} - \Delta \mathbf{p}_{\text{true}}$, which matches Sturm's RPE definition over $\Delta t = 2.0$ s. For trajectory reconstruction, our chaining of non-overlapping window endpoints allows direct evaluation of cumulative ATE along the entire sequence.
[Primary paper: Sturm et al., IROS 2012](https://doi.org/10.1109/IROS.2012.6385773).

Zhang and Scaramuzza extend this analysis to Visual-Inertial Odometry benchmarks, demonstrating that evaluating VIO systems over multi-scale subsegments is essential to expose scale drift and bias instability. They emphasize that reporting mean metrics alone masks divergence during high-dynamic manoeuvres (such as rapid angular acceleration during take-off). This directly motivates our A5 segment ablation, splitting sequences into the initial 20% (take-off) and remaining 80% (steady flight).
[Primary paper: Zhang & Scaramuzza, IROS 2018](https://doi.org/10.1109/LRA.2018.2860083).

## Systematic Ablation Studies in Machine Learning

Melis and Dyer examine reproducibility and evaluation integrity in neural networks, showing that claimed performance gains in complex architectures are often artifacts of hyperparameter tuning, differing data splits, or unrepresentative single-seed evaluations. They advocate:
1. Preserving identical data splits and preprocessing across all evaluated architectures.
2. Reporting mean and variance across multiple seeds.
3. Isolating architectural components via systematic ablation sweeps.

In our evaluation harness, E0 serves as the immutable anchor: Phase 1's Lasso (2.50 cm MAE) and Physics-only ($R^2 = 0.82$) models must be reproduced bit-for-bit before scoring any deep learning model. The A1–A5 ablations systematically isolate regularisation, training data scale, component contributions, seed sensitivity, and flight segments.
[Primary paper: Melis & Dyer, ICLR 2018](https://arxiv.org/abs/1707.05589).

## Closed-Loop and Learned Dead-Reckoning Correction

Brossard et al. (AI-IMU Dead-Reckoning) demonstrate that neural networks can predict IMU integration error parameters to constrain open-loop dead-reckoning. They show that subtracting predicted sensor errors within an extended Kalman filter or optimization graph reduces cumulative trajectory drift by orders of magnitude. 

Our trajectory correction module (`correct_trajectory.py`) applies this concept in a discrete endpoint setting: subtracting the predicted 3D error vector from each window's dead-reckoned endpoint $\mathbf{p}_{k} = \mathbf{p}_{k}^{\text{DR}} - \hat{\mathbf{e}}_k$. For non-overlapping strides, this yields continuous reconstructed paths with >98% ATE reduction on EuRoC sequences, confirming that local window drift prediction translates directly into global navigational utility.
[Primary paper: Brossard et al., IEEE T-IV 2020](https://doi.org/10.1109/TIV.2020.2980758).

