# R4 literature-review draft: sensor fusion and physics-informed learning

The R4 question is how to combine three kinds of knowledge about one 2 s IMU
window: a physics model (the drift that a calibrated bias would cause), the
104 hand-made Phase 1 features, and what a network learns from the raw IMU
and camera. The two easy Machine Hall flights give 1,312 windows, with 450
inner-training windows per fold. The papers below explain the design choices
in `src/vio_drift/models/dl/hybrid.py`. Their reported accuracies come from
much larger data and different targets, so they are not baselines we can quote.

> Links: check each DOI before the IEEE report is frozen (R1 integrates the bib).

## Physics prior + learned residual (H2, E4)

Karniadakis et al. survey physics-informed machine learning. They name three
ways to put physics into a model: through the data, through the architecture
(inductive bias), or through the loss. H2 is the architecture route. The
physics answer `a * phys_mag + b` is a fixed part of the forward pass, and the
network only adds a correction. With 450 training windows this matters: the
network does not have to rediscover what the integrator already knows.
[Survey](https://doi.org/10.1038/s42254-021-00314-5).

Willard et al. review the same idea for engineering and environmental
systems, where a process model is combined with a learned "residual model"
that predicts the process model's error. That is H2's design. Their review
also notes the risk: if the residual has little structure, the learned part
adds variance without lowering bias. Our early stopping can therefore keep
the epoch-0 (physics-only) checkpoint.
[Survey](https://doi.org/10.1145/3514228).

Zeng et al. (TossingBot) predict a throwing velocity from a ballistic model
and let a network learn only a residual on top of it. Johannink et al. do the
same in control, adding a learned policy to a hand-designed controller. Both
report that residual learning is more data-efficient and degrades gracefully
to the analytic answer. H2 copies the mechanism directly: the head's last
layer is initialised to zero, so the untrained model *is* the physics answer,
and training can only move away from it if validation improves.
[TossingBot](https://doi.org/10.1109/TRO.2020.2988642),
[Residual RL](https://doi.org/10.1109/ICRA.2019.8794127).

## Learning to correct inertial integration

Brossard, Barrau and Bonnabel (AI-IMU) keep a classical invariant Kalman
filter for dead reckoning and train a small CNN only to set its noise
parameters. Brossard, Bonnabel and Barrau also denoise gyroscope
measurements with a CNN before standard integration. Both keep the physics
integrator and learn a correction, rather than regressing pose end-to-end.
This is the same split of work as H2 (physics prior, learned residual) and the
reason we did not try to make the network replace the integrator.
[AI-IMU](https://doi.org/10.1109/TIV.2020.2980758),
[Gyro denoising](https://doi.org/10.1109/LRA.2020.3003256).

## Fusing learned and hand-made features (H1, E3) and modalities (H3, E6)

Baltrušaitis, Ahuja and Morency classify multimodal fusion as early (join raw
inputs), late (join decisions) or intermediate (join learned
representations). All R4 hybrids use intermediate fusion. Each branch (IMU
LSTM, feature MLP, camera CNN) gives a 64-number embedding, the embeddings are
concatenated, and one fusion layer mixes them. This keeps each branch
swappable (the Section 6 contract) and lets one flag turn a branch off for the
ablation. [Survey](https://doi.org/10.1109/TPAMI.2018.2798607).

Clark et al. (VINet) fuse a CNN optical-flow encoder and an LSTM IMU encoder
by concatenating their features before a recurrent core. H3 follows that
pattern at a much smaller scale: R3's 2D CNN on three 64 × 64 frames plus R2's
LSTM. VINet trains on long sequences with full pose labels. With 450 windows
from one flight, our camera branch can memorise scene appearance, so H3 is
judged against its IMU-only and features-only ablations, not against VINet.
[Primary paper](https://doi.org/10.1609/aaai.v31i1.11215).

## How this shapes our claims

1. A hybrid is "better" only if it beats **physics alone** (R² 0.82 mean over
   folds), not just Lasso, because the physics feature is the strongest single
   predictor we have.
2. Differences are judged against the 3-seed spread.
3. Physics-only is a valid result for H2. It means the network found no
   correction that held up on the validation block.
