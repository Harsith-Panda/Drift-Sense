# R1 literature-review draft: sensor augmentation and regularization

Small inertial datasets require validation that separates correlated observations.
Here, 1,312 overlapping windows come from only two flights. Their count does not
represent 1,312 independent trajectories. We therefore preserve flight-level test
folds and purge a four-second interval before the chronological validation block.
This is our experimental design, not a claimed result of the papers below.

Iwana and Uchida evaluate twelve augmentation methods across 128 classification
datasets and six neural architectures. Their comparison motivates testing each
transformation rather than assuming universal benefit. Classification evidence
does not establish that a transformation preserves continuous drift labels.
[Primary paper](https://arxiv.org/abs/2007.15951).

Wen et al. review augmentation across classification, forecasting and anomaly
detection, emphasizing task dependence. For Drift-Sense we infer that shortening
an integration interval cannot reuse the original endpoint-error label. Our
random-crop switch therefore masks at most four edge samples while retaining
all 400 positions, timestamps and the original target. It models missing
observations rather than changing physical motion. Small clipped jitter models
observation uncertainty; neither perturbation is claimed to preserve the exact
numerically integrated trajectory. Both are disabled by default and require A1
validation. [Primary paper](https://www.ijcai.org/proceedings/2021/0631.pdf).

Srivastava et al. introduce dropout as random removal of units during training.
Our A1 includes dropout 0 versus 0.3, with the implementation owned by R2.
Results from their tasks do not predict improvement on these flights.
[Primary paper](https://www.jmlr.org/papers/v15/srivastava14a.html).

Loshchilov and Hutter distinguish decoupled weight decay from an L2 penalty in
adaptive optimizers. R2 should record the optimizer and its decay semantics;
the configuration alone does not determine them. A1 crosses decay 0 versus
1e-4 with dropout and early stopping, keeping data and seeds fixed.
[Primary paper](https://arxiv.org/abs/1711.05101).

The resulting research question is whether bounded observation corruption and
model regularization improve held-out-flight generalization. No model results
are available yet. A2 changes only a nested chronological prefix of eligible
inner training, refitting every learned preprocessing statistic. Its limitation
is that training quantity and flight phase change together; this is not a causal
estimate of sample count alone. R5 should report three-seed spread and this caveat.
