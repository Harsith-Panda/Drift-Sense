# R2 literature-review draft: sequence models for inertial windows

Drift-Sense asks a narrower question than pedestrian inertial odometry. Each
example is one 2 s, 400-step, 6-channel window, and the target is the scalar
endpoint drift `err_mag_m`. The two easy Machine Hall flights supply 1,312
overlapping windows. Papers below motivate the 1D-CNN and the LSTM. Their
reported trajectory errors are not forecasts of our leave-one-flight scores.

Hochreiter and Schmidhuber introduce the LSTM cell so a recurrent net can
keep a state across many steps instead of forgetting it. That is the reason
E2 reads the window in time order and takes the last hidden state: the state
is the only path by which step 1 can still affect the prediction at step 400.
[Primary paper](https://doi.org/10.1162/neco.1997.9.8.1735).

Ordóñez and Roggen stack short temporal convolutions and then an LSTM on
wearable accelerometer and gyroscope streams. The convolutions are local
pattern detectors; the recurrent layer is what carries order. E1 keeps only
the first idea. Three stride-1 kernels of size 7, 5 and 3 see 13 samples,
which is 65 ms at the EuRoC 200 Hz rate. Global average pooling then
summarises those local detections over the whole window, so a spike anywhere
in the 2 s can move the 64-number embedding, without any single filter seeing
the full window. [Primary paper](https://doi.org/10.3390/s16010115).

Chen, Lu, Markham and Trigoni, in IONet, cut inertial streams into windows and
train a recurrent network to regress a window-level motion instead of
integrating raw acceleration without bound. The window formulation is the
part we share. IONet targets polar velocity for pedestrian and wheeled
motion on a large multi-device set. We target a drone's integrated position
error on two flights, so their accuracy is not a baseline we can quote.
[Primary paper](https://doi.org/10.1609/aaai.v32i1.12102).

Herath, Yan and Furukawa compare three backbones on the same inertial task:
a 1D ResNet, a stacked LSTM, and a temporal convolutional net. RoNIN ResNet
reads 200 samples of 6-channel IMU and regresses a 2D velocity; the LSTM
variant is three layers of 100 units. Their comparison is why E1 and E2 are
separate rows rather than one mixed network: convolution and recurrence
answer different questions, and only a shared protocol shows which one
helped. RoNIN is trained on tens of hours from many people. A result there
does not imply that a 24k-parameter CNN will beat 104 hand-made features on
450 training windows. [Primary paper](https://doi.org/10.1109/ICRA40945.2020.9196860).

Liu and colleagues, in TLIO, regress a 3D displacement and an uncertainty
from an IMU segment and fuse that measurement in an extended Kalman filter.
The relevant point for this role is the displacement head on a raw window.
Their filter, uncertainty output and headset corpus are out of scope for E1
and E2. Phase 1 already states that our labels start each window from the
ground-truth state; a learned window error is not yet an online navigator.
[Primary paper](https://doi.org/10.1109/LRA.2020.3007421).

Batch normalisation and dropout are part of each CNN block, and the same
dropout rate is applied to the LSTM's 64-d state, because a one-layer LSTM
does not use the recurrent dropout argument. Ioffe and Szegedy introduce
batch normalisation; Srivastava and colleagues introduce dropout. Weight
decay is AdamW's decoupled decay, following Loshchilov and Hutter, which is
the optimiser recorded in the checkpoints. Whether those three regularisers
change the held-out flight is A1, not a claim inherited from these papers.
[Batch normalisation](https://arxiv.org/abs/1502.03167).
[Dropout](https://www.jmlr.org/papers/v15/srivastava14a.html).
[Decoupled weight decay](https://arxiv.org/abs/1711.05101).

The design question E1 and E2 can answer is whether a local convolutional
summary, or a last-state memory of all 400 steps, predicts `err_mag_m` on
the flight that was not used for training or early stopping. It is not
whether a deep inertial odometry system outperforms RoNIN or IONet.
