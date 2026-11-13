# R4 viva notes: hybrid models

**Q: What is a hybrid here?**
A model that combines learned embeddings (LSTM on raw IMU, CNN on frames)
with hand-made knowledge (104 Phase 1 features, the physics feature). All
branches output 64 numbers, are concatenated, mixed by one layer, and go to
R2's shared regression head (intermediate fusion).

**Q: Why is H2 the main hybrid?**
Physics alone is the strongest single predictor (mean R² 0.82, better than
Lasso's 0.76). H2 keeps it as a fixed part of the forward pass,
`ŷ = a·phys_mag + b + s·z`, and only learns the correction `z`. With 450
training windows the network does not have to rediscover what the
integrator already knows (residual physics: TossingBot, residual RL).

**Q: How does H2 start exactly at the physics answer?**
The regression head's last linear layer (weight and bias) is initialised to
zero, so `z = 0` before training. `tests/test_hybrid.py` checks this for
both the 1- and 4-value heads. Early stopping may keep epoch 0, which means
"the network found nothing that holds up on validation".

**Q: Where do a, b and the scales come from? Any leakage?**
`fit_scaling` uses only the inner-training split (it raises an error on
any other split). That covers least squares for `a, b`, the physics
mean/std, and the target scale. The physics table itself is R1's fold-specific
`physics_dl` calibration. The test flight is used only for the final
prediction, and E7's base model was chosen on validation MAE.

**Q: Why standardise the target?**
`err_mag_m` is about 1.2 m with a spread of only about 6 cm. A head that
starts at 0 spends its first epochs learning the offset. The networks
predict standardised values, and `ȳ + s·z` converts them back to metres.
This is the main reason R2's E1 CNN failed (R² −8.55). The same CNN
encoder with a standardised target (IMUcnn, 3 seeds) reaches R² 0.72.
My trainer also clips gradients and adds one fusion layer, so scaling is
the main, but not the only, difference.

**Q: Did the hybrid work?**
H2 is the best Phase 2 model: mean 2.15 cm and R² 0.82 with the LSTM
encoder, 2.13 cm and 0.82 with the CNN encoder. Phase 1 Lasso gets 2.50 cm
and 0.76. It ties with physics alone (2.11 cm, 0.82), which is at the 0.83
ceiling. H1 (2.64 cm with warm-up, 3.15 cm without) and H3/camera (2.85 /
3.52 cm) are worse. The flow numbers (E7) change nothing beyond the seed
spread.

**Q: Where is the CNN?**
R2's 1D-CNN plugs into the same contract. As a raw-IMU-only model with a
standardised target it reaches R² 0.72 (2.60 cm), close to Lasso on 104
hand-made features. As H2's encoder (E4cnn) it gives 2.13 cm, the same as
with the LSTM. The physics backbone, not the encoder, decides the result.

**Q: Isn't the 10-epoch warm-up tuning on the test set?**
No. It was added after seeing that direct models kept their epoch-1
checkpoint. The value 10 comes from where the *training* curves flatten.
Both the original and warm-up rows are reported; the warm-up rows are a
sensitivity analysis. Validation MAE got worse with the warm-up while test
got better, which is evidence that the landing validation block mis-ranks
checkpoints.

**Q: Is your DL model better than ML? Isn't comparing with physics-free Lasso unfair?**
Yes, that comparison is unfair, so I added ML references under the Phase 2 rules: same
inner-training windows, R1's `physics_dl` numbers, and alpha chosen on the same validation
block (`src/vio_drift/models/dl/ml_reference.py`). Lasso on the 104 features gives 3.59 cm.
Adding the 4 physics numbers gives 3.27 cm, so the physics features do carry extra
information. H2 gives 2.15 cm. The gap comes from the selection rule. The landing
validation block favours a big Lasso penalty, so Lasso zeroes everything (MH_02, no physics)
or keeps only `phys_mag` with a shrunk weight. H2's physics part is plain least squares, so
it is not exposed to that. With Phase 1's time-block tuning, Lasso + physics reaches about
2.15 cm, a tie. So H2 is more robust, not more informative.

**Q: Why couldn't the network improve on physics?**
1. The validation block is the last 20% of the training flight (landing):
   label std 2.2–3.6 cm vs 6.3–8.8 cm in training, and physics R² ≈ 0 there.
   Checkpoint selection rests on a block with almost no signal.
2. The residual it learns helps mid-flight but hurts take-off on MH_02
   (2.7 vs 2.05 cm), a motion type the validation block never contains.
3. There are only 450 training windows from one flight per fold.

**Q: Is a difference real or noise?**
Compare it with the seed std. The E4 seed std is 0.015–0.018 cm per fold.
The MH_01 gain (0.01 cm) is noise; the MH_02 loss (0.10 cm) is larger than
the spread, so physics really is better there.

**Q: What does the camera add?**
Variance, not accuracy. H3's seed std is up to 0.52 cm, and its MAE is
worse than IMU-only H1 on MH_02. This matches R3's E5 (camera alone ≈ mean
baseline) and the Tier 1 Lasso check.

**Q: What does R5 get from you?**
Prediction files `results/predictions/<exp>_<fold>.csv`, with seeds 43/44 as
`<exp>_seed<s>_<fold>.csv`; `PHYS_<fold>.csv`; and `E4vec_*` files with
`y_pred_dx/dy/dz` for drift correction. Physics alone reduces the mean error
vector from about 121 cm to 5.5–6.6 cm; E4vec gives 5.7–6.4 cm.

**Q: How is a missing camera handled?**
Its branch contributes a zero embedding, so the other branches still
predict (tested).
