# R2 presentation content (three slides)

## Slide 1 — Two ways to read one IMU window

- Input is the normalised window, shape (6, 400). Both models emit a 64-number embedding and the same linear head.
- E1, 1D-CNN: kernels 7, 5, 3. Receptive field 13 samples, 65 ms. Global average pooling summarises local detections over the full 2 s.
- E2, LSTM: one layer, hidden size 64. The embedding is the hidden state after step 400.
- Parameter counts: CNN 24,417; LSTM 18,497. E1 and E2 train the magnitude head only.

Speaker notes: A convolution is indifferent to where a short spike occurs. The LSTM is the model that can still depend on the first sample at the last step. RoNIN compared these families separately; we do the same. The 65 ms field is a limitation to say out loud, not a bug in the slide.

## Slide 2 — One trainer, validation only

- AdamW, learning rate 0.001, weight decay 0.0001, dropout 0.3, seed 42.
- Plateau schedule halves the learning rate. Early stopping restores the best validation epoch (patience 15).
- Batch-norm statistics are refreshed on the training windows with dropout off, then the checkpoint is saved.
- Test flight is untouched until that checkpoint is scored. Files: `models/dl/`, `results/metrics/*_loss.csv`, `results/predictions/E1_*.csv` and `E2_*.csv`.

Speaker notes: The training curve for the CNN is the loss with dropout on. Validation is the loss of the model we actually save. Do not compare those two numbers as if they were the same measurement.

## Slide 3 — What the two flights did

- LSTM test MAE 3.52 cm and 3.53 cm. Mean baseline is 4.53 cm. Lasso is 2.50 cm.
- LSTM $R^2$ is 0.382 on MH_01 and 0.627 on MH_02. Lasso's mean $R^2$ is 0.756.
- CNN test MAE is 10.2 cm and 17.5 cm, with negative $R^2$. Training loss keeps falling after validation has stopped improving. Early stopping holds the early epoch. Validation does not diverge.
- Neither network beats the hand-made features. The LSTM is the one that clears the mean baseline.

Speaker notes: Say the negative CNN $R^2$ plainly. With 450 to 553 training windows, a local convolutional summary memorises the early part of the flight. The result is the comparison, not a failure to finish the runs.
