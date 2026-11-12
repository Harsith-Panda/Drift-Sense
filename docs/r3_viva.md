# R3 viva preparation

1. **What does R3 own?** Timestamp-aligned camera preprocessing, ten image
   summary features, the 64-dimensional camera encoder, Tier 1 Lasso checks
   and camera-only E5. IMU fusion and trajectory correction remain R4/R5.
2. **How do images match IMU windows?** Use existing integer-nanosecond
   `[t_start_ns, t_end_ns)` intervals. Select first, nearest midpoint and
   last in-window frames. Preserve CSV order and reject uncovered intervals.
3. **What did verification find?** 3,682 MH_01 and 3,040 MH_02 frames,
   approximately 20 Hz, all images readable at 752 × 480, 40 frames/window.
4. **Why cache frames?** Training loads small uint8 arrays instead of thousands
   of large PNGs. Cache checksums detect accidental changes. Teammates only
   need processed arrays/CSVs/manifests for training.
5. **Are the channels RGB?** No. They are three grayscale times. The network
   may learn visual changes, but its embedding is not guaranteed to be motion.
6. **What is optical flow here?** Farneback estimates dense adjacent-frame
   displacement. Dividing by the measured interval gives resized pixels/s.
   Positive x is right; positive y is down. It is not metric velocity.
7. **What are the ten features?** Flow magnitude mean/std/mean per-pair p90;
   dx/dy mean and std; Laplacian-variance mean/std; mean corner count.
8. **Could feature extraction leak labels?** It reads only window keys and
   timestamps, camera metadata and images. Tests change/poison label columns
   without changing outputs. Only within-window frame pairs are aggregated.
9. **How is Tier 1 fair?** Both variants call Phase 1's unchanged Lasso tuning:
   the same outer folds, alpha grid, five purged training blocks and median
   rule. Scalers fit inside CV pipelines. Both refit on the training flight.
10. **Did flow help Lasso?** No. Equal-fold MAE increased from 2.50 to 3.63 cm.
    One training-flight CV run chose alpha 0.1, giving a constant predictor.
    We do not tune using the test flight to hide this result.
11. **What is the CNN architecture?** Three stride-2 3 × 3 convolutions,
    16/32/64 channels, four-group GroupNorm, ReLU and Dropout2d, then global
    average pooling. Encoder: 23,808 parameters; magnitude head: 65.
12. **Why GroupNorm?** It uses each example's channel groups and needs no
    batch running statistics. This is convenient for small batches and
    differing flights, but does not guarantee generalisation.
13. **What is the receptive field?** 3, then 7, then 15 input pixels along
    each dimension. Global pooling summarises all final-map positions.
14. **How is E5 trained?** MSE/AdamW, lr 0.001, weight decay 0.0001, dropout
    0.3, up to 100 epochs. Validation plateau schedule and early stopping
    (patience 15, minimum improvement 1e-5), seeds 42/43/44.
15. **What prevents test leakage?** R1's training/validation gap is 4 s.
    Only validation chooses the checkpoint; the runner rejects a test
    dataset as validation. Test tensors are evaluated after restoring weights.
16. **What did E5 achieve?** MH_01: 3.79 ± 0.11 cm; MH_02: 5.05 ± 0.21 cm.
    Equal-fold MAE is 4.42 ± 0.12 cm across seeds. R² is weak (mean about
    0.074). It does not beat the 2.50 cm Lasso reference.
17. **Does this prove camera fusion fails?** No. An image embedding might
    supply complementary information even if its standalone model is weak.
    E6/A3 must test the combined model under the same protocol.
18. **Why do Lasso and E5 train on different counts?** Phase 1 refits Lasso
    on the whole training flight after CV. E5 keeps an inner validation
    block and purge. This difference is explicitly reported.
19. **Are vector outputs trained?** No. The shared head supports four outputs,
    but E5 trains only `err_mag_m`. Vector training and correction are later work.
20. **What was changed in teammates' work?** No R1/R2 production code or
    artifacts. With approval, one existing test's missing-camera assertion
    moved to its temporary fixture, since real camera files now exist.
