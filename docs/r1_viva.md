# R1 viva preparation

1. Why not random windows? Neighboring two-second windows share up to 1.75 seconds;
   flight-level testing and timestamp purging prevent shared samples across partitions.
2. Why four seconds between ends and starts? It ensures an actual unused interval;
   four seconds between starts alone would give a shorter gap.
3. Where are scalers fitted? Explicit inner-training IDs only, separately per fold
   and A2 subset. Validation and test only transform.
4. Why recompute physics? The original outer-training calibration includes validation.
   The DL calibration uses unique rows in the union of inner-training intervals.
5. Does augmentation preserve exact integrated drift? No physical simulation claim is
   made. It corrupts observations while retaining the original latent full-window target.
   A true shortened crop needs new labels and is excluded.
6. Why are auxiliary features unchanged? They remain reference modalities for the
   original window; only the learned IMU branch receives observation corruption.
7. What does camera-off return? No cam key. Camera-on requires R3 arrays and matching IDs.
8. What proves reproducibility? Explicit loader generators, worker seeds, and augmentation
   keyed by fold, sample, seed and epoch. Call set_epoch before each epoch.
9. What can A2 conclude? Generalization with more chronological training coverage;
   count and flight phase are confounded, and only two test flights exist.
10. What remains for teammates? R2 trainer/models and A1 execution, R3 camera artifacts,
    R4 physics-prior integration, R5 scoring/seed spread and A2 execution.
11. Can this run without ground truth? Current labels and physics initialize from GT;
    deployment claims require a different initial-state source and validation.
12. What has been measured? Data contracts and tests, not DL predictive performance.

Buddy review remains pending an actual teammate PR; no external review is claimed.
