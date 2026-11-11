# R1 presentation content (two slides)

## Slide 1 — Reliable data before models

- 1,312 windows, two flights; IMU (6,400), features (104), physics (4).
- Leave one flight out; chronological final 20% validation, 4-second interval gap.
- Fold sizes: 450/119/720 and 553/144/592; 23 purged in each.
- Scalers and physics calibration see only inner training.

Speaker notes: overlapping windows are correlated. A random row split could share
sensor samples between partitions. Physics calibration is also fitted preprocessing.
Ground-truth initial states limit deployment claims. Camera-on awaits R3 arrays.

## Slide 2 — Regularization with reproducible evidence

- Training-only bounded jitter and crop-and-mask; fixed labels and timeline.
- A1: 8 regularization combinations + 3 augmentation probes.
- A2: nested 50/75/100% inner-training prefixes; preprocessing refitted each time.
- Two folds x three seeds; 84 jobs ready, results pending R2/R5.

Speaker notes: crop-and-mask models incomplete observations; it is not a shorter
integration interval. R2 owns dropout and early stopping behavior. R5 owns scoring
and spread. A2 changes flight coverage as well as count. Report this limitation.
