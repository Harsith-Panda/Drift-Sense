# How the drift label is made (Part A) — plain words

We don't get an "error" column from EuRoC — we build it ourselves. Here's
exactly how, in order:

## 1. Load
For each sequence we read two files:
- `imu0/data.csv` — 200 readings per second of rotation speed (gyro) and
  acceleration (accel).
- `state_groundtruth_estimate0/data.csv` — the *true* position, orientation,
  and velocity, measured by a precise motion-capture system.

## 2. Align
The two files aren't sampled at exactly the same times. For every IMU
reading, we find the closest-in-time ground-truth reading and attach it.
If nothing is close enough (a gap), we drop that reading rather than pair
it with a stale position — `align.py` prints how many rows this affected.

## 3. Window
We cut the aligned stream into ~1-second chunks ("windows"), sliding
forward by 0.25 seconds each time (so windows overlap). Each window is
one training example.

## 4. Integrate (the model-free "prediction")
For each window, we ask: *if all we had was the IMU readings, where would
we think the drone ended up?* Starting from the window's true starting
position/velocity/orientation, we add up ("integrate") the IMU readings
step by step — this mimics what a real VIO system's IMU-only estimate
would drift toward. This is standard textbook physics (Newton's laws +
rotation tracking), not a machine-learning model.

## 5. Compare to truth = the label
We already know the window's *true* ending position from ground truth.
The **drift label** is simply:

```
label = distance( predicted_ending_position, true_ending_position )
```

in metres. We also keep the 3-axis version (error in x, y, z separately)
because Phase 2's correction step needs it, not just the ML models.

That raw-IMU magnitude (`err_mag_m`) is the **headline Phase 1 target**.
It matches the approved problem statement: integrate the IMU readings.

## 5b. Bias-corrected comparison (not a replacement)
The same window is integrated a second time after subtracting the
ground-truth gyro/accel bias estimates that EuRoC already stores
(`gt_bg*`, `gt_ba*`). Those extra columns are named `err_mag_m_bc` and
`err_dx_bc` / `err_dy_bc` / `err_dz_bc`.

This is an **oracle diagnostic**: a live estimator would not be handed
those biases. Features for Part C stay **raw IMU**. Do not treat
`err_*_bc` as the main leaderboard unless the team later votes to.

Whether the residual is more motion-dependent is something Part B
measures — it is not assumed.

## 6. Why two "unknowns" needed calibrating first
Turning IMU readings into "true" world-frame motion needs two physical
choices that vary slightly by convention:
- which way gravity should be added back in, and
- which direction the orientation quaternion rotates.

Instead of guessing, `imu_integration.py` tries all 4 combinations on a
tiny real slice of data and keeps whichever one is most accurate — this
runs automatically the first time you run `make_windows.py`, and prints
what it picked so you can record it in `configs/config.yaml`.

## 7. How to know it worked
`tests/test_label_sanity.py` checks something that must be true physically:
**a longer window should accumulate more error than a shorter one.** If
that test fails, don't trust the labels yet — it almost always means a
column got mis-mapped in `loader.py` (run `scripts/verify_data.py` to
check) rather than a deep bug in the math.

## What Part C and Part D receive
- `data/processed/windows_<seq>.csv`: one row per window —
  `seq_id, window_id, t_start_ns, t_end_ns, err_mag_m, err_dx, err_dy, err_dz,
  err_mag_m_bc, err_dx_bc, err_dy_bc, err_dz_bc`
- `data/processed/windows_raw_<seq>.npy`: the raw IMU readings for each
  window, shape `(n_windows, 200, 6)`, columns `gx,gy,gz,ax,ay,az` — this
  is what Part C turns into features (never bias-corrected).
- `results/diagnostics/window_length.csv`: mean raw vs bias-corrected
  error at 0.25–2.0 s (Phase 2 D5 groundwork). Primary windows stay 1 s.
