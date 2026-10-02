# Drift-Sense

Semester ML project: predict IMU-integration position drift on EuRoC MAV sequences.

## Environment (conda)

This repo does **not** use a checked-in virtualenv. Phase 1 packages are listed in `environment.yml` (conda) and `requirements.txt` (pip fallback).

```bash
conda env create -f environment.yml
conda activate drift-sense
```

Register the kernel so the notebooks in `notebooks/` use this env:

```bash
python -m ipykernel install --user --name drift-sense --display-name "Python (drift-sense)"
```

If you cannot use conda:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run scripts from the repo root, e.g. `python scripts/verify_data.py`.

```bash
python scripts/make_windows.py    # Part A labels
python scripts/run_eda.py         # Part B figures → figures/eda/
```
