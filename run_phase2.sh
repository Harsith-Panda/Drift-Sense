#!/usr/bin/env bash
# run_phase2.sh — Reproduce all Phase 2 (R5) results in one command.
# Usage: bash run_phase2.sh [--skip-tests]
#
# Prerequisites:
#   - Python 3.14+ at /usr/local/bin/python3
#   - PyTorch 2.x and scikit-learn 1.x installed
#   - Phase 1 data already in data/processed/ (run Phase 1 pipeline first)
#
# Outputs:
#   results/ablations/      — all ablation CSV files
#   results/trajectories/   — trajectory CSVs
#   results/predictions/    — per-seed prediction CSVs
#   figures/dl/             — ablation plots (PNG)
#   figures/trajectories/   — trajectory correction plots (PNG)

set -euo pipefail
cd "$(dirname "$0")"

PYTHON=python3
SKIP_TESTS=0

for arg in "$@"; do
  [[ "$arg" == "--skip-tests" ]] && SKIP_TESTS=1
done

echo "============================================================"
echo "  Drift-Sense Phase 2 Reproduction Script"
echo "  Working directory: $(pwd)"
echo "  Python: $($PYTHON --version)"
echo "============================================================"

# ── Step 0: Sanity check ───────────────────────────────────────────────────
echo ""
echo "[Step 0] Checking data files..."
for f in data/processed/splits.json data/processed/physics_leave_MH_01_easy.csv data/processed/physics_leave_MH_02_easy.csv; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Required file $f not found. Run Phase 1 pipeline first." >&2
    exit 1
  fi
done
echo "  Data OK."

# ── Step 1: Run all experiments and ablations ──────────────────────────────
echo ""
echo "[Step 1] Running ablation experiments (E0-E7, A1-A5, C1)..."
$PYTHON scripts/run_ablations.py
echo "  Ablations done. CSVs written to results/ablations/"

# ── Step 2: Run trajectory correction ─────────────────────────────────────
echo ""
echo "[Step 2] Running trajectory correction..."
PYTHONPATH=src $PYTHON -c "
from vio_drift.correction.correct_trajectory import run_sequence_correction
from pathlib import Path
root = Path('.')
for seq in ['MH_01_easy', 'MH_02_easy']:
    print(f'  Correcting {seq}...')
    df, stats = run_sequence_correction(root, seq)
    print(f'    ATE reduction: {stats[\"ate_reduction_pct\"]:.1f}%')
"
echo "  Trajectories written to results/trajectories/"

# ── Step 3: Generate figures ───────────────────────────────────────────────
echo ""
echo "[Step 3] Generating figures..."
$PYTHON scripts/generate_figures.py
echo "  Figures written to figures/dl/ and figures/trajectories/"

# ── Step 4: Run test suite ─────────────────────────────────────────────────
if [[ $SKIP_TESTS -eq 0 ]]; then
  echo ""
  echo "[Step 4] Running R5 test suite..."
  $PYTHON -m pytest tests/test_dl_eval.py tests/test_correction.py -v --tb=short
  echo "  Tests passed."
else
  echo ""
  echo "[Step 4] Skipping tests (--skip-tests flag set)."
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 2 reproduction complete."
echo ""
echo "  Key results:"
$PYTHON - <<'PYEOF'
import pandas as pd
from pathlib import Path
try:
    df = pd.read_csv('results/ablations/experiments_e0_e7.csv')
    best = df.loc[df['mae_mean_cm'].idxmin()]
    lasso = df[df['experiment_id']=='E0'].iloc[0]
    print(f"    Lasso (E0):        MAE={lasso['mae_mean_cm']:.2f} cm, R²={lasso['r2_mean']:.4f}")
    print(f"    Best DL ({best['experiment_id']}):  MAE={best['mae_mean_cm']:.2f} cm, R²={best['r2_mean']:.4f}  ← {best['name']}")
except Exception as e:
    print(f"    (Could not load results: {e})")
try:
    dc = pd.read_csv('results/ablations/correction_c1.csv')
    for _, r in dc.iterrows():
        print(f"    {r['sequence']}: ATE {r['final_drift_raw_m']:.2f}m -> {r['final_drift_corr_m']:.3f}m ({r['ate_reduction_pct']:.1f}% reduction)")
except Exception as e:
    print(f"    (Correction CSV not found: {e})")
PYEOF
echo ""
echo "  Notebooks: notebooks/11_ablation_studies.ipynb"
echo "             notebooks/12_trajectory_correction.ipynb"
echo "  Decisions: docs/decisions_log.md"
echo "============================================================"
