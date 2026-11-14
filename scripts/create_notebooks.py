#!/usr/bin/env python3
"""Create R5 Jupyter notebooks."""
import json
from pathlib import Path

root = Path("/Users/pranavimathur/Desktop/AML+DL project/Drift-Sense")
nb_dir = root / "notebooks"
nb_dir.mkdir(exist_ok=True)

def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source],
    }

def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.2"},
        },
        "cells": cells,
    }

# ── Notebook 11: Ablation Studies ──────────────────────────────────────────
nb11_cells = [
    cell("# Notebook 11: Ablation Studies\n\n**R5 — Evaluation, Ablations and Drift Correction**\n\nThis notebook presents the full ablation study for the Drift-Sense VIO project.\nAll numbers are reproduced from pre-computed CSVs in `results/ablations/`.", "markdown"),

    cell([
        "import sys\n",
        "from pathlib import Path\n",
        "root = Path('.').resolve().parent\n",
        "sys.path.insert(0, str(root / 'src'))\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.patches as mpatches\n",
        "\n",
        "RESULTS = root / 'results' / 'ablations'\n",
        "FIGURES = root / 'figures' / 'dl'\n",
        "FIGURES.mkdir(parents=True, exist_ok=True)\n",
        "print('Root:', root)\n",
        "pd.set_option('display.float_format', '{:.4f}'.format)\n",
    ]),

    cell("## Section 1: Phase 1 Comparability Check\n\nWe first verify that the new harness reproduces the Phase 1 benchmark numbers exactly.\nThe test suite (`tests/test_dl_eval.py`) already enforces:\n- Mean baseline: 4.53 cm\n- Lasso: 2.50 cm (MAE), R² = 0.76\n- Physics-only: R² = 0.82", "markdown"),

    cell([
        "from vio_drift.evaluation.dl_eval import reproduce_phase1_benchmarks\n",
        "benchmarks = reproduce_phase1_benchmarks()\n",
        "print('=== Phase 1 Benchmark Reproduction ===')\n",
        "for name, metrics in benchmarks.items():\n",
        "    print(f\"  {name}: MAE={metrics['mae_cm']:.4f} cm, R²={metrics['r2']:.4f}\")\n",
    ]),

    cell("## Section 2: Core Experiments E0–E7\n\nComparison of all model families against the Phase 1 Lasso baseline.\n- **E0**: Phase 1 Lasso (anchor)\n- **E1**: 1D-CNN (IMU-only)\n- **E2**: LSTM (IMU-only)\n- **E3**: Hybrid H1 (IMU-Net + hand-crafted + physics features)\n- **E4**: Hybrid H2 (physics prior + residual network) ← **best**\n- **E5**: Camera-only CNN\n- **E6**: Hybrid H3 (IMU + Camera)\n- **E7**: Best overall hybrid", "markdown"),

    cell([
        "df_e = pd.read_csv(RESULTS / 'experiments_e0_e7.csv')\n",
        "display_cols = ['experiment_id','name','family','mae_mean_cm','mae_std_cm','rmse_mean_cm','r2_mean','bias_mean_cm']\n",
        "print(df_e[display_cols].to_string(index=False))\n",
    ]),

    cell([
        "LASSO_MAE = 2.5044\n",
        "COL = {\n",
        "    'Phase 1 (ML / Physics)': '#2196F3',\n",
        "    'DL (IMU-only)': '#FF9800',\n",
        "    'Hybrid': '#4CAF50',\n",
        "}\n",
        "fig, ax = plt.subplots(figsize=(12, 5))\n",
        "colors = [COL.get(f, '#9E9E9E') for f in df_e['family']]\n",
        "bars = ax.bar(df_e['experiment_id'], df_e['mae_mean_cm'],\n",
        "              yerr=df_e['mae_std_cm'], capsize=4,\n",
        "              color=colors, edgecolor='k', linewidth=0.6, zorder=3)\n",
        "ax.axhline(LASSO_MAE, color='#E53935', linestyle='--', linewidth=1.4,\n",
        "           label=f'Lasso baseline {LASSO_MAE:.2f} cm')\n",
        "for bar, mae, std in zip(bars, df_e['mae_mean_cm'], df_e['mae_std_cm']):\n",
        "    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+std+0.4,\n",
        "            f'{mae:.1f}', ha='center', va='bottom', fontsize=8)\n",
        "patches = [mpatches.Patch(color=v, label=k) for k,v in COL.items()]\n",
        "ax.legend(handles=patches+[plt.Line2D([0],[0],color='#E53935',linestyle='--',label=f'Lasso {LASSO_MAE:.2f} cm')], fontsize=8)\n",
        "ax.set_xlabel('Experiment'); ax.set_ylabel('MAE (cm)')\n",
        "ax.set_title('E0-E7: Experiment Comparison (MAE ± std, 3 seeds)')\n",
        "ax.set_ylim(0, df_e['mae_mean_cm'].max()*1.25)\n",
        "ax.grid(axis='y', alpha=0.4, zorder=0)\n",
        "plt.tight_layout(); plt.show()\n",
        "print(f\"\\nKey finding: E4 (Hybrid H2) MAE={df_e[df_e.experiment_id=='E4']['mae_mean_cm'].values[0]:.2f} cm < Lasso {LASSO_MAE:.2f} cm\")\n",
    ]),

    cell("## Section 3: A1 — Regularisation Ablation\n\nEffect of dropout, weight-decay, and early stopping on Hybrid H2.", "markdown"),

    cell([
        "df_a1 = pd.read_csv(RESULTS / 'regularization_a1.csv')\n",
        "print(df_a1[['variation','mae_mean_cm','mae_std_cm','rmse_mean_cm','r2_mean']].to_string(index=False))\n",
    ]),

    cell([
        "fig, ax = plt.subplots(figsize=(8, 4))\n",
        "x = range(len(df_a1))\n",
        "ax.bar(x, df_a1['mae_mean_cm'], yerr=df_a1['mae_std_cm'], capsize=4,\n",
        "       color='#26A69A', edgecolor='k', linewidth=0.6, zorder=3)\n",
        "ax.set_xticks(list(x))\n",
        "ax.set_xticklabels([str(v).replace('_',chr(10)) for v in df_a1['variation']], fontsize=9)\n",
        "ax.set_ylabel('MAE (cm)'); ax.set_title('A1: Regularisation Ablation')\n",
        "ax.grid(axis='y', alpha=0.4, zorder=0); plt.tight_layout(); plt.show()\n",
    ]),

    cell("## Section 4: A2 — Data Scaling\n\nMAE and R² as a function of training data fraction (50%, 75%, 100%).", "markdown"),

    cell([
        "df_a2 = pd.read_csv(RESULTS / 'data_scaling_a2.csv')\n",
        "print(df_a2.to_string(index=False))\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))\n",
        "ax1.errorbar(df_a2['fraction']*100, df_a2['mae_mean_cm'], yerr=df_a2['mae_std_cm'],\n",
        "             marker='o', color='#1976D2', linewidth=1.8, capsize=4)\n",
        "ax1.axhline(LASSO_MAE, color='#E53935', linestyle='--', label=f'Lasso {LASSO_MAE:.2f}')\n",
        "ax1.set_xlabel('Training data (%)'); ax1.set_ylabel('MAE (cm)')\n",
        "ax1.set_title('A2: MAE vs Training Size'); ax1.legend(fontsize=8); ax1.grid(alpha=0.4)\n",
        "ax2.plot(df_a2['fraction']*100, df_a2['r2_mean'], marker='s', color='#388E3C', linewidth=1.8)\n",
        "ax2.set_xlabel('Training data (%)'); ax2.set_ylabel('R²')\n",
        "ax2.set_title('A2: R² vs Training Size'); ax2.grid(alpha=0.4)\n",
        "plt.tight_layout(); plt.show()\n",
    ]),

    cell("## Section 5: A3 — Component Contributions\n\nAbsolving each component of the Hybrid H2 model (physics prior, residual net, etc.).", "markdown"),

    cell([
        "df_a3 = pd.read_csv(RESULTS / 'components_a3.csv')\n",
        "print(df_a3.to_string(index=False))\n",
    ]),

    cell("## Section 6: A4 — Seed Spread\n\nMAE ± std across seeds 42, 123, 999 for all major models. Confirms reproducibility.", "markdown"),

    cell([
        "df_a4 = pd.read_csv(RESULTS / 'seeds_a4.csv')\n",
        "print(df_a4[['model_id','n_seeds','mae_mean_cm','mae_std_cm','mae_min_cm','mae_max_cm','r2_mean']].to_string(index=False))\n",
        "import numpy as np\n",
        "x = np.arange(len(df_a4))\n",
        "fig, ax = plt.subplots(figsize=(9, 4))\n",
        "ax.bar(x, df_a4['mae_mean_cm'], yerr=df_a4['mae_std_cm'], capsize=5,\n",
        "       color='#5C6BC0', edgecolor='k', linewidth=0.6, zorder=3)\n",
        "ax.axhline(LASSO_MAE, color='#E53935', linestyle='--', linewidth=1.4, label=f'Lasso {LASSO_MAE:.2f} cm')\n",
        "for bar, mae, std in zip(ax.patches, df_a4['mae_mean_cm'], df_a4['mae_std_cm']):\n",
        "    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+std+0.3,\n",
        "            f'{mae:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=7)\n",
        "ax.set_xticks(list(x)); ax.set_xticklabels(df_a4['model_id'])\n",
        "ax.set_ylabel('MAE (cm)'); ax.set_title('A4: Seed Spread (3 seeds each)')\n",
        "ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.4); plt.tight_layout(); plt.show()\n",
    ]),

    cell("## Section 7: A5 — Take-off vs Steady-state Segments\n\nMAE split between the first 20% (take-off) and remaining 80% (steady flight) of each sequence.\nPhase 1 required this segmented evaluation for comparability.", "markdown"),

    cell([
        "df_a5 = pd.read_csv(RESULTS / 'segments_a5.csv')\n",
        "print(df_a5.to_string(index=False))\n",
        "seqs = df_a5['flight'].unique()\n",
        "fig, axes = plt.subplots(1, len(seqs), figsize=(12, 5), sharey=False)\n",
        "if len(seqs) == 1: axes = [axes]\n",
        "for ax, seq in zip(axes, seqs):\n",
        "    sub = df_a5[df_a5['flight']==seq]\n",
        "    x2 = np.arange(len(sub)); w=0.35\n",
        "    ax.bar(x2-w/2, sub['takeoff_mae_cm'], w, label='Take-off (first 20%)', color='#EF5350', edgecolor='k', linewidth=0.5)\n",
        "    ax.bar(x2+w/2, sub['steady_mae_cm'],  w, label='Rest (80%)',           color='#42A5F5', edgecolor='k', linewidth=0.5)\n",
        "    ax.set_xticks(list(x2)); ax.set_xticklabels(sub['model'], rotation=15, ha='right', fontsize=8)\n",
        "    ax.set_title(seq); ax.set_ylabel('MAE (cm)'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.4)\n",
        "fig.suptitle('A5: Take-off vs Steady-state MAE'); plt.tight_layout(); plt.show()\n",
    ]),

    cell("## Summary\n\n| Experiment | MAE (cm) | R² | Note |\n|---|---|---|---|\n| E0 Lasso | 2.50 | 0.76 | Phase 1 baseline |\n| E1 1D-CNN | 19.87 | -10.1 | Underfits small data |\n| E2 LSTM | 4.36 | 0.33 | Moderate |\n| E3 Hybrid H1 | 17.05 | -6.73 | — |\n| **E4 Hybrid H2** | **2.30** | **0.79** | **Beats Lasso ✓** |\n| E5 Camera | 16.45 | -8.30 | No real camera data |\n| E6 IMU+Camera | 17.30 | -7.62 | — |\n| E7 Best Hybrid | 2.41 | 0.78 | Ensemble variant |\n\n**Key finding:** Hybrid H2 (physics prior + residual network) achieves 2.30 cm MAE, beating Lasso by 8% with comparable variance.", "markdown"),
]

# ── Notebook 12: Trajectory Correction ─────────────────────────────────────
nb12_cells = [
    cell("# Notebook 12: Trajectory Correction\n\n**R5 — Drift Correction via Predicted Error Subtraction**\n\nThis notebook demonstrates the practical payoff of drift prediction:\nwe subtract the predicted 3-axis error from each window's dead-reckoned endpoint\nand reconstruct corrected flight paths for MH_01_easy and MH_02_easy.", "markdown"),

    cell([
        "import sys\n",
        "from pathlib import Path\n",
        "root = Path('.').resolve().parent\n",
        "sys.path.insert(0, str(root / 'src'))\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "TRAJ_DIR = root / 'results' / 'trajectories'\n",
        "CORR_CSV = root / 'results' / 'ablations' / 'correction_c1.csv'\n",
        "FIG_DIR  = root / 'figures' / 'trajectories'\n",
        "FIG_DIR.mkdir(parents=True, exist_ok=True)\n",
        "print('Trajectory dir:', TRAJ_DIR)\n",
    ]),

    cell("## Section 1: Correction Summary\n\nLoad the correction metrics table from `results/ablations/correction_c1.csv`.", "markdown"),

    cell([
        "df_corr = pd.read_csv(CORR_CSV)\n",
        "cols = ['sequence','windows_count','endpoint_mae_before_cm','endpoint_mae_after_cm',\n",
        "        'endpoint_reduction_pct','cumulative_ate_raw_m','cumulative_ate_corr_m',\n",
        "        'final_drift_raw_m','final_drift_corr_m','ate_reduction_pct']\n",
        "print(df_corr[cols].to_string(index=False))\n",
        "for _, row in df_corr.iterrows():\n",
        "    print(f\"\\n{row['sequence']}:\")\n",
        "    print(f\"  Endpoint MAE: {row['endpoint_mae_before_cm']:.1f} -> {row['endpoint_mae_after_cm']:.1f} cm \"\n",
        "          f\"({row['endpoint_reduction_pct']:.1f}% reduction)\")\n",
        "    print(f\"  Final drift: {row['final_drift_raw_m']:.2f} -> {row['final_drift_corr_m']:.3f} m \"\n",
        "          f\"({row['ate_reduction_pct']:.1f}% ATE reduction)\")\n",
    ]),

    cell("## Section 2: MH_01_easy Trajectory\n\nXY path and ATE over time for sequence MH_01_easy.", "markdown"),

    cell([
        "df1 = pd.read_csv(TRAJ_DIR / 'trajectory_MH_01_easy.csv')\n",
        "print(f'MH_01_easy: {len(df1)} non-overlapping window steps')\n",
        "print(df1[['step','ate_raw','ate_corr']].describe())\n",
    ]),

    cell([
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# XY path\n",
        "ax1.plot(df1['true_x'], df1['true_y'], 'g-', linewidth=2.0, label='Ground truth', zorder=4)\n",
        "ax1.plot(df1['raw_x'],  df1['raw_y'],  'r--', linewidth=1.2, alpha=0.7, label='Dead-reckoned (raw)', zorder=3)\n",
        "ax1.plot(df1['corr_x'], df1['corr_y'], 'b-',  linewidth=1.5, label='Corrected', zorder=4)\n",
        "ax1.scatter([df1['true_x'].iloc[0]], [df1['true_y'].iloc[0]], c='g', s=80, marker='^', zorder=5)\n",
        "ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')\n",
        "ax1.set_title('MH_01_easy: XY Trajectory')\n",
        "ax1.legend(fontsize=9); ax1.grid(alpha=0.4); ax1.set_aspect('equal')\n",
        "\n",
        "# ATE vs step\n",
        "ax2.plot(df1['step'], df1['ate_raw'],  'r-', linewidth=1.5, label='ATE raw',       alpha=0.8)\n",
        "ax2.plot(df1['step'], df1['ate_corr'], 'b-', linewidth=1.5, label='ATE corrected', alpha=0.8)\n",
        "ax2.set_xlabel('Step'); ax2.set_ylabel('ATE (m)')\n",
        "ax2.set_title('MH_01_easy: ATE vs Step')\n",
        "ax2.legend(fontsize=9); ax2.grid(alpha=0.4)\n",
        "plt.suptitle('MH_01_easy Trajectory Correction', fontweight='bold')\n",
        "plt.tight_layout(); plt.show()\n",
        "print(f\"Final ATE raw={df1['ate_raw'].iloc[-1]:.3f}m  corrected={df1['ate_corr'].iloc[-1]:.3f}m\")\n",
    ]),

    cell("## Section 3: MH_02_easy Trajectory", "markdown"),

    cell([
        "df2 = pd.read_csv(TRAJ_DIR / 'trajectory_MH_02_easy.csv')\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
        "ax1.plot(df2['true_x'], df2['true_y'], 'g-', linewidth=2.0, label='Ground truth', zorder=4)\n",
        "ax1.plot(df2['raw_x'],  df2['raw_y'],  'r--', linewidth=1.2, alpha=0.7, label='Dead-reckoned (raw)', zorder=3)\n",
        "ax1.plot(df2['corr_x'], df2['corr_y'], 'b-',  linewidth=1.5, label='Corrected', zorder=4)\n",
        "ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')\n",
        "ax1.set_title('MH_02_easy: XY Trajectory')\n",
        "ax1.legend(fontsize=9); ax1.grid(alpha=0.4); ax1.set_aspect('equal')\n",
        "ax2.plot(df2['step'], df2['ate_raw'],  'r-', linewidth=1.5, label='ATE raw',       alpha=0.8)\n",
        "ax2.plot(df2['step'], df2['ate_corr'], 'b-', linewidth=1.5, label='ATE corrected', alpha=0.8)\n",
        "ax2.set_xlabel('Step'); ax2.set_ylabel('ATE (m)')\n",
        "ax2.set_title('MH_02_easy: ATE vs Step')\n",
        "ax2.legend(fontsize=9); ax2.grid(alpha=0.4)\n",
        "plt.suptitle('MH_02_easy Trajectory Correction', fontweight='bold')\n",
        "plt.tight_layout(); plt.show()\n",
        "print(f\"Final ATE raw={df2['ate_raw'].iloc[-1]:.3f}m  corrected={df2['ate_corr'].iloc[-1]:.3f}m\")\n",
    ]),

    cell("## Section 4: Side-by-side Comparison — ATE over Time\n\nBoth sequences overlaid: raw drift (dashed) vs corrected (solid).", "markdown"),

    cell([
        "fig, ax = plt.subplots(figsize=(11, 5))\n",
        "ax.plot(df1['step'], df1['ate_raw'],  color='#E53935', linewidth=1.4, linestyle='--', label='MH_01 raw')\n",
        "ax.plot(df1['step'], df1['ate_corr'], color='#1E88E5', linewidth=1.8,                 label='MH_01 corrected')\n",
        "ax.plot(df2['step'], df2['ate_raw'],  color='#FB8C00', linewidth=1.4, linestyle='--', label='MH_02 raw')\n",
        "ax.plot(df2['step'], df2['ate_corr'], color='#43A047', linewidth=1.8,                 label='MH_02 corrected')\n",
        "ax.set_xlabel('Window step index'); ax.set_ylabel('ATE (m)')\n",
        "ax.set_title('ATE over Time — Before vs After Drift Correction')\n",
        "ax.legend(fontsize=9); ax.grid(alpha=0.4); plt.tight_layout(); plt.show()\n",
    ]),

    cell("## Section 5: Correction Method Summary\n\n**Method:** For each non-overlapping window (every 8th, 2.0 s stride at 0.25 s step):\n1. Physics prior predicts 3-axis drift vector [Δx, Δy, Δz]\n2. Corrected endpoint = dead-reckoned endpoint − predicted drift\n3. Reconstruct continuous path by chaining corrected endpoints\n\n**Results summary:**\n\n| Sequence | Windows | Endpoint MAE Before | Endpoint MAE After | Reduction | ATE Reduction |\n|---|---|---|---|---|---|\n| MH_01_easy | 90 | 121.5 cm | 6.9 cm | 94.3% | 98.0% |\n| MH_02_easy | 74 | 120.0 cm | 5.3 cm | 95.6% | 99.0% |\n\n**Conclusion:** Drift correction reduces absolute trajectory error by >98%, demonstrating clear practical value.", "markdown"),
]

# Save notebooks
nb11_path = nb_dir / "11_ablation_studies.ipynb"
nb12_path = nb_dir / "12_trajectory_correction.ipynb"

with open(nb11_path, "w") as f:
    json.dump(nb(nb11_cells), f, indent=1)
print(f"Written {nb11_path} ({nb11_path.stat().st_size} bytes)")

with open(nb12_path, "w") as f:
    json.dump(nb(nb12_cells), f, indent=1)
print(f"Written {nb12_path} ({nb12_path.stat().st_size} bytes)")
print("Done.")
