"""
Tests for Phase 2 evaluation harness (dl_eval.py).

Verifies that the new harness accurately reproduces Phase 1 benchmarks
(Mean baseline 4.53 cm, Lasso 2.50 cm, and Physics-only R^2 0.82) to ensure
rigorous comparability across phases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vio_drift.evaluation.dl_eval import (
    aggregate_seed_metrics,
    build_unified_comparison_table,
    reproduce_phase1_benchmarks,
    score_metrics,
    score_prediction_dataframe,
    score_segments,
)


def test_score_metrics_exact() -> None:
    """Check mathematical accuracy of metric calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    m = score_metrics(y_true, y_pred)
    assert pytest.approx(m["mae"], abs=1e-5) == 0.15
    assert pytest.approx(m["bias"], abs=1e-5) == 0.0
    assert pytest.approx(m["rmse"], abs=1e-4) == np.sqrt(0.025)
    assert m["r2"] > 0.9


def test_reproduce_phase1_mean_baseline() -> None:
    """Mean baseline across folds must reproduce Phase 1's 4.53 cm (0.0453 m)."""
    benchmarks = reproduce_phase1_benchmarks()
    mean_row = benchmarks[(benchmarks["model"] == "mean") & (benchmarks["fold_id"] == "mean")]
    assert len(mean_row) == 1
    mae = mean_row["mae"].values[0]
    assert pytest.approx(mae, abs=1e-4) == 0.04525
    assert pytest.approx(mae * 100, abs=0.02) == 4.53


def test_reproduce_phase1_lasso_2_50_cm() -> None:
    """Lasso across folds must reproduce Phase 1's 2.50 cm headline MAE (0.02504 m)."""
    benchmarks = reproduce_phase1_benchmarks()
    lasso_row = benchmarks[(benchmarks["model"] == "lasso") & (benchmarks["fold_id"] == "mean")]
    assert len(lasso_row) == 1
    mae = lasso_row["mae"].values[0]
    r2 = lasso_row["r2"].values[0]

    assert pytest.approx(mae, abs=1e-4) == 0.02504
    assert pytest.approx(mae * 100, abs=0.02) == 2.50
    assert pytest.approx(r2, abs=0.01) == 0.76


def test_reproduce_phase1_physics_only_r2_0_82() -> None:
    """Physics-only feature across folds must reproduce R^2 ≈ 0.82."""
    benchmarks = reproduce_phase1_benchmarks()
    phys_row = benchmarks[(benchmarks["model"] == "physics_only") & (benchmarks["fold_id"] == "mean")]
    assert len(phys_row) == 1
    r2 = phys_row["r2"].values[0]
    assert pytest.approx(r2, abs=0.01) == 0.82


def test_segment_evaluation_split() -> None:
    """Verify first-20% (take-off) vs remaining 80% segments computation."""
    n = 100
    df = pd.DataFrame({
        "t_start_ns": np.arange(n) * 1_000_000_000,
        "y_true": np.ones(n) * 2.0,
        "y_pred": np.concatenate([np.ones(20) * 2.5, np.ones(80) * 2.1]),
    })

    segments = score_segments(df, target_col="y_true", pred_col="y_pred", time_col="t_start_ns")
    assert pytest.approx(segments["first_20"]["mae"], abs=1e-5) == 0.5
    assert pytest.approx(segments["remaining_80"]["mae"], abs=1e-5) == 0.1
    assert pytest.approx(segments["overall"]["mae"], abs=1e-5) == 0.18


def test_aggregate_seed_spread() -> None:
    """Verify that multi-seed runs correctly report mean and spread (std)."""
    seeds = [
        {"mean_fold": {"mae": 0.024, "rmse": 0.030, "r2": 0.78, "bias": -0.001}},
        {"mean_fold": {"mae": 0.025, "rmse": 0.031, "r2": 0.77, "bias": -0.001}},
        {"mean_fold": {"mae": 0.026, "rmse": 0.032, "r2": 0.76, "bias": -0.002}},
    ]

    agg = aggregate_seed_metrics(seeds)
    assert pytest.approx(agg["mae"]["mean"], abs=1e-5) == 0.025
    assert agg["mae"]["std"] > 0.0
    assert pytest.approx(agg["r2"]["mean"], abs=1e-5) == 0.77


def test_unified_comparison_table_structure() -> None:
    """Verify unified comparison table outputs Phase 1 benchmarks with proper column headers."""
    table = build_unified_comparison_table()
    assert "Family" in table.columns
    assert "Model" in table.columns
    assert "MAE (cm)" in table.columns
    assert "R²" in table.columns
    assert "Seed Spread (std)" in table.columns
    assert len(table) >= 5
    models_in_table = table["Model"].tolist()
    assert any("Lasso" in m for m in models_in_table)
    assert any("Mean" in m for m in models_in_table)

