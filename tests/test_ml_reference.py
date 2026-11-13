"""R4's Phase 2 ML reference: Lasso on features (+ 4 physics numbers)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vio_drift.models.dl.ml_reference import design_matrix, fit_lasso_reference, predict_reference


class _Split:
    """Stands in for DriftDataset: the array attributes the reference reads."""

    def __init__(self, n, seed, split):
        rng = np.random.default_rng(seed)
        self.split = split
        self.feats = rng.standard_normal((n, 104)).astype(np.float32)
        self.phys = (rng.uniform(1, 2, (n, 4))).astype(np.float32)
        y = 1.0 * self.phys[:, 3] + 0.02 * self.feats[:, 0] + 0.002 * rng.standard_normal(n)
        self.labels = np.column_stack([y, rng.standard_normal((n, 3))]).astype(np.float32)
        self.meta = [{"seq_id": "toy", "window_id": i, "t_start_ns": i} for i in range(n)]


def test_design_matrix_has_108_columns_with_physics_and_104_without():
    data = _Split(10, 0, "train")
    assert design_matrix(data, use_physics=True).shape == (10, 108)
    assert design_matrix(data, use_physics=False).shape == (10, 104)
    np.testing.assert_array_equal(design_matrix(data, True)[:, 104:], data.phys)


def test_physics_columns_lower_the_error_when_target_depends_on_them():
    train, val, test = _Split(300, 0, "train"), _Split(100, 1, "validation"), _Split(100, 2, "test")
    with_phys = fit_lasso_reference(train, val, use_physics=True)
    without = fit_lasso_reference(train, val, use_physics=False)
    err = lambda r: np.abs(predict_reference(r, test).eval("y_pred - y_true")).mean()
    assert err(with_phys) < 0.5 * err(without)


def test_alpha_is_chosen_on_validation_from_the_grid():
    result = fit_lasso_reference(_Split(200, 0, "train"), _Split(80, 1, "validation"),
                                 alphas=[1e-4, 1e-1])
    assert result.alpha in (1e-4, 1e-1)
    assert set(result.val_mae) == {1e-4, 1e-1}
    assert result.val_mae[result.alpha] == min(result.val_mae.values())


def test_test_split_cannot_choose_alpha():
    with pytest.raises(ValueError, match="validation"):
        fit_lasso_reference(_Split(50, 0, "train"), _Split(20, 1, "test"))


def test_must_fit_on_train_split():
    with pytest.raises(ValueError, match="train"):
        fit_lasso_reference(_Split(50, 0, "validation"), _Split(20, 1, "validation"))


def test_predictions_follow_contract_and_order():
    result = fit_lasso_reference(_Split(200, 0, "train"), _Split(80, 1, "validation"))
    test = _Split(15, 2, "test")
    frame = predict_reference(result, test)
    assert list(frame.columns) == ["seq_id", "window_id", "y_true", "y_pred"]
    assert frame.window_id.tolist() == list(range(15))
    np.testing.assert_allclose(frame.y_true, test.labels[:, 0])
