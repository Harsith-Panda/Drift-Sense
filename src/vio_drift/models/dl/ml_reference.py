"""R4's Phase 2 ML reference: Lasso on the 104 features, optionally + 4 physics numbers.

This is the like-for-like ML row for the hybrids. It uses exactly what H2 sees:
R1's inner-training windows, R1's fold-specific ``physics_dl`` numbers, and
alpha chosen on the validation block (the rule that picks DL checkpoints).
Phase 1 code and results are not touched.

  ML_FEAT  Lasso, 104 features
  ML_PHYS  Lasso, 104 features + phys_dx, phys_dy, phys_dz, phys_mag
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = [float(a) for a in np.logspace(-6, -1, 16)]  # Phase 1's Lasso grid


@dataclass
class ReferenceResult:
    model: Pipeline
    alpha: float
    use_physics: bool
    val_mae: dict  # alpha -> validation MAE (m)


def design_matrix(data, use_physics: bool) -> np.ndarray:
    """R1-normalised features, then raw physics numbers (scaled in the pipeline)."""
    parts = [np.asarray(data.feats, dtype=np.float64)]
    if use_physics:
        parts.append(np.asarray(data.phys, dtype=np.float64))
    return np.concatenate(parts, axis=1)


def _fit(alpha, X, y):
    model = Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=50_000))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence at tiny alpha
        model.fit(X, y)
    return model


def fit_lasso_reference(train, val, use_physics=True, alphas=ALPHAS) -> ReferenceResult:
    """Fit on train for every alpha, keep the lowest validation MAE. Test never selects."""
    if train.split != "train":
        raise ValueError("fit the reference on the train split")
    if val.split != "validation":
        raise ValueError("alpha must be chosen on the validation split, not the test flight")
    X, y = design_matrix(train, use_physics), np.asarray(train.labels)[:, 0]
    Xv, yv = design_matrix(val, use_physics), np.asarray(val.labels)[:, 0]
    val_mae = {}
    for alpha in alphas:
        val_mae[alpha] = float(np.abs(_fit(alpha, X, y).predict(Xv) - yv).mean())
    best = min(val_mae, key=val_mae.get)
    return ReferenceResult(_fit(best, X, y), best, use_physics, val_mae)


def predict_reference(result: ReferenceResult, data) -> pd.DataFrame:
    """Shared prediction contract, in dataset order."""
    pred = result.model.predict(design_matrix(data, result.use_physics))
    return pd.DataFrame({"seq_id": [m["seq_id"] for m in data.meta],
                         "window_id": [int(m["window_id"]) for m in data.meta],
                         "y_true": np.asarray(data.labels)[:, 0].astype(np.float64),
                         "y_pred": pred})


__all__ = ["ALPHAS", "ReferenceResult", "design_matrix", "fit_lasso_reference", "predict_reference"]
