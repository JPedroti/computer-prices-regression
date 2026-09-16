"""Ensembling: weighted averaging and stacking.

Both estimators are ordinary scikit-learn regressors, so they can be dropped
into :func:`src.evaluate.cross_validate_model` and compared against single
models under exactly the same protocol.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.optimize import nnls

from .config import RANDOM_SEED


class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """Average member predictions with fixed or learned non-negative weights.

    With ``weights=None`` the weights are learned on out-of-fold predictions by
    non-negative least squares, which keeps the blend interpretable and stops
    any member from being given a negative (error-amplifying) contribution.
    Weight learning uses only training-fold data, so it introduces no leakage.
    """

    def __init__(
        self,
        members: Sequence[tuple[str, Any]],
        weights: Sequence[float] | None = None,
        n_splits: int = 5,
        seed: int = RANDOM_SEED,
    ):
        self.members = members
        self.weights = weights
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, X: pd.DataFrame, y) -> "WeightedEnsemble":
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)

        if self.weights is None:
            oof = np.zeros((len(X), len(self.members)), dtype=float)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            for tr_idx, va_idx in kf.split(X):
                for j, (_, est) in enumerate(self.members):
                    m = clone(est)
                    m.fit(X.iloc[tr_idx], y[tr_idx])
                    oof[va_idx, j] = m.predict(X.iloc[va_idx])
            w, _ = nnls(oof, y)
            total = w.sum()
            self.weights_ = w / total if total > 0 else np.full(len(self.members), 1 / len(self.members))
            self.oof_ = oof
        else:
            w = np.asarray(self.weights, dtype=float)
            self.weights_ = w / w.sum()
            self.oof_ = None

        self.fitted_ = []
        for _, est in self.members:
            m = clone(est)
            m.fit(X, y)
            self.fitted_.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self.fitted_])
        return preds @ self.weights_

    def weight_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"member": [n for n, _ in self.members], "weight": np.round(self.weights_, 4)}
        ).sort_values("weight", ascending=False)


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """Stacking with a ridge meta-learner over out-of-fold member predictions."""

    def __init__(
        self,
        members: Sequence[tuple[str, Any]],
        n_splits: int = 5,
        seed: int = RANDOM_SEED,
        passthrough_alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    ):
        self.members = members
        self.n_splits = n_splits
        self.seed = seed
        self.passthrough_alphas = passthrough_alphas

    def fit(self, X: pd.DataFrame, y) -> "StackingEnsemble":
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)

        oof = np.zeros((len(X), len(self.members)), dtype=float)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for tr_idx, va_idx in kf.split(X):
            for j, (_, est) in enumerate(self.members):
                m = clone(est)
                m.fit(X.iloc[tr_idx], y[tr_idx])
                oof[va_idx, j] = m.predict(X.iloc[va_idx])

        self.meta_ = RidgeCV(alphas=list(self.passthrough_alphas))
        self.meta_.fit(oof, y)
        self.oof_ = oof

        self.fitted_ = []
        for _, est in self.members:
            m = clone(est)
            m.fit(X, y)
            self.fitted_.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self.fitted_])
        return self.meta_.predict(preds)

    def weight_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"member": [n for n, _ in self.members], "coef": np.round(self.meta_.coef_, 4)}
        ).sort_values("coef", ascending=False)


def blend_search(
    oof_preds: dict[str, np.ndarray], y_true: np.ndarray
) -> tuple[dict[str, float], float]:
    """Find non-negative blend weights on existing out-of-fold predictions.

    Returns the normalised weights and the RMSE they achieve. Operating on
    stored OOF predictions makes blend selection cheap enough to compare many
    candidate subsets without refitting anything.
    """
    names = list(oof_preds)
    matrix = np.column_stack([oof_preds[n] for n in names])
    mask = np.isfinite(matrix).all(axis=1) & np.isfinite(y_true)
    w, _ = nnls(matrix[mask], y_true[mask])
    total = w.sum()
    w = w / total if total > 0 else np.full(len(names), 1 / len(names))
    blended = matrix[mask] @ w
    rmse = float(np.sqrt(np.mean((y_true[mask] - blended) ** 2)))
    return dict(zip(names, w.round(6))), rmse
