"""Model factory.

Every model is returned as a complete pipeline

    feature engineering -> column representation -> estimator

optionally wrapped in a target transform. Building models only through
:func:`build_model` guarantees that training, validation and inference always
apply exactly the same steps (CLAUDE.md section 11).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .config import RANDOM_SEED
from .features import FeatureConfig
from .preprocessing import build_preprocessor

# Which column representation each estimator family expects by default.
DEFAULT_PREPROCESSOR: dict[str, str] = {
    "mean": "ordinal",
    "median": "ordinal",
    "linear": "onehot",
    "ridge": "onehot",
    "lasso": "onehot",
    "elasticnet": "onehot",
    "tree": "ordinal",
    "rf": "ordinal",
    "et": "ordinal",
    "hgb": "ordinal",
    "lgbm": "native",
    "xgb": "native",
    "cat": "native",
}


class CatBoostNative(BaseEstimator, RegressorMixin):
    """CatBoost wrapper that discovers categorical columns from the DataFrame.

    CatBoost needs the categorical columns declared explicitly and rejects NaN
    in them, so unseen categories are mapped to an explicit sentinel token.
    """

    _MISSING = "__missing__"

    def __init__(self, **params: Any):
        self.params = params

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"params": self.params}

    def set_params(self, **kwargs: Any) -> "CatBoostNative":
        if "params" in kwargs:
            self.params = kwargs.pop("params")
        self.params.update(kwargs)
        return self

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.cat_features_:
            out[col] = out[col].astype(object).where(out[col].notna(), self._MISSING).astype(str)
        return out

    def fit(self, X: pd.DataFrame, y) -> "CatBoostNative":
        from catboost import CatBoostRegressor

        self.cat_features_ = [
            c for c in X.columns if str(X[c].dtype) in ("category", "object", "str", "string")
        ]
        defaults = {"verbose": 0, "random_seed": RANDOM_SEED, "allow_writing_files": False}
        self.model_ = CatBoostRegressor(**{**defaults, **self.params})
        self.model_.fit(self._prepare(X), np.asarray(y, dtype=float), cat_features=self.cat_features_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._prepare(X))


def _make_estimator(name: str, seed: int, params: dict[str, Any]):
    """Instantiate the bare estimator for ``name`` with sensible defaults."""
    p = dict(params)

    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "median":
        return DummyRegressor(strategy="median")

    if name == "linear":
        return LinearRegression(**p)
    if name == "ridge":
        return Ridge(**{"alpha": 1.0, "random_state": seed, **p})
    if name == "lasso":
        return Lasso(**{"alpha": 0.1, "random_state": seed, "max_iter": 5000, **p})
    if name == "elasticnet":
        return ElasticNet(**{"alpha": 0.1, "l1_ratio": 0.5, "random_state": seed, "max_iter": 5000, **p})

    if name == "tree":
        return DecisionTreeRegressor(**{"random_state": seed, **p})
    if name == "rf":
        return RandomForestRegressor(**{"n_estimators": 300, "random_state": seed, "n_jobs": -1, **p})
    if name == "et":
        return ExtraTreesRegressor(**{"n_estimators": 300, "random_state": seed, "n_jobs": -1, **p})
    if name == "hgb":
        return HistGradientBoostingRegressor(**{"random_state": seed, **p})

    if name == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            **{"random_state": seed, "n_jobs": -1, "verbose": -1, "n_estimators": 400, **p}
        )
    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            **{
                "random_state": seed,
                "n_jobs": -1,
                "enable_categorical": True,
                "tree_method": "hist",
                "n_estimators": 400,
                **p,
            }
        )
    if name == "cat":
        return CatBoostNative(**{"random_seed": seed, **p})

    raise ValueError(f"unknown model name {name!r}")


def build_model(
    name: str,
    *,
    feature_config: FeatureConfig | None = None,
    preprocessor: str | None = None,
    target_transform: str | None = None,
    seed: int = RANDOM_SEED,
    **params: Any,
):
    """Assemble the full pipeline for ``name``.

    ``target_transform`` may be ``None`` (raw target) or ``"log"``, which fits
    on ``log(price)`` and inverts the prediction back to the original scale.
    """
    kind = preprocessor or DEFAULT_PREPROCESSOR.get(name, "native")
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(kind, feature_config)),
            ("model", _make_estimator(name, seed, params)),
        ]
    )

    if target_transform is None:
        return pipe
    if target_transform == "log":
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    if target_transform == "sqrt":
        return TransformedTargetRegressor(regressor=pipe, func=np.sqrt, inverse_func=np.square)
    raise ValueError(f"unknown target_transform {target_transform!r}")


def describe(name: str, preprocessor: str | None, target_transform: str | None) -> str:
    """Short human-readable tag used in experiment logs."""
    kind = preprocessor or DEFAULT_PREPROCESSOR.get(name, "native")
    tag = f"{name}|{kind}"
    if target_transform:
        tag += f"|{target_transform}"
    return tag
