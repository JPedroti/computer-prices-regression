"""Experiment 08 -- boosting constrained to be additive.

Motivation
----------
Two facts are now established:

* the process is additive (exp 02, exp 06);
* a saturated additive Ridge (211.9) beats unconstrained boosting
  (HistGradientBoosting 214.7, LightGBM 214.9), because the boosters spend
  capacity on interactions that are not there.

But the Ridge and the boosters differ in another way that has nothing to do
with interactions: the Ridge estimates each level's effect independently, while
a booster *shrinks* neighbouring levels towards each other through its stagewise
fitting. Shrinkage should help exactly where a level is thinly observed.

Hypothesis: a booster forced to be additive gets the best of both -- no wasted
capacity on interactions, plus shrinkage of the per-level effects.

Two ways to force additivity are tried:

``stumps``      depth-1 trees: each tree splits on one feature, so the ensemble
                is a sum of per-feature step functions;
``constrained`` LightGBM ``interaction_constraints`` putting every feature in
                its own group: trees may be deep but can only ever use a single
                feature, giving a richer per-feature shape while staying additive.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, RANDOM_SEED
from src.data import load_dev_holdout
from src.evaluate import cross_validate_model
from src.features import FeatureConfig, FeatureEngineer
from src.models import build_model
from src.preprocessing import ColumnTyper
from src.utils import ExperimentTracker


class AdditiveLGBM(BaseEstimator, RegressorMixin):
    """LightGBM restricted to additive fits.

    ``interaction_constraints`` must name feature indices, which are only known
    once the preprocessing has run, so they are built at fit time: every column
    becomes its own group, meaning no tree can combine two features.
    """

    def __init__(self, params: dict | None = None, mode: str = "constrained"):
        self.params = params
        self.mode = mode

    def fit(self, X, y) -> "AdditiveLGBM":
        from lightgbm import LGBMRegressor

        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        defaults = {
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        params = {**defaults, **(self.params or {})}

        if self.mode == "stumps":
            params["num_leaves"] = 2
            params["max_depth"] = 1
        elif self.mode == "constrained":
            params["interaction_constraints"] = [[i] for i in range(X.shape[1])]
        elif self.mode != "free":
            raise ValueError(f"unknown mode {self.mode!r}")

        self.model_ = LGBMRegressor(**params)
        self.model_.fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(X)


def build_additive_booster(mode: str, params: dict, cfg: FeatureConfig) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureEngineer(cfg)),
            ("columns", ColumnTyper("native")),
            ("model", AdditiveLGBM(params=params, mode=mode)),
        ]
    )


RUNS: list[tuple[str, str, dict]] = [
    ("stumps lr.05 n2000", "stumps", {"n_estimators": 2000, "learning_rate": 0.05}),
    ("stumps lr.05 n5000", "stumps", {"n_estimators": 5000, "learning_rate": 0.05}),
    ("stumps lr.02 n8000", "stumps", {"n_estimators": 8000, "learning_rate": 0.02}),
    ("constrained nl15 n1500", "constrained", {"n_estimators": 1500, "num_leaves": 15}),
    ("constrained nl31 n1500", "constrained", {"n_estimators": 1500, "num_leaves": 31}),
    ("constrained nl31 n3000 lr.03", "constrained",
     {"n_estimators": 3000, "num_leaves": 31, "learning_rate": 0.03}),
    ("constrained nl63 n2000 mcs50", "constrained",
     {"n_estimators": 2000, "num_leaves": 63, "min_child_samples": 50}),
    ("free nl31 n1500 (reference)", "free", {"n_estimators": 1500, "num_leaves": 31}),
]


def main() -> None:
    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()
    cfg = FeatureConfig()
    strategy = f"KFold({CV_FOLDS},seed={RANDOM_SEED})"

    # Reference: the best additive Ridge found in exp04.
    ridge = build_model("ridge", feature_config=cfg, preprocessor="levels_plus",
                        seed=RANDOM_SEED, alpha=100.0)
    ref = cross_validate_model(ridge, X_dev, y_dev, name="ridge|levels_plus alpha=100",
                               n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED, return_oof=False)
    print(ref.summary(), flush=True)

    results = [(ref, "ridge reference")]
    for label, mode, params in RUNS:
        model = build_additive_booster(mode, params, cfg)
        res = cross_validate_model(model, X_dev, y_dev, name=label,
                                   n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED,
                                   return_oof=False)
        print(res.summary(), flush=True)
        tracker.log(
            model=f"lgbm-additive[{mode}]", features=cfg.enabled(), preprocessing="native",
            hyperparameters=params, seed=RANDOM_SEED, validation_strategy=strategy,
            train_rmse=res.train_rmse, validation_rmse=res.val_rmse,
            train_mae=res.train_mae, validation_mae=res.val_mae,
            validation_rmse_std=res.val_rmse_std, fit_seconds=res.fit_seconds,
            notes=f"exp08 additive boosting: {label}",
        )
        results.append((res, label))

    print("\n=== paired difference vs ridge|levels_plus (negative = better) ===")
    for res, label in sorted(results, key=lambda t: t[0].val_rmse):
        diff = np.array(res.fold_val_rmse) - np.array(ref.fold_val_rmse)
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        print(f"  {label:32s} valRMSE={res.val_rmse:8.3f} gap={res.gap:+6.2f} "
              f"diff={diff.mean():+7.3f} +-{se:5.3f}")


if __name__ == "__main__":
    main()
