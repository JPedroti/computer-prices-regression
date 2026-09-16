"""Experiment 10 -- targeted CatBoost tuning.

CatBoost reached 211.38 out of the box, marginally ahead of the additive Ridge
(211.92) and well ahead of every other booster (HistGradientBoosting 214.68,
LightGBM 214.89, XGBoost 217.61). Two of its design choices plausibly explain
that: symmetric (oblivious) trees, which are a strong regulariser, and ordered
target statistics for categorical columns.

Phase 1 of the tuning (this script) explores the directions that the additive
finding says should matter, rather than a large blind grid:

* **depth** -- if the truth is additive, shallower trees should not lose much
  and should shrink the train/validation gap;
* **l2_leaf_reg** -- direct control of that gap;
* **one_hot_max_size** -- forcing plain one-hot encoding for the low-cardinality
  categoricals instead of ordered target statistics, which tests whether the
  target statistics are helping or just adding variance;
* **learning rate / iterations** -- the usual trade, kept to two settings.

Each configuration costs about six minutes, so the grid is deliberately small
(CLAUDE.md section 31).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

from src.config import CV_FOLDS, RANDOM_SEED
from src.data import load_dev_holdout
from src.evaluate import cross_validate_model
from src.features import FeatureConfig
from src.models import build_model
from src.utils import ExperimentTracker

GRID: list[tuple[str, dict]] = [
    ("depth6 it600 lr.06 (reference)", {"depth": 6, "iterations": 600, "learning_rate": 0.06}),
    ("depth4 it1000 lr.06", {"depth": 4, "iterations": 1000, "learning_rate": 0.06}),
    ("depth5 it800 lr.06", {"depth": 5, "iterations": 800, "learning_rate": 0.06}),
    ("depth6 it600 l2=10", {"depth": 6, "iterations": 600, "learning_rate": 0.06, "l2_leaf_reg": 10.0}),
    ("depth6 it600 l2=30", {"depth": 6, "iterations": 600, "learning_rate": 0.06, "l2_leaf_reg": 30.0}),
    ("depth6 it1500 lr.03", {"depth": 6, "iterations": 1500, "learning_rate": 0.03}),
    ("depth6 it600 onehot64", {"depth": 6, "iterations": 600, "learning_rate": 0.06,
                               "one_hot_max_size": 64}),
    ("depth4 it2000 lr.03 l2=10", {"depth": 4, "iterations": 2000, "learning_rate": 0.03,
                                   "l2_leaf_reg": 10.0}),
]


def main() -> None:
    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()
    cfg = FeatureConfig()
    strategy = f"KFold({CV_FOLDS},seed={RANDOM_SEED})"

    results = []
    for label, params in GRID:
        model = build_model("cat", feature_config=cfg, seed=RANDOM_SEED, **params)
        res = cross_validate_model(
            model, X_dev, y_dev, name=label,
            n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED, return_oof=False,
        )
        print(res.summary(), flush=True)
        tracker.log(
            model="cat|native", features=cfg.enabled(), preprocessing="native",
            hyperparameters=params, seed=RANDOM_SEED, validation_strategy=strategy,
            train_rmse=res.train_rmse, validation_rmse=res.val_rmse,
            train_mae=res.train_mae, validation_mae=res.val_mae,
            validation_rmse_std=res.val_rmse_std, fit_seconds=res.fit_seconds,
            notes=f"exp10 catboost tuning phase 1: {label}",
        )
        results.append((res, label))

    ref = results[0][0]
    print("\n=== paired difference vs the reference configuration ===")
    for res, label in sorted(results, key=lambda t: t[0].val_rmse):
        diff = np.array(res.fold_val_rmse) - np.array(ref.fold_val_rmse)
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"  {label:32s} valRMSE={res.val_rmse:8.3f} gap={res.gap:+6.2f} "
              f"diff={diff.mean():+7.3f} +-{se:5.3f}")


if __name__ == "__main__":
    main()
