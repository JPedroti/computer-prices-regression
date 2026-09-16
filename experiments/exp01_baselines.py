"""Experiment 01 -- baselines and a first pass over every model family.

Hypothesis
----------
The dataset audit showed a nearly additive generating process (a one-hot Ridge
landed within 1.5% of a tuned LightGBM on a single holdout). This experiment
establishes the quantitative reference points required by CLAUDE.md section 8
and checks whether that near-tie survives cross-validation.

Protocol
--------
Screening protocol: 5-fold CV, seed 42, on the development split only. All
experiments share these exact folds, so model-to-model differences are paired
and far more precise than the raw fold-to-fold spread suggests (that spread is
large here because a handful of extreme prices dominate the squared error).
Finalists are later re-checked with repeats and multiple seeds.
The sealed overfitting holdout is never touched here.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.config import CV_FOLDS, RANDOM_SEED
from src.data import load_dev_holdout
from src.evaluate import cross_validate_model
from src.features import FeatureConfig
from src.models import build_model, describe
from src.utils import ExperimentTracker, set_seed

SCREEN_STRATEGY = f"KFold({CV_FOLDS},seed={RANDOM_SEED})"

# name, extra params, preprocessor override, target transform
MODEL_GRID: list[tuple[str, dict, str | None, str | None]] = [
    ("mean", {}, None, None),
    ("median", {}, None, None),
    # --- linear family -------------------------------------------------- #
    ("linear", {}, None, None),
    ("ridge", {"alpha": 10.0}, None, None),
    ("ridge", {"alpha": 100.0}, None, None),
    # Alphas are on the price scale (~2000), so small alphas leave coordinate
    # descent essentially unregularised and painfully slow to converge.
    ("lasso", {"alpha": 1.0, "max_iter": 3000, "tol": 1e-3}, None, None),
    ("elasticnet", {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 3000, "tol": 1e-3}, None, None),
    # --- trees ---------------------------------------------------------- #
    ("tree", {"max_depth": 12, "min_samples_leaf": 50}, None, None),
    ("rf", {"n_estimators": 200, "min_samples_leaf": 5}, None, None),
    ("et", {"n_estimators": 200, "min_samples_leaf": 5}, None, None),
    # --- boosting ------------------------------------------------------- #
    ("hgb", {"max_iter": 400, "learning_rate": 0.06}, None, None),
    ("lgbm", {"n_estimators": 400, "learning_rate": 0.06}, None, None),
    ("xgb", {"n_estimators": 400, "learning_rate": 0.06, "max_depth": 6}, None, None),
    ("cat", {"iterations": 600, "learning_rate": 0.06, "depth": 6}, None, None),
    # --- target transform probe ----------------------------------------- #
    ("lgbm", {"n_estimators": 400, "learning_rate": 0.06}, None, "log"),
    ("ridge", {"alpha": 10.0}, None, "log"),
]


def main() -> None:
    set_seed(RANDOM_SEED)
    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()
    print(f"development rows={len(X_dev)}  raw features={X_dev.shape[1]}")

    # Baseline feature set: structural decompositions only, no derived families.
    cfg = FeatureConfig()
    print(f"feature families: {cfg.enabled()}\n")

    results = []
    for name, params, prep, ttf in MODEL_GRID:
        tag = describe(name, prep, ttf)
        model = build_model(
            name, feature_config=cfg, preprocessor=prep, target_transform=ttf,
            seed=RANDOM_SEED, **params
        )
        res = cross_validate_model(
            model, X_dev, y_dev, name=f"{tag} {params}",
            n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED, return_oof=False,
        )
        print(res.summary(), flush=True)
        tracker.log(
            model=tag,
            features=cfg.enabled(),
            preprocessing=tag.split("|")[1],
            hyperparameters=params,
            seed=RANDOM_SEED,
            validation_strategy=SCREEN_STRATEGY,
            train_rmse=res.train_rmse,
            validation_rmse=res.val_rmse,
            train_mae=res.train_mae,
            validation_mae=res.val_mae,
            validation_rmse_std=res.val_rmse_std,
            fit_seconds=res.fit_seconds,
            notes="exp01 baseline sweep, base feature families",
        )
        results.append(res)

    print("\n=== ranked by validation RMSE ===")
    for r in sorted(results, key=lambda r: r.val_rmse):
        print(r.summary())


if __name__ == "__main__":
    main()
