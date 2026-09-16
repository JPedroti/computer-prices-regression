"""Experiment 01b -- the baseline entries that experiment 01 could not finish.

The first run of exp01 aborted at CatBoost: the wrapper was not clonable by
scikit-learn, so cross-validation could not copy it per fold. With the wrapper
fixed, this script fills in CatBoost and the two target-transform probes under
the identical screening protocol, so all baseline rows are comparable.
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
from src.utils import ExperimentTracker

GRID: list[tuple[str, dict, str | None, str | None]] = [
    ("cat", {"iterations": 600, "learning_rate": 0.06, "depth": 6}, None, None),
    ("lgbm", {"n_estimators": 400, "learning_rate": 0.06}, None, "log"),
    ("ridge", {"alpha": 10.0}, None, "log"),
    ("ridge", {"alpha": 10.0}, "levels", "log"),
]


def main() -> None:
    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()
    cfg = FeatureConfig()
    strategy = f"KFold({CV_FOLDS},seed={RANDOM_SEED})"

    for name, params, prep, ttf in GRID:
        tag = describe(name, prep, ttf)
        model = build_model(
            name, feature_config=cfg, preprocessor=prep,
            target_transform=ttf, seed=RANDOM_SEED, **params
        )
        res = cross_validate_model(
            model, X_dev, y_dev, name=f"{tag} {params}",
            n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED, return_oof=False,
        )
        print(res.summary(), flush=True)
        tracker.log(
            model=tag, features=cfg.enabled(), preprocessing=tag.split("|")[1],
            hyperparameters=params, seed=RANDOM_SEED, validation_strategy=strategy,
            train_rmse=res.train_rmse, validation_rmse=res.val_rmse,
            train_mae=res.train_mae, validation_mae=res.val_mae,
            validation_rmse_std=res.val_rmse_std, fit_seconds=res.fit_seconds,
            notes="exp01b baseline completion",
        )


if __name__ == "__main__":
    main()
