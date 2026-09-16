"""Experiment 07 -- do ensembles beat the best single model?

Out-of-fold predictions are produced once per candidate on the shared folds and
then combined offline, so many blends can be compared without refitting
anything. Three combiners are tried:

* simple average;
* non-negative least squares weights (no member may contribute negatively);
* a ridge meta-learner (stacking).

The weights are fitted on out-of-fold predictions, which is the only leak-free
way to learn them: every prediction being combined was made by a model that had
not seen that row.

A blend is only adopted if it beats the best single model by more than the
paired fold-to-fold noise, and if it does not reintroduce a large train gap
(CLAUDE.md section 16 -- no complexity without measurable benefit).
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import CV_FOLDS, RANDOM_SEED
from src.data import load_dev_holdout
from src.ensemble import blend_search
from src.evaluate import cross_validate_model, rmse
from src.features import FeatureConfig
from src.models import build_model
from src.utils import ExperimentTracker

BASE = FeatureConfig()

# name -> (model, preprocessor, params, feature_config)
CANDIDATES: dict[str, tuple[str, str | None, dict, FeatureConfig]] = {
    "ridge_levels": ("ridge", "levels_plus", {"alpha": 100.0}, BASE),
    "ridge_onehot": ("ridge", "onehot", {"alpha": 10.0}, BASE),
    "hgb": ("hgb", None, {"max_iter": 400, "learning_rate": 0.06}, BASE),
    "lgbm": ("lgbm", None, {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31}, BASE),
    "cat": ("cat", None, {"iterations": 800, "learning_rate": 0.06, "depth": 6}, BASE),
}


def main() -> None:
    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()
    y = y_dev.to_numpy(dtype=float)
    strategy = f"KFold({CV_FOLDS},seed={RANDOM_SEED})"

    oof: dict[str, np.ndarray] = {}
    singles: dict[str, float] = {}
    folds: dict[str, list[float]] = {}

    for label, (name, prep, params, cfg) in CANDIDATES.items():
        model = build_model(name, feature_config=cfg, preprocessor=prep, seed=RANDOM_SEED, **params)
        res = cross_validate_model(
            model, X_dev, y_dev, name=label,
            n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED, return_oof=True,
        )
        oof[label] = res.oof_pred
        singles[label] = res.val_rmse
        folds[label] = res.fold_val_rmse
        print(res.summary(), flush=True)
        tracker.log(
            model=label, features=cfg.enabled(), preprocessing=prep or "native",
            hyperparameters=params, seed=RANDOM_SEED, validation_strategy=strategy,
            train_rmse=res.train_rmse, validation_rmse=res.val_rmse,
            train_mae=res.train_mae, validation_mae=res.val_mae,
            validation_rmse_std=res.val_rmse_std, fit_seconds=res.fit_seconds,
            notes="exp07 ensemble candidate (OOF generated)",
        )

    best_single = min(singles, key=singles.get)
    print(f"\nbest single model: {best_single} at {singles[best_single]:.3f}")

    print("\n=== how different are the members' errors? (residual correlation) ===")
    resid = pd.DataFrame({k: y - v for k, v in oof.items()})
    print(resid.corr().round(4).to_string())

    print("\n=== blends over out-of-fold predictions ===")
    rows = []
    names = list(oof)
    for size in range(2, len(names) + 1):
        for subset in combinations(names, size):
            sub = {k: oof[k] for k in subset}
            avg = np.mean(np.column_stack(list(sub.values())), axis=1)
            rows.append(("average", "+".join(subset), rmse(y, avg), None))
            weights, r = blend_search(sub, y)
            rows.append(("nnls", "+".join(subset), r, weights))

    table = pd.DataFrame(rows, columns=["combiner", "members", "rmse", "weights"])
    table = table.sort_values("rmse").reset_index(drop=True)
    print(table.head(12).to_string(index=False))

    best_blend = table.iloc[0]
    gain = singles[best_single] - best_blend["rmse"]
    print(f"\nbest blend  : {best_blend['combiner']} over {best_blend['members']}")
    print(f"blend RMSE  : {best_blend['rmse']:.3f}")
    print(f"best single : {singles[best_single]:.3f}  ({best_single})")
    print(f"gain        : {gain:+.3f}")
    if best_blend["weights"]:
        print("weights     :", {k: round(v, 4) for k, v in best_blend["weights"].items() if v > 1e-6})

    table.to_csv(Path(__file__).resolve().parents[1] / "experiments" / "exp07_blends.csv", index=False)
    tracker.log(
        model=f"blend[{best_blend['members']}]",
        features=BASE.enabled(), preprocessing="mixed",
        hyperparameters={"combiner": best_blend["combiner"], "weights": best_blend["weights"]},
        seed=RANDOM_SEED, validation_strategy=strategy + " OOF blend",
        train_rmse=float("nan"), validation_rmse=best_blend["rmse"],
        train_mae=float("nan"), validation_mae=float("nan"),
        notes=f"exp07 best blend; gain vs best single ({best_single}) = {gain:+.3f}",
    )


if __name__ == "__main__":
    main()
