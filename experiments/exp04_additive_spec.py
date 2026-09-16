"""Experiment 04 -- find the best additive specification.

Findings that motivate this
---------------------------
Experiment 02 showed, on one held-out split, that

* a saturated additive model beats a 1500-tree LightGBM (225.8 vs 232.8);
* boosting the additive model's residual makes it *worse* (230.9), and the
  residual importances are flat and diffuse -- the signature of noise rather
  than of unexploited interactions;
* a log-space (multiplicative) fit is clearly worse (233.1).

So the generating process is additive and the remaining question is purely how
to *specify* that additive model. Three representations are compared here under
identical folds:

``onehot``       numeric columns enter as one linear term each;
``levels``       every low-cardinality column becomes one dummy per level;
``levels_plus``  dummies *and* the original numeric term, so the model keeps a
                 monotone extrapolation for thinly observed levels.

Regularisation is scanned over a wide range because the alphas used in
experiment 02 (<= 10) turned out to be negligible against a 415-column design
matrix and 48k rows.
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

SCREEN_STRATEGY = f"KFold({CV_FOLDS},seed={RANDOM_SEED})"

ALPHAS = (1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessors", nargs="*", default=["onehot", "levels", "levels_plus"],
        help="restrict the scan, e.g. to rerun one representation after a fix",
    )
    args = parser.parse_args()
    specs = [(prep, "ridge", {"alpha": a}) for prep in args.preprocessors for a in ALPHAS]

    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()
    cfg = FeatureConfig()
    print(f"development rows={len(X_dev)}  feature families={cfg.enabled()}\n")

    results = []
    for prep, model_name, params in specs:
        model = build_model(
            model_name, feature_config=cfg, preprocessor=prep, seed=RANDOM_SEED, **params
        )
        res = cross_validate_model(
            model, X_dev, y_dev, name=f"{model_name}|{prep} {params}",
            n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED, return_oof=False,
        )
        print(res.summary(), flush=True)
        tracker.log(
            model=f"{model_name}|{prep}",
            features=cfg.enabled(),
            preprocessing=prep,
            hyperparameters=params,
            seed=RANDOM_SEED,
            validation_strategy=SCREEN_STRATEGY,
            train_rmse=res.train_rmse,
            validation_rmse=res.val_rmse,
            train_mae=res.train_mae,
            validation_mae=res.val_mae,
            validation_rmse_std=res.val_rmse_std,
            fit_seconds=res.fit_seconds,
            notes="exp04 additive specification scan",
        )
        results.append((res, prep, params))

    print("\n=== ranked ===")
    for res, prep, params in sorted(results, key=lambda t: t[0].val_rmse):
        print(f"{prep:12s} {str(params):26s} valRMSE={res.val_rmse:8.3f} gap={res.gap:+7.3f}")

    best = min(results, key=lambda t: t[0].val_rmse)
    print(f"\nbest specification: {best[1]} {best[2]} -> valRMSE={best[0].val_rmse:.3f}")

    # Paired fold-by-fold comparison against the onehot reference: the absolute
    # fold spread is large, but the folds are shared so differences are paired.
    reference = [r for r, p, pr in results if p == "onehot" and pr["alpha"] == 10.0]
    if not reference:
        return  # scan restricted to one representation; no shared reference to pair against
    ref = reference[0]
    print("\npaired differences vs ridge|onehot(alpha=10) per fold:")
    for res, prep, params in sorted(results, key=lambda t: t[0].val_rmse)[:6]:
        diff = np.array(res.fold_val_rmse) - np.array(ref.fold_val_rmse)
        print(f"  {prep:12s} {str(params):20s} mean={diff.mean():+7.3f} "
              f"sd={diff.std(ddof=1):6.3f} folds={np.round(diff, 2).tolist()}")


if __name__ == "__main__":
    main()
