"""Experiment 06b -- combine only the interactions that helped on their own.

exp06 showed that adding all ten candidate pairs makes the model worse (+0.355).
That is not a fair test of combining, though, because the ten include pairs that
hurt individually. The honest follow-up is to combine only the four that showed
a negative paired difference, and see whether their small individual gains add
up or wash out.

Selecting these four by their measured effect and then re-scoring them on the
same folds is mildly optimistic -- the folds that made them look good are the
folds they are re-scored on. That is precisely why a gain here would have to be
clearly larger than the sum of the individual effects to be worth adopting, and
why the outcome is reported with that caveat attached.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp06_interactions import build  # noqa: E402

from src.config import CV_FOLDS, RANDOM_SEED  # noqa: E402
from src.data import load_dev_holdout  # noqa: E402
from src.evaluate import cross_validate_model  # noqa: E402
from src.features import FeatureConfig  # noqa: E402
from src.utils import ExperimentTracker  # noqa: E402

# The four pairs whose paired difference in exp06 was negative.
BEST_PAIRS = [
    ("brand", "cpu_family"),
    ("brand", "device_type"),
    ("device_type", "cpu_family"),
    ("device_type", "gpu_model"),
]
SUM_OF_INDIVIDUAL_GAINS = -0.396  # -0.131 -0.094 -0.094 -0.077


def main() -> None:
    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()
    cfg = FeatureConfig()
    strategy = f"KFold({CV_FOLDS},seed={RANDOM_SEED})"

    runs = [("additive only", []), ("four best pairs", BEST_PAIRS)]
    results = {}
    for label, pairs in runs:
        model = build(pairs, "levels_plus", 100.0, cfg)
        res = cross_validate_model(
            model, X_dev, y_dev, name=label,
            n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED, return_oof=False,
        )
        results[label] = res
        print(res.summary(), flush=True)
        tracker.log(
            model="ridge|levels_plus+interactions", features=cfg.enabled(),
            preprocessing="levels_plus",
            hyperparameters={"alpha": 100.0, "pairs": [f"{a}x{b}" for a, b in pairs]},
            seed=RANDOM_SEED, validation_strategy=strategy,
            train_rmse=res.train_rmse, validation_rmse=res.val_rmse,
            train_mae=res.train_mae, validation_mae=res.val_mae,
            validation_rmse_std=res.val_rmse_std, fit_seconds=res.fit_seconds,
            notes=f"exp06b: {label}",
        )

    base, combo = results["additive only"], results["four best pairs"]
    diff = np.array(combo.fold_val_rmse) - np.array(base.fold_val_rmse)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"\ncombined effect : {diff.mean():+.3f} +-{se:.3f}")
    print(f"sum of individual effects measured in exp06: {SUM_OF_INDIVIDUAL_GAINS:+.3f}")
    print(f"train gap: additive {base.gap:+.2f} -> with interactions {combo.gap:+.2f}")
    print(
        "\nAdopt only if the combined gain is both clearly negative and large enough\n"
        "to matter against a fold spread of about 15 RMSE points."
    )


if __name__ == "__main__":
    main()
