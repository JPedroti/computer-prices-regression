"""Experiment 09 -- robustness of the finalists across seeds and folds.

A single cross-validation run fixes one particular partition of the data. Here
each finalist is re-scored with five different fold partitions (seeds 42, 7,
2024, 1337, 99), giving 25 validation folds per model.

What is checked (CLAUDE.md section 15):

* the mean RMSE across all folds and its spread;
* whether the ranking between candidates is stable, judged by the *paired*
  per-fold differences -- the absolute fold spread is dominated by how many
  extreme prices land in a fold, which cancels when models share the fold;
* whether the train/validation gap stays small enough for the overfitting rule.

A candidate that wins on average but loses on several seeds is treated with
caution, as required by CLAUDE.md section 30.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import CV_FOLDS, ROBUSTNESS_SEEDS
from src.data import load_dev_holdout
from src.evaluate import cross_validate_model
from src.features import FeatureConfig
from src.models import build_model
from src.utils import ExperimentTracker

BASE = FeatureConfig()

# label -> kwargs for build_model.
# The three contenders that came out of the screening phase, plus the one-hot
# Ridge as a sanity anchor with a known value.
FINALISTS: dict[str, dict] = {
    "ridge|levels_plus a=100": dict(name="ridge", preprocessor="levels_plus", alpha=100.0),
    "ridge|levels a=10": dict(name="ridge", preprocessor="levels", alpha=10.0),
    "catboost d6 it800": dict(name="cat", iterations=800, learning_rate=0.06, depth=6),
    "ridge|onehot a=10": dict(name="ridge", preprocessor="onehot", alpha=10.0),
}


def main() -> None:
    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()

    per_model_folds: dict[str, list[float]] = {}
    per_model_seed_means: dict[str, dict[int, float]] = {}
    summary_rows = []

    for label, kwargs in FINALISTS.items():
        name = kwargs.pop("name")
        all_folds: list[float] = []
        seed_means: dict[int, float] = {}
        train_rmses, val_maes, gaps = [], [], []

        for seed in ROBUSTNESS_SEEDS:
            # The seed drives both the fold partition and the model's own
            # randomness, so this measures what actually matters: how the
            # candidate behaves on a different draw of everything.
            model = build_model(name, feature_config=BASE, seed=seed, **kwargs)
            res = cross_validate_model(
                model, X_dev, y_dev, name=label,
                n_splits=CV_FOLDS, n_repeats=1, seed=seed, return_oof=False,
            )
            all_folds.extend(res.fold_val_rmse)
            seed_means[seed] = res.val_rmse
            train_rmses.append(res.train_rmse)
            val_maes.append(res.val_mae)
            gaps.append(res.gap)
            print(f"  {label:26s} seed={seed:<5d} valRMSE={res.val_rmse:8.3f} gap={res.gap:+6.2f}",
                  flush=True)

        kwargs["name"] = name
        per_model_folds[label] = all_folds
        per_model_seed_means[label] = seed_means
        row = {
            "model": label,
            "mean_rmse": float(np.mean(all_folds)),
            "sd_across_folds": float(np.std(all_folds, ddof=1)),
            "sd_across_seed_means": float(np.std(list(seed_means.values()), ddof=1)),
            "worst_seed_rmse": float(max(seed_means.values())),
            "best_seed_rmse": float(min(seed_means.values())),
            "mean_train_rmse": float(np.mean(train_rmses)),
            "mean_gap": float(np.mean(gaps)),
            "mean_val_mae": float(np.mean(val_maes)),
        }
        summary_rows.append(row)
        print(f"  -> {label}: {row['mean_rmse']:.3f} (+-{row['sd_across_folds']:.3f} over 25 folds)\n",
              flush=True)

        tracker.log(
            model=label, features=BASE.enabled(),
            preprocessing=kwargs.get("preprocessor", "native"),
            hyperparameters={k: v for k, v in kwargs.items() if k not in ("name", "preprocessor")},
            seed=-1, validation_strategy=f"{len(ROBUSTNESS_SEEDS)}x KFold({CV_FOLDS}) seeds={list(ROBUSTNESS_SEEDS)}",
            train_rmse=row["mean_train_rmse"], validation_rmse=row["mean_rmse"],
            train_mae=float("nan"), validation_mae=row["mean_val_mae"],
            validation_rmse_std=row["sd_across_folds"],
            notes="exp09 multi-seed robustness (25 folds)",
        )

    summary = pd.DataFrame(summary_rows).sort_values("mean_rmse").reset_index(drop=True)
    print("=== robustness summary ===")
    print(summary.round(3).to_string(index=False))

    best = summary.iloc[0]["model"]
    print(f"\n=== paired per-fold differences vs {best} (25 shared folds) ===")
    ref = np.array(per_model_folds[best])
    for label in summary["model"]:
        if label == best:
            continue
        diff = np.array(per_model_folds[label]) - ref
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        wins = int((diff < 0).sum())
        print(f"  {label:26s} diff={diff.mean():+7.3f} +-{se:5.3f}  "
              f"beats reference on {wins}/{len(diff)} folds")

    print("\n=== per-seed means ===")
    print(pd.DataFrame(per_model_seed_means).round(3).to_string())

    out = Path(__file__).resolve().parents[1] / "experiments" / "exp09_robustness.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "summary": summary_rows,
                "per_seed_means": {k: {str(s): v for s, v in d.items()}
                                   for k, d in per_model_seed_means.items()},
            },
            fh, indent=2,
        )
    print(f"\nsaved -> {out.name}")


if __name__ == "__main__":
    main()
