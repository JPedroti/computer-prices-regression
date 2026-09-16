"""Experiment 09 -- robustness of the finalists across seeds and folds.

A single cross-validation run fixes one particular partition of the data. Here
each finalist is re-scored over several fold partitions, and what is checked is
whether the *ranking* survives, judged on the paired per-fold differences (the
absolute spread is dominated by how many extreme prices land in each fold, which
cancels when models share the fold).

Efficiency note: the blend is a fixed-weight combination of the two members, so
its fold predictions are the weighted sum of theirs. Fitting each member once per
fold and deriving the blend from those predictions gives all three candidates for
the price of two, instead of refitting CatBoost a second time inside the blend.
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
from sklearn.base import clone
from sklearn.model_selection import KFold

from src.config import CV_FOLDS, ROBUSTNESS_SEEDS
from src.data import load_dev_holdout
from src.evaluate import mae, rmse
from src.features import FeatureConfig
from src.models import build_model
from src.utils import ExperimentTracker

BASE = FeatureConfig()
SEEDS = ROBUSTNESS_SEEDS[:3]
BLEND_WEIGHTS = {"ridge|levels_plus": 0.3717, "catboost": 0.6283}


def members(seed: int) -> dict:
    return {
        "ridge|levels_plus": build_model(
            "ridge", feature_config=BASE, preprocessor="levels_plus", seed=seed, alpha=100.0
        ),
        "catboost": build_model(
            "cat", feature_config=BASE, seed=seed, iterations=800, learning_rate=0.06, depth=6
        ),
    }


def main() -> None:
    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()
    X_dev = X_dev.reset_index(drop=True)
    y = pd.Series(np.asarray(y_dev, dtype=float))

    candidates = list(BLEND_WEIGHTS) + ["blend"]
    folds: dict[str, list[float]] = {c: [] for c in candidates}
    trains: dict[str, list[float]] = {c: [] for c in candidates}
    maes: dict[str, list[float]] = {c: [] for c in candidates}
    seed_means: dict[str, dict[int, list[float]]] = {c: {} for c in candidates}

    for seed in SEEDS:
        specs = members(seed)
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        per_seed: dict[str, list[float]] = {c: [] for c in candidates}

        for tr_idx, va_idx in kf.split(X_dev):
            X_tr, X_va = X_dev.iloc[tr_idx], X_dev.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            val_pred, train_pred = {}, {}
            for label, spec in specs.items():
                model = clone(spec)
                model.fit(X_tr, y_tr)
                val_pred[label] = model.predict(X_va)
                train_pred[label] = model.predict(X_tr)

            # The blend is linear in its members, so it needs no extra fitting.
            val_pred["blend"] = sum(BLEND_WEIGHTS[k] * v for k, v in val_pred.items())
            train_pred["blend"] = sum(BLEND_WEIGHTS[k] * v for k, v in train_pred.items())

            for c in candidates:
                r = rmse(y_va, val_pred[c])
                folds[c].append(r)
                per_seed[c].append(r)
                trains[c].append(rmse(y_tr, train_pred[c]))
                maes[c].append(mae(y_va, val_pred[c]))

        for c in candidates:
            seed_means[c][seed] = float(np.mean(per_seed[c]))
            print(f"  {c:20s} seed={seed:<5d} valRMSE={seed_means[c][seed]:8.3f}", flush=True)
        print(flush=True)

    summary = pd.DataFrame(
        [
            {
                "model": c,
                "mean_rmse": float(np.mean(folds[c])),
                "sd_across_folds": float(np.std(folds[c], ddof=1)),
                "sd_across_seed_means": float(np.std(list(seed_means[c].values()), ddof=1)),
                "worst_seed": float(max(seed_means[c].values())),
                "best_seed": float(min(seed_means[c].values())),
                "mean_train_rmse": float(np.mean(trains[c])),
                "mean_gap": float(np.mean(folds[c]) - np.mean(trains[c])),
                "mean_val_mae": float(np.mean(maes[c])),
            }
            for c in candidates
        ]
    ).sort_values("mean_rmse").reset_index(drop=True)

    print("=== robustness summary over "
          f"{len(SEEDS)} seeds x {CV_FOLDS} folds = {len(folds[candidates[0]])} folds ===")
    print(summary.round(3).to_string(index=False))

    best = summary.iloc[0]["model"]
    print(f"\n=== paired per-fold differences vs {best} ===")
    ref = np.array(folds[best])
    for label in summary["model"]:
        if label == best:
            continue
        diff = np.array(folds[label]) - ref
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"  {label:20s} diff={diff.mean():+7.3f} +-{se:5.3f}  "
              f"better on {int((diff < 0).sum())}/{len(diff)} folds")

    print("\n=== per-seed means ===")
    print(pd.DataFrame(seed_means).round(3).to_string())

    for c in candidates:
        row = summary[summary["model"] == c].iloc[0]
        tracker.log(
            model=c, features=BASE.enabled(),
            preprocessing="mixed" if c == "blend" else ("levels_plus" if "ridge" in c else "native"),
            hyperparameters=BLEND_WEIGHTS if c == "blend" else {},
            seed=-1,
            validation_strategy=f"{len(SEEDS)}x KFold({CV_FOLDS}) seeds={list(SEEDS)}",
            train_rmse=row["mean_train_rmse"], validation_rmse=row["mean_rmse"],
            train_mae=float("nan"), validation_mae=row["mean_val_mae"],
            validation_rmse_std=row["sd_across_folds"],
            notes=f"exp09 multi-seed robustness ({len(folds[c])} folds)",
        )

    out = Path(__file__).resolve().parents[1] / "experiments" / "exp09_robustness.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "summary": summary.to_dict("records"),
                "per_seed_means": {k: {str(s): v for s, v in d.items()} for k, d in seed_means.items()},
                "fold_rmse": folds,
            },
            fh, indent=2,
        )
    print(f"\nsaved -> {out.name}")


if __name__ == "__main__":
    main()
