"""Experiment 05 -- ablation of every feature family.

CLAUDE.md section 10 requires each family of engineered features to be judged
on evidence rather than added wholesale. Each row below removes one family from
the working set or adds one candidate family, and is scored on the same folds
as everything else so the differences are paired.

The reference specification is whichever additive representation won
experiment 04; it is passed in via ``--preprocessor`` and ``--alpha`` so this
script does not hard-code a result that experiments may later revise.
"""
from __future__ import annotations

import argparse
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

BASE = dict(model_decomp=True, cpu_decomp=True, structural_zeros=True, resolution=True)

VARIANTS: dict[str, dict] = {
    "base": {},
    "-model_decomp": {"model_decomp": False},
    "-cpu_decomp": {"cpu_decomp": False},
    "-structural_zeros": {"structural_zeros": False},
    "-resolution": {"resolution": False},
    "+cpu_generation": {"cpu_generation": True},
    "+capacity": {"capacity": True},
    "+premium": {"premium": True},
    "+capacity+premium": {"capacity": True, "premium": True},
    "all_on": {"cpu_generation": True, "capacity": True, "premium": True},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessor", default="levels_plus")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--model", default="ridge")
    args = parser.parse_args()

    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()
    strategy = f"KFold({CV_FOLDS},seed={RANDOM_SEED})"
    print(f"reference spec: {args.model}|{args.preprocessor} alpha={args.alpha}\n")

    results = {}
    for label, overrides in VARIANTS.items():
        cfg = FeatureConfig(**{**BASE, **overrides})
        model = build_model(
            args.model, feature_config=cfg, preprocessor=args.preprocessor,
            seed=RANDOM_SEED, alpha=args.alpha,
        )
        res = cross_validate_model(
            model, X_dev, y_dev, name=f"{label}",
            n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED, return_oof=False,
        )
        results[label] = res
        print(res.summary(), flush=True)
        tracker.log(
            model=f"{args.model}|{args.preprocessor}",
            features=cfg.enabled(),
            preprocessing=args.preprocessor,
            hyperparameters={"alpha": args.alpha},
            seed=RANDOM_SEED,
            validation_strategy=strategy,
            train_rmse=res.train_rmse,
            validation_rmse=res.val_rmse,
            train_mae=res.train_mae,
            validation_mae=res.val_mae,
            validation_rmse_std=res.val_rmse_std,
            fit_seconds=res.fit_seconds,
            notes=f"exp05 feature ablation: {label}",
        )

    base = results["base"]
    print("\n=== paired difference vs base (negative = better) ===")
    rows = []
    for label, res in results.items():
        diff = np.array(res.fold_val_rmse) - np.array(base.fold_val_rmse)
        rows.append((label, res.val_rmse, diff.mean(), diff.std(ddof=1), np.abs(diff.mean()) > 2 * diff.std(ddof=1) / np.sqrt(len(diff))))
    for label, val, m, sd, sig in sorted(rows, key=lambda r: r[2]):
        flag = "significant" if sig and label != "base" else ""
        print(f"  {label:20s} valRMSE={val:8.3f}  diff={m:+7.3f} (sd {sd:5.3f})  {flag}")


if __name__ == "__main__":
    main()
