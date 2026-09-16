"""Experiment 06 -- do explicit interactions buy anything?

Experiment 02 concluded "no exploitable interactions" from an indirect test:
a booster trained on the additive model's residual made predictions worse and
showed flat, diffuse importances. That test can fail for the wrong reason -- a
booster can simply overfit the residual noise -- so this experiment makes the
question direct.

Concrete interaction terms are appended to the winning additive design and
scored on the same folds. If a real interaction exists, adding exactly that
product of dummies must lower the cross-validated RMSE; a linear model cannot
"overfit its way" into a paired improvement across all five folds.

Candidate pairs are chosen from domain structure rather than from a search over
the target, so no selection leakage is introduced.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, RANDOM_SEED
from src.data import load_dev_holdout
from src.evaluate import cross_validate_model
from src.features import FeatureConfig, FeatureEngineer
from src.models import _make_estimator
from src.preprocessing import ColumnTyper
from src.utils import ExperimentTracker

# Pairs a domain expert would expect to interact if anything does:
# a brand premium applied to the components, a form-factor dependent pricing of
# the GPU, laptop-vs-desktop treatment of shared components, and so on.
CANDIDATE_PAIRS: list[tuple[str, str]] = [
    ("brand", "cpu_family"),
    ("brand", "gpu_model"),
    ("brand", "device_type"),
    ("device_type", "gpu_model"),
    ("device_type", "cpu_family"),
    ("form_factor", "gpu_model"),
    ("os", "cpu_family"),
    ("device_type", "storage_type"),
    ("brand", "model_line"),
    ("cpu_family", "gpu_model"),
]


class AddInteractions(BaseEstimator, TransformerMixin):
    """Append the string-concatenation of selected column pairs as new columns.

    Concatenating the two level labels and letting the downstream encoder expand
    the result is exactly a full interaction of the two factors.
    """

    def __init__(self, pairs: list[tuple[str, str]] | None = None):
        self.pairs = pairs or []

    def fit(self, X: pd.DataFrame, y=None) -> "AddInteractions":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for a, b in self.pairs:
            if a in out.columns and b in out.columns:
                ka = out[a].astype("string").fillna("NA")
                kb = out[b].astype("string").fillna("NA")
                out[f"{a}_X_{b}"] = (ka + "|" + kb).astype(str)
        return out


def build(pairs, preprocessor: str, alpha: float, cfg: FeatureConfig):
    return Pipeline(
        [
            ("features", FeatureEngineer(cfg)),
            ("interactions", AddInteractions(pairs)),
            ("columns", ColumnTyper(preprocessor)),
            ("model", _make_estimator("ridge", RANDOM_SEED, {"alpha": alpha})),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessor", default="levels_plus")
    parser.add_argument("--alpha", type=float, default=100.0)
    args = parser.parse_args()

    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()
    cfg = FeatureConfig()
    strategy = f"KFold({CV_FOLDS},seed={RANDOM_SEED})"
    print(f"reference: ridge|{args.preprocessor} alpha={args.alpha}\n")

    runs: list[tuple[str, list]] = [("none (additive only)", [])]
    runs += [(f"{a} x {b}", [(a, b)]) for a, b in CANDIDATE_PAIRS]
    runs.append(("all pairs", list(CANDIDATE_PAIRS)))

    results = {}
    for label, pairs in runs:
        model = build(pairs, args.preprocessor, args.alpha, cfg)
        res = cross_validate_model(
            model, X_dev, y_dev, name=label,
            n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED, return_oof=False,
        )
        results[label] = res
        print(res.summary(), flush=True)
        tracker.log(
            model=f"ridge|{args.preprocessor}+interactions",
            features=cfg.enabled() + [f"{a}_X_{b}" for a, b in pairs],
            preprocessing=args.preprocessor,
            hyperparameters={"alpha": args.alpha, "pairs": [f"{a}x{b}" for a, b in pairs]},
            seed=RANDOM_SEED,
            validation_strategy=strategy,
            train_rmse=res.train_rmse,
            validation_rmse=res.val_rmse,
            train_mae=res.train_mae,
            validation_mae=res.val_mae,
            validation_rmse_std=res.val_rmse_std,
            fit_seconds=res.fit_seconds,
            notes=f"exp06 interaction test: {label}",
        )

    base = results["none (additive only)"]
    print("\n=== paired difference vs additive-only (negative = the interaction helps) ===")
    rows = []
    for label, res in results.items():
        if label == "none (additive only)":
            continue
        diff = np.array(res.fold_val_rmse) - np.array(base.fold_val_rmse)
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append((label, res.val_rmse, diff.mean(), se, res.gap))
    for label, val, m, se, gap in sorted(rows, key=lambda r: r[2]):
        verdict = "HELPS" if m < -2 * se else ("hurts" if m > 2 * se else "no effect")
        print(f"  {label:26s} valRMSE={val:8.3f} diff={m:+7.3f} +-{se:5.3f} "
              f"trainGap={gap:+6.2f}  {verdict}")


if __name__ == "__main__":
    main()
