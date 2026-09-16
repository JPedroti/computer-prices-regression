"""Experiment 11 -- how much overfitting margin does each finalist have?

The official rule is worth 15 points and is binary: the bootstrap confidence
intervals of the train and validation RMSE either overlap or they do not. A
candidate that passes with a hair's margin is a bad bet, because the margin
depends on which rows happen to land in validation.

**The sealed holdout may not be used for this comparison.** Using it to choose
between candidates would turn it into a selection set and destroy the meaning of
the final number. So the rule is rehearsed here on an *inner* split carved out of
the development data only — the same shape as the real thing (80/20), the same
bootstrap procedure — and repeated over several inner splits so the margin can be
reported with its own spread.

The sealed holdout is touched exactly once, in `src/train.py`, after the model
has been chosen.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import BOOTSTRAP_N, ROBUSTNESS_SEEDS
from src.data import load_dev_holdout
from src.evaluate import overfitting_report
from src.features import FeatureConfig
from src.models import build_model

BASE = FeatureConfig()

FINALISTS: dict[str, dict] = {
    "ridge|levels_plus a=100": dict(name="ridge", preprocessor="levels_plus", alpha=100.0),
    "ridge|levels a=10": dict(name="ridge", preprocessor="levels", alpha=10.0),
    "catboost d6 it800": dict(name="cat", iterations=800, learning_rate=0.06, depth=6),
    "hgb it400": dict(name="hgb", max_iter=400, learning_rate=0.06),
    "lgbm n600": dict(name="lgbm", n_estimators=600, learning_rate=0.05, num_leaves=31),
}

INNER_SEEDS = ROBUSTNESS_SEEDS[:3]


def main() -> None:
    X_dev, y_dev, _, _ = load_dev_holdout()
    print(f"development rows={len(X_dev)}; sealed holdout NOT used in this experiment\n")

    rows = []
    for label, kwargs in FINALISTS.items():
        name = dict(kwargs).pop("name")
        params = {k: v for k, v in kwargs.items() if k != "name"}

        for seed in INNER_SEEDS:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_dev, y_dev, test_size=0.20, random_state=seed
            )
            model = build_model(name, feature_config=BASE, seed=seed, **params)
            model.fit(X_tr, y_tr)
            report = overfitting_report(
                y_tr, model.predict(X_tr), y_va, model.predict(X_va), n_boot=BOOTSTRAP_N
            )
            d = report.to_dict()
            rows.append(
                {
                    "model": label,
                    "inner_seed": seed,
                    "train_rmse": d["train_rmse"],
                    "val_rmse": d["validation_rmse"],
                    "gap": d["validation_rmse"] - d["train_rmse"],
                    "train_ci_high": d["train_ci_high"],
                    "val_ci_low": d["validation_ci_low"],
                    "margin": d["margin"],
                    "overlap": d["intervals_overlap"],
                    "points": d["points"],
                }
            )
            print(
                f"  {label:26s} seed={seed:<5d} train={d['train_rmse']:7.2f} "
                f"val={d['validation_rmse']:7.2f} gap={rows[-1]['gap']:+6.2f} "
                f"margin={d['margin']:+7.2f} {'PASS' if d['intervals_overlap'] else 'FAIL'}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    summary = (
        table.groupby("model")
        .agg(
            mean_gap=("gap", "mean"),
            mean_margin=("margin", "mean"),
            worst_margin=("margin", "min"),
            passes=("overlap", "sum"),
            n=("overlap", "size"),
        )
        .sort_values("worst_margin", ascending=False)
    )
    print("\n=== overfitting margin across inner splits ===")
    print(summary.round(2).to_string())
    print(
        "\nmargin = upper end of the train CI minus the lower end of the validation CI.\n"
        "Positive means the intervals overlap, so the rule awards 15 points;\n"
        "the size of the margin is the safety buffer against a different split."
    )

    out = Path(__file__).resolve().parents[1] / "experiments" / "exp11_overfitting_margin.csv"
    table.to_csv(out, index=False)
    print(f"\nsaved -> {out.name}")


if __name__ == "__main__":
    main()
