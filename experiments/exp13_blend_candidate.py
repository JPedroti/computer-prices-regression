"""Experiment 13 -- is a two-member blend the right final model?

What exp07 established (pooled out-of-fold RMSE):

    CatBoost alone                211.686
    Ridge levels_plus alone       212.341
    NNLS over all five members    211.308   (honest half-split gain +0.437)
    NNLS over Ridge + CatBoost    211.333

The five-member blend beats the two-member one by 0.025 RMSE, which is nothing.
So almost the entire ensemble gain comes from combining the additive Ridge with
CatBoost; HistGradientBoosting, LightGBM and the one-hot Ridge contribute
essentially zero. That matters, because LightGBM **fails the overfitting rule on
its own** (exp11), and there is no reason to carry a member that adds no accuracy
and does carry risk.

This experiment therefore:

1. recovers the two-member NNLS weights from the cached out-of-fold predictions;
2. measures the resulting blend's overfitting margin on inner splits, the same
   way exp11 did for the single models, so the comparison is like for like.

The sealed holdout is not touched.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import BOOTSTRAP_N, ROBUSTNESS_SEEDS
from src.data import load_dev_holdout
from src.ensemble import WeightedEnsemble, blend_search
from src.evaluate import overfitting_report, rmse
from src.features import FeatureConfig
from src.models import build_model

OOF_CACHE = Path(__file__).resolve().parents[1] / "experiments" / "exp07_oof.npz"
INNER_SEEDS = ROBUSTNESS_SEEDS[:3]
BASE = FeatureConfig()

# Which cached out-of-fold vectors correspond to the two members.
OOF_KEYS = ("ridge_levels", "cat_tuned")


def members(seed: int, specs):
    """Build the blend's members from the final-model configuration itself, so
    this experiment can never drift from what `src/train.py` will fit."""
    return [
        (
            spec.get("name", spec["model"]),
            build_model(
                spec["model"],
                feature_config=BASE,
                preprocessor=spec.get("preprocessor"),
                seed=seed,
                **spec.get("hyperparameters", {}),
            ),
        )
        for spec in specs
    ]


def main() -> None:
    import json

    if not OOF_CACHE.exists():
        raise SystemExit("run experiments/exp07_ensembles.py first to build the OOF cache")

    config_path = Path(__file__).resolve().parents[1] / "configs" / "final_model.json"
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    specs = config["members"]
    print(f"members from {config_path.name}: "
          + ", ".join(f"{s.get('name', s['model'])} {s.get('hyperparameters', {})}" for s in specs))

    with np.load(OOF_CACHE, allow_pickle=True) as data:
        oof = {k: data[k] for k in data.files}
    y = oof.pop("y_true")

    missing = [k for k in OOF_KEYS if k not in oof]
    if missing:
        raise SystemExit(f"missing cached OOF for {missing}; rerun exp07_ensembles.py")

    pair = {k: oof[k] for k in OOF_KEYS}
    weights, blended_rmse = blend_search(pair, y)
    print("\n=== two-member blend, weights from out-of-fold predictions ===")
    for k, v in weights.items():
        print(f"  {k:14s} {v:.4f}   (alone: {rmse(y, oof[k]):.3f})")
    print(f"  pooled OOF RMSE : {blended_rmse:.3f}")

    fixed = [float(weights[k]) for k in OOF_KEYS]

    # ------------------------------------------------------------------ #
    # Overfitting margin on inner splits, comparable with exp11
    # ------------------------------------------------------------------ #
    X_dev, y_dev, _, _ = load_dev_holdout()
    print("\n=== overfitting rule rehearsed on inner splits (holdout untouched) ===")
    rows = []
    for seed in INNER_SEEDS:
        X_tr, X_va, y_tr, y_va = train_test_split(X_dev, y_dev, test_size=0.20, random_state=seed)
        model = WeightedEnsemble(members=members(seed, specs), weights=fixed)
        model.fit(X_tr, y_tr)
        report = overfitting_report(
            y_tr, model.predict(X_tr), y_va, model.predict(X_va), n_boot=BOOTSTRAP_N
        )
        d = report.to_dict()
        rows.append({
            "inner_seed": seed,
            "train_rmse": d["train_rmse"],
            "val_rmse": d["validation_rmse"],
            "gap": d["validation_rmse"] - d["train_rmse"],
            "margin": d["margin"],
            "overlap": d["intervals_overlap"],
        })
        print(f"  seed={seed:<5d} train={d['train_rmse']:7.2f} val={d['validation_rmse']:7.2f} "
              f"gap={rows[-1]['gap']:+6.2f} margin={d['margin']:+7.2f} "
              f"{'PASS' if d['intervals_overlap'] else 'FAIL'}", flush=True)

    table = pd.DataFrame(rows)
    print(f"\nmean margin  : {table['margin'].mean():+.2f}")
    print(f"worst margin : {table['margin'].min():+.2f}")
    print(f"passes       : {int(table['overlap'].sum())}/{len(table)}")
    print("\nfor comparison (exp11): Ridge alone     +27.74 mean / +25.20 worst")
    print("                        CatBoost alone  +20.11 mean / +17.53 worst")
    print("                        HistGB alone     +9.40 mean /  +7.50 worst")
    print("                        LightGBM alone  -12.96 mean, fails on every split")

    out = Path(__file__).resolve().parents[1] / "experiments" / "exp13_blend_margin.csv"
    table.to_csv(out, index=False)
    print(f"\nsaved -> {out.name}")


if __name__ == "__main__":
    main()
