"""Experiment 07 -- do ensembles beat the best single model?

Out-of-fold predictions are produced once per candidate on the shared folds and
then combined offline, so many blends can be compared without refitting. Weights
are fitted on out-of-fold predictions, which is the only leak-free way to learn
them: every prediction being combined was made by a model that had not seen that
row.

Two measurement points that are easy to get wrong, and are handled explicitly:

1. **Aggregation must match.** The mean of per-fold RMSEs is not the RMSE of the
   pooled out-of-fold vector; the two differ by a few tenths here because fold
   RMSEs are dominated by how many extreme prices each fold happened to get.
   Comparing a blend's pooled RMSE against a single model's mean-of-folds would
   flatter or penalise the blend for no reason, so every number in the blend
   table below is the pooled out-of-fold RMSE.
2. **Weights fitted and scored on the same vector are optimistic.** With five
   members that optimism is tiny, but it is measured rather than assumed: the
   weights are refitted on one half of the rows and scored on the other.

A blend is adopted only if it beats the best single model by more than that
honest margin (CLAUDE.md section 16).
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
OOF_CACHE = Path(__file__).resolve().parents[1] / "experiments" / "exp07_oof.npz"

CANDIDATES: dict[str, tuple[str, str | None, dict, FeatureConfig]] = {
    "ridge_levels": ("ridge", "levels_plus", {"alpha": 100.0}, BASE),
    "ridge_onehot": ("ridge", "onehot", {"alpha": 10.0}, BASE),
    "hgb": ("hgb", None, {"max_iter": 400, "learning_rate": 0.06}, BASE),
    "lgbm": ("lgbm", None, {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31}, BASE),
    "cat": ("cat", None, {"iterations": 800, "learning_rate": 0.06, "depth": 6}, BASE),
    # Best configuration from the exp10 tuning sweep (-0.193 +-0.031 paired).
    "cat_tuned": ("cat", None, {"iterations": 1500, "learning_rate": 0.03, "depth": 6}, BASE),
}


def main() -> None:
    tracker = ExperimentTracker()
    X_dev, y_dev, _, _ = load_dev_holdout()
    y = y_dev.to_numpy(dtype=float)
    strategy = f"KFold({CV_FOLDS},seed={RANDOM_SEED})"

    oof: dict[str, np.ndarray] = {}
    fold_mean: dict[str, float] = {}
    folds: dict[str, list[float]] = {}

    cached = {}
    if OOF_CACHE.exists():
        with np.load(OOF_CACHE, allow_pickle=True) as data:
            cached = {k: data[k] for k in data.files}
        print(f"reusing cached out-of-fold predictions for: {sorted(cached)}\n")

    for label, (name, prep, params, cfg) in CANDIDATES.items():
        if label in cached and len(cached[label]) == len(X_dev):
            oof[label] = cached[label]
            fold_mean[label] = float("nan")
            folds[label] = []
            print(f"{label:16s} loaded from cache")
            continue
        model = build_model(name, feature_config=cfg, preprocessor=prep, seed=RANDOM_SEED, **params)
        res = cross_validate_model(
            model, X_dev, y_dev, name=label,
            n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED, return_oof=True,
        )
        oof[label] = res.oof_pred
        fold_mean[label] = res.val_rmse
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

    np.savez_compressed(OOF_CACHE, **oof, y_true=y)
    print(f"\ncached out-of-fold predictions -> {OOF_CACHE.name}")

    # ------------------------------------------------------------------ #
    # Consistent baseline: pooled out-of-fold RMSE of each single model
    # ------------------------------------------------------------------ #
    pooled = {k: rmse(y, v) for k, v in oof.items()}
    print("\n=== single models, both aggregations ===")
    for k in sorted(pooled, key=pooled.get):
        fm = fold_mean.get(k, float("nan"))
        fm_text = f"{fm:8.3f}" if np.isfinite(fm) else "  cached"
        print(f"  {k:16s} pooled OOF RMSE={pooled[k]:8.3f}   mean of fold RMSEs={fm_text}")
    best_single = min(pooled, key=pooled.get)
    print(f"\nbest single model (pooled): {best_single} at {pooled[best_single]:.3f}")

    print("\n=== how different are the members' errors? (residual correlation) ===")
    resid = pd.DataFrame({k: y - v for k, v in oof.items()})
    print(resid.corr().round(4).to_string())

    # ------------------------------------------------------------------ #
    # Blends, scored the same way
    # ------------------------------------------------------------------ #
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
    table["gain_vs_best_single"] = pooled[best_single] - table["rmse"]
    table = table.sort_values("rmse").reset_index(drop=True)
    print("\n=== blends (pooled OOF RMSE) ===")
    print(table.head(10)[["combiner", "members", "rmse", "gain_vs_best_single"]].round(4).to_string(index=False))

    best = table.iloc[0]
    print(f"\nbest blend  : {best['combiner']} over {best['members']}")
    print(f"blend RMSE  : {best['rmse']:.3f}")
    print(f"best single : {pooled[best_single]:.3f}  ({best_single})")
    print(f"gain        : {best['gain_vs_best_single']:+.3f}")
    if best["weights"]:
        print("weights     :", {k: round(v, 4) for k, v in best["weights"].items() if v > 1e-6})

    # ------------------------------------------------------------------ #
    # Honest check: fit the weights on one half, score on the other
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(RANDOM_SEED)
    half = rng.permutation(len(y))
    a, b = half[: len(y) // 2], half[len(y) // 2 :]
    members = best["members"].split("+")
    fit_sub = {k: oof[k][a] for k in members}
    w, _ = blend_search(fit_sub, y[a])
    held = np.column_stack([oof[k][b] for k in members]) @ np.array([w[k] for k in members])
    honest = rmse(y[b], held)
    single_on_b = rmse(y[b], oof[best_single][b])
    print("\n=== weights fitted on half the rows, scored on the other half ===")
    print(f"  blend on held-out half  : {honest:.3f}")
    print(f"  {best_single} on the same half : {single_on_b:.3f}")
    print(f"  honest gain             : {single_on_b - honest:+.3f}")

    table.drop(columns=["weights"]).to_csv(
        Path(__file__).resolve().parents[1] / "experiments" / "exp07_blends.csv", index=False
    )
    tracker.log(
        model=f"blend[{best['members']}]",
        features=BASE.enabled(), preprocessing="mixed",
        hyperparameters={"combiner": best["combiner"], "weights": best["weights"]},
        seed=RANDOM_SEED, validation_strategy=strategy + " pooled OOF blend",
        train_rmse=float("nan"), validation_rmse=float(best["rmse"]),
        train_mae=float("nan"), validation_mae=float("nan"),
        notes=(f"exp07 best blend; pooled gain vs best single ({best_single}) "
               f"= {best['gain_vs_best_single']:+.3f}; honest half-split gain = {single_on_b - honest:+.3f}"),
    )


if __name__ == "__main__":
    main()
