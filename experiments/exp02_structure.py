"""Experiment 02 -- recover the shape of the generating process.

Hypothesis
----------
The audit suggested the price is close to a sum of per-component contributions.
If that is true, a *saturated additive model* -- one-hot encoding every feature
at its own discrete levels, with no interactions -- should come very close to
the best achievable error, and whatever a gradient booster adds on top is
interaction structure.

This experiment therefore:

1. fits a saturated additive Ridge (all features as categorical levels);
2. measures how much a booster can still extract from its residual;
3. ranks candidate two-way interactions by the residual variance they explain.

Everything is measured with an honest train/validation separation inside the
development split; the sealed overfitting holdout is not touched.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.config import RANDOM_SEED
from src.data import load_dev_holdout
from src.evaluate import rmse, mae
from src.features import FeatureConfig, FeatureEngineer

# Features whose discrete levels are treated as categories for the saturated model.
MAX_LEVELS = 60


def as_levels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Represent every low-cardinality column by its discrete levels.

    Returns the frame plus the explicit column groups: pandas 3 gives
    ``.astype(str)`` the ``str`` dtype rather than ``object``, so the groups are
    tracked directly instead of being re-inferred from dtypes.
    """
    out = pd.DataFrame(index=df.index)
    level_cols: list[str] = []
    numeric_cols: list[str] = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= MAX_LEVELS:
            out[col] = df[col].astype(str)
            level_cols.append(col)
        else:
            out[col] = pd.to_numeric(df[col], errors="coerce")  # genuinely continuous
            numeric_cols.append(col)
    return out, level_cols, numeric_cols


def main() -> None:
    X_dev, y_dev, _, _ = load_dev_holdout()
    fe = FeatureEngineer(FeatureConfig()).fit(X_dev)
    F = fe.transform(X_dev)
    y = y_dev.to_numpy(dtype=float)

    print(f"rows={len(F)} engineered features={F.shape[1]}")
    levels, cat_cols, num_cols = as_levels(F)
    n_levels = {c: levels[c].nunique() for c in levels.columns}
    print(f"treated as categorical: {len(cat_cols)}  | left numeric: {num_cols}")
    print("levels per column:", {k: v for k, v in sorted(n_levels.items(), key=lambda kv: -kv[1])})

    idx_tr, idx_va = train_test_split(
        np.arange(len(F)), test_size=0.25, random_state=RANDOM_SEED
    )

    # ---------------------------------------------------------------- #
    # 1. Saturated additive model (main effects only)
    # ---------------------------------------------------------------- #
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    A_tr = enc.fit_transform(levels.iloc[idx_tr][cat_cols])
    A_va = enc.transform(levels.iloc[idx_va][cat_cols])
    if num_cols:
        extra_tr = np.nan_to_num(levels.iloc[idx_tr][num_cols].to_numpy(dtype=float))
        extra_va = np.nan_to_num(levels.iloc[idx_va][num_cols].to_numpy(dtype=float))
        from scipy.sparse import hstack, csr_matrix

        A_tr = hstack([A_tr, csr_matrix(extra_tr)]).tocsr()
        A_va = hstack([A_va, csr_matrix(extra_va)]).tocsr()

    print(f"\nsaturated design matrix: {A_tr.shape[1]} columns")
    best = None
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        ridge = Ridge(alpha=alpha, solver="sparse_cg", random_state=RANDOM_SEED)
        ridge.fit(A_tr, y[idx_tr])
        p_tr, p_va = ridge.predict(A_tr), ridge.predict(A_va)
        r = rmse(y[idx_va], p_va)
        print(f"  additive Ridge alpha={alpha:<6} trainRMSE={rmse(y[idx_tr],p_tr):8.3f} "
              f"valRMSE={r:8.3f} valMAE={mae(y[idx_va],p_va):8.3f}")
        if best is None or r < best[0]:
            best = (r, alpha, ridge, p_tr, p_va)

    val_rmse_additive, alpha, ridge, p_tr, p_va = best
    print(f"\nbest additive model: alpha={alpha} valRMSE={val_rmse_additive:.3f}")

    # ---------------------------------------------------------------- #
    # 1b. Is the process multiplicative instead of additive?
    #     If price = base * factors, additivity holds in log space.
    #     Exponentiating a log-space fit returns a conditional median, so a
    #     smearing correction is applied before comparing RMSE fairly.
    # ---------------------------------------------------------------- #
    ridge_log = Ridge(alpha=alpha, solver="sparse_cg", random_state=RANDOM_SEED)
    ridge_log.fit(A_tr, np.log(y[idx_tr]))
    lg_tr = ridge_log.predict(A_tr)
    lg_va = ridge_log.predict(A_va)
    smear = float(np.mean(np.exp(np.log(y[idx_tr]) - lg_tr)))  # Duan smearing factor
    for name, corr in (("raw exp", 1.0), (f"smearing x{smear:.4f}", smear)):
        pv = np.exp(lg_va) * corr
        print(f"  multiplicative (log-space) Ridge [{name:>18}] valRMSE={rmse(y[idx_va], pv):8.3f}")
    print(f"  -> additive={val_rmse_additive:.3f} vs multiplicative="
          f"{rmse(y[idx_va], np.exp(lg_va) * smear):.3f}: "
          f"{'ADDITIVE' if val_rmse_additive < rmse(y[idx_va], np.exp(lg_va)*smear) else 'MULTIPLICATIVE'} wins")

    # ---------------------------------------------------------------- #
    # 2. What is left for a booster to find?
    # ---------------------------------------------------------------- #
    from lightgbm import LGBMRegressor

    Xb = F.copy()
    for c in Xb.columns:
        if Xb[c].dtype == object or str(Xb[c].dtype) in ("string", "str"):
            Xb[c] = Xb[c].astype("category")

    booster = LGBMRegressor(
        n_estimators=1500, learning_rate=0.05, num_leaves=63,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    booster.fit(Xb.iloc[idx_tr], y[idx_tr])
    b_va = booster.predict(Xb.iloc[idx_va])
    print(f"direct LGBM      valRMSE={rmse(y[idx_va], b_va):8.3f}")

    resid_tr = y[idx_tr] - p_tr
    resid_va = y[idx_va] - p_va
    print(f"additive residual SD (val) = {resid_va.std():8.3f}")

    booster_r = LGBMRegressor(
        n_estimators=1500, learning_rate=0.05, num_leaves=63,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    booster_r.fit(Xb.iloc[idx_tr], resid_tr)
    r_hat_va = booster_r.predict(Xb.iloc[idx_va])
    boosted = p_va + r_hat_va
    print(f"additive + LGBM(residual) valRMSE={rmse(y[idx_va], boosted):8.3f}  "
          f"(improvement over additive: {val_rmse_additive - rmse(y[idx_va], boosted):+.3f})")

    imp = pd.Series(booster_r.booster_.feature_importance("gain"), index=Xb.columns)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    print("\ntop features carrying the residual structure (% gain):")
    print(imp.head(15).round(2).to_string())

    # ---------------------------------------------------------------- #
    # 3. Explicit two-way interaction screen on the additive residual
    # ---------------------------------------------------------------- #
    print("\nscreening two-way interactions on the additive residual ...")
    cands = [c for c in imp.head(12).index if c in cat_cols]
    lev_tr = levels.iloc[idx_tr]
    base_var = float(np.var(resid_tr))
    rows = []
    for i, a in enumerate(cands):
        for b in cands[i + 1 :]:
            # string dtype keeps <NA> through astype(str); fill it so no group is dropped
            ka = lev_tr[a].astype("string").fillna("NA")
            kb = lev_tr[b].astype("string").fillna("NA")
            key = (ka + "|" + kb).to_numpy(dtype=object)
            grp = pd.Series(resid_tr).groupby(key)
            cell_mean = grp.transform("mean").to_numpy()
            counts = grp.transform("size").to_numpy()
            # Only trust cells with enough support.
            explained = 1.0 - float(np.var(resid_tr - cell_mean)) / base_var
            rows.append({
                "pair": f"{a} x {b}",
                "cells": int(pd.Series(key).nunique()),
                "min_cell": int(counts.min()),
                "variance_explained": explained,
            })
    screen = pd.DataFrame(rows).sort_values("variance_explained", ascending=False)
    print(screen.head(20).to_string(index=False))

    out = Path(__file__).resolve().parents[1] / "experiments" / "exp02_interaction_screen.csv"
    screen.to_csv(out, index=False)
    print(f"\nsaved interaction screen -> {out.name}")


if __name__ == "__main__":
    main()
