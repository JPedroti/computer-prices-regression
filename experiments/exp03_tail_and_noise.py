"""Experiment 03 -- anatomy of the error tail and the irreducible noise floor.

Motivation
----------
In the initial audit, 10 validation rows (0.06%) carried 30.8% of the total
squared error and the top decile of the *true* price showed a +189 bias.

Statistical caution
-------------------
Conditioning on the true target is misleading: if ``y = f(x) + noise``, then
selecting rows with a high ``y`` also selects rows with a high positive noise
draw, so even a perfect predictor of ``E[y|x]`` shows a positive bias in the
top decile of ``y``. Calibration must therefore be judged on deciles of the
*prediction*, not of the truth. This experiment reports both so the artefact
is visible, and then characterises the noise:

* is the residual spread constant, or proportional to the price level?
* what RMSE floor does that noise imply?

Knowing the floor tells us when further modelling stops paying.
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

from src.config import RANDOM_SEED
from src.data import load_dev_holdout
from src.evaluate import error_concentration, residual_by_decile, rmse
from src.features import FeatureConfig
from src.models import build_model


def decile_table(values: np.ndarray, resid: np.ndarray, label: str) -> pd.DataFrame:
    d = pd.qcut(values, 10, labels=False, duplicates="drop")
    frame = pd.DataFrame({label: values, "err": resid, "d": d})
    return frame.groupby("d").agg(
        n=("err", "size"),
        level=(label, "mean"),
        bias=("err", "mean"),
        resid_sd=("err", "std"),
        rmse=("err", lambda s: float(np.sqrt(np.mean(s**2)))),
    )


def main() -> None:
    X_dev, y_dev, _, _ = load_dev_holdout()
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=RANDOM_SEED
    )

    cfg = FeatureConfig()
    model = build_model("lgbm", feature_config=cfg, n_estimators=600, learning_rate=0.05,
                        num_leaves=31, seed=RANDOM_SEED)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    y_true = y_va.to_numpy(dtype=float)
    resid = y_true - pred
    print(f"reference model: LGBM  valRMSE={rmse(y_true, pred):.3f}\n")

    print("=== error concentration ===")
    print(error_concentration(y_true, pred).to_string(index=False))

    print("\n=== residuals by decile of TRUE price (selection artefact expected) ===")
    print(residual_by_decile(y_true, pred).round(2).to_string())

    print("\n=== residuals by decile of PREDICTED price (true calibration check) ===")
    print(decile_table(pred, resid, "pred").round(2).to_string())

    # ------------------------------------------------------------------ #
    # Noise shape: constant vs proportional
    # ------------------------------------------------------------------ #
    print("\n=== noise shape ===")
    tbl = decile_table(pred, resid, "pred")
    lvl = tbl["level"].to_numpy()
    sd = tbl["resid_sd"].to_numpy()
    # Fit sd = a (constant) versus sd = b * level (proportional).
    a = sd.mean()
    b = float(np.sum(sd * lvl) / np.sum(lvl**2))
    sse_const = float(np.sum((sd - a) ** 2))
    sse_prop = float(np.sum((sd - b * lvl) ** 2))
    print(f"constant model      sd = {a:7.2f}          SSE={sse_const:10.1f}")
    print(f"proportional model  sd = {b:.5f} * price  SSE={sse_prop:10.1f}")
    print(f"-> {'PROPORTIONAL' if sse_prop < sse_const else 'CONSTANT'} noise fits the spread better")
    print(f"   implied relative noise: {b*100:.2f}% of price")

    ratio = resid / np.maximum(pred, 1.0)
    print(f"\nrelative residual (resid/pred): sd={ratio.std():.5f} "
          f"mean={ratio.mean():+.5f} skew={pd.Series(ratio).skew():.3f}")
    print(f"absolute residual              : sd={resid.std():.3f} "
          f"mean={resid.mean():+.3f} skew={pd.Series(resid).skew():.3f}")

    # If noise is proportional, the RMSE floor is sqrt(E[(b*price)^2]).
    floor_prop = float(np.sqrt(np.mean((b * y_true) ** 2)))
    print(f"\nimplied RMSE floor if noise is purely proportional: {floor_prop:.2f}")
    print(f"current model RMSE                                 : {rmse(y_true, pred):.2f}")

    # ------------------------------------------------------------------ #
    # Who are the worst rows?
    # ------------------------------------------------------------------ #
    print("\n=== 15 worst validation rows ===")
    worst = np.argsort(np.abs(resid))[::-1][:15]
    cols = ["device_type", "brand", "os", "cpu_model", "gpu_model", "ram_gb",
            "storage_gb", "gpu_tier", "cpu_tier"]
    show = X_va.iloc[worst][cols].copy()
    show.insert(0, "residual", resid[worst].round(1))
    show.insert(0, "pred", pred[worst].round(1))
    show.insert(0, "price", y_true[worst])
    print(show.to_string(index=False))

    # Are the worst rows unusual, or ordinary rows with a big noise draw?
    print("\n=== are large residuals concentrated in any segment? ===")
    big = np.abs(resid) > np.percentile(np.abs(resid), 99)
    for col in ["device_type", "brand", "os", "form_factor", "cpu_brand"]:
        share_all = X_va[col].value_counts(normalize=True)
        share_big = X_va.loc[big, col].value_counts(normalize=True)
        lift = (share_big / share_all).dropna().sort_values(ascending=False)
        print(f"  {col:12s} " + ", ".join(f"{k}={v:.2f}x" for k, v in lift.head(4).items()))


if __name__ == "__main__":
    main()
