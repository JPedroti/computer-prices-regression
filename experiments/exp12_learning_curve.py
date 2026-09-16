"""Experiment 12 -- how much does the protocol cost in accuracy?

CLAUDE.md section 17 says the final model is trained on the training split only,
which means the delivered model sees 64,000 rows rather than all 80,000. That is
a real cost against the secret test, and it should be measured rather than
assumed away.

A learning curve answers it directly: hold the validation rows fixed and grow the
training set. If the curve has flattened by 64k, the protocol costs almost
nothing and can be followed literally with a clear conscience. If it is still
falling steeply, the trade-off deserves a second look.
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

from src.config import ROBUSTNESS_SEEDS
from src.data import load_dev_holdout
from src.evaluate import rmse
from src.features import FeatureConfig
from src.models import build_model

# The validation block takes 12,000 rows, so the training pool is 52,000.
SIZES = [4000, 8000, 16000, 32000, 44000, 52000]
SEEDS = ROBUSTNESS_SEEDS[:3]


def main() -> None:
    X_dev, y_dev, _, _ = load_dev_holdout()
    cfg = FeatureConfig()
    rows = []

    for seed in SEEDS:
        # A fixed validation block per seed, so only the training size varies.
        X_pool, X_val, y_pool, y_val = train_test_split(
            X_dev, y_dev, test_size=12000, random_state=seed
        )
        for size in SIZES:
            if size > len(X_pool):
                continue
            X_tr = X_pool.iloc[:size]
            y_tr = y_pool.iloc[:size]
            model = build_model("ridge", feature_config=cfg, preprocessor="levels_plus",
                                seed=seed, alpha=100.0)
            model.fit(X_tr, y_tr)
            r = rmse(y_val, model.predict(X_val))
            rows.append({"seed": seed, "n_train": size, "val_rmse": r})
            print(f"  seed={seed:<5d} n_train={size:6d}  valRMSE={r:8.3f}", flush=True)

    table = pd.DataFrame(rows)
    curve = table.groupby("n_train")["val_rmse"].agg(["mean", "std"]).round(3)
    print("\n=== learning curve (validation block fixed at 12,000 rows) ===")
    print(curve.to_string())

    # Extrapolate the remaining gain from 64k to 80k.
    # For a model with p parameters, expected squared error scales roughly as
    # sigma^2 (1 + p/n), so the curve is fitted as rmse^2 = a + b/n.
    x = 1.0 / curve.index.to_numpy(dtype=float)
    y = curve["mean"].to_numpy() ** 2
    b, a = np.polyfit(x, y, 1)
    at_64k = float(np.sqrt(a + b / 64000))
    at_80k = float(np.sqrt(a + b / 80000))
    print(f"\nfitted rmse^2 = {a:.1f} + {b:.3g}/n")
    print(f"  implied RMSE at n=64,000 (the protocol) : {at_64k:.3f}")
    print(f"  implied RMSE at n=80,000 (all the data) : {at_80k:.3f}")
    print(f"  cost of training on the split only      : {at_64k - at_80k:+.3f} RMSE")

    measured = curve["mean"]
    last, prev = measured.index[-1], measured.index[-2]
    print(f"\nmeasured drop from {prev:,} to {last:,} rows: "
          f"{measured.loc[prev] - measured.loc[last]:+.3f}")
    print("If the curve is flat at this end, the protocol in section 17 can be")
    print("followed literally at negligible cost, which is the defensible choice.")

    out = Path(__file__).resolve().parents[1] / "experiments" / "exp12_learning_curve.csv"
    table.to_csv(out, index=False)
    print(f"\nsaved -> {out.name}")


if __name__ == "__main__":
    main()
