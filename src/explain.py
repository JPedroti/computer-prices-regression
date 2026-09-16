"""SHAP explainability for the final model.

Produces the global and local analyses required by CLAUDE.md section 21.

SHAP attributes a model's prediction to its inputs; it describes what *the
model* does, not what causes prices in the world. No causal reading is implied
anywhere in this module or in the figures it writes.

    python -m src.explain
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, RANDOM_SEED
from .data import load_dev_holdout
from .inference import load_model

DEFAULT_SAMPLE = 2000


def _split_pipeline(model):
    """Return ``(preprocessing, estimator)`` for a pipeline, unwrapping target transforms."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline

    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    if isinstance(model, Pipeline):
        return model[:-1], model[-1]
    raise TypeError(f"cannot decompose model of type {type(model).__name__}")


def _feature_names(prep, n_columns: int) -> list[str]:
    try:
        names = list(prep.get_feature_names_out())
        if len(names) == n_columns:
            return names
    except Exception:
        pass
    return [f"f{i}" for i in range(n_columns)]


def compute_shap(model, X: pd.DataFrame, sample: int = DEFAULT_SAMPLE, seed: int = RANDOM_SEED):
    """Compute SHAP values on a random sample of rows.

    Returns ``(shap_values, transformed_frame, estimator)``. The explainer is
    chosen to match the estimator: exact for linear and tree models.
    """
    import shap

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    X_sample = X.iloc[idx]

    prep, estimator = _split_pipeline(model)
    transformed = prep.transform(X_sample)
    if not isinstance(transformed, pd.DataFrame):
        transformed = pd.DataFrame(
            np.asarray(transformed), columns=_feature_names(prep, np.shape(transformed)[1])
        )
    transformed = transformed.reset_index(drop=True)

    name = type(estimator).__name__
    if name in {"Ridge", "LinearRegression", "Lasso", "ElasticNet"}:
        explainer = shap.LinearExplainer(estimator, transformed)
    else:
        explainer = shap.TreeExplainer(estimator)

    values = explainer(transformed) if hasattr(explainer, "__call__") else explainer.shap_values(transformed)
    return values, transformed, estimator


def global_report(values, transformed: pd.DataFrame, top: int = 20, out_dir: Path = FIGURES_DIR):
    """Global feature ranking plus beeswarm and bar summary plots."""
    import shap

    out_dir.mkdir(parents=True, exist_ok=True)
    arr = values.values if hasattr(values, "values") else np.asarray(values)

    ranking = (
        pd.DataFrame(
            {
                "feature": transformed.columns,
                "mean_abs_shap": np.abs(arr).mean(axis=0),
                "mean_shap": arr.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure()
    shap.summary_plot(arr, transformed, max_display=top, show=False, plot_size=(10, 8))
    plt.title("SHAP global summary -- effect of each feature on predicted price")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(arr, transformed, plot_type="bar", max_display=top, show=False, plot_size=(10, 8))
    plt.title("SHAP global importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    ranking.to_csv(out_dir.parent / "shap_global_ranking.csv", index=False)
    return ranking


def local_report(values, transformed: pd.DataFrame, row: int = 0, out_dir: Path = FIGURES_DIR):
    """Waterfall plot and a signed contribution table for one observation."""
    import shap

    out_dir.mkdir(parents=True, exist_ok=True)
    arr = values.values if hasattr(values, "values") else np.asarray(values)
    base = float(np.asarray(values.base_values).ravel()[row]) if hasattr(values, "base_values") else 0.0

    contrib = (
        pd.DataFrame(
            {
                "feature": transformed.columns,
                "value": transformed.iloc[row].to_numpy(),
                "shap": arr[row],
            }
        )
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure()
    if hasattr(values, "base_values"):
        shap.plots.waterfall(values[row], max_display=18, show=False)
    else:
        shap.plots.bar(shap.Explanation(arr[row], base_values=base, data=transformed.iloc[row]), show=False)
    plt.title(f"SHAP local explanation -- observation {row}")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_local_row{row}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return base, contrib


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP analysis of the final model.")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--rows", type=int, nargs="*", default=[0])
    args = parser.parse_args()

    model = load_model()
    _, _, X_hold, y_hold = load_dev_holdout()
    print(f"explaining on {min(args.sample, len(X_hold))} holdout rows")

    values, transformed, estimator = compute_shap(model, X_hold, sample=args.sample)
    print(f"estimator: {type(estimator).__name__}   features: {transformed.shape[1]}")

    ranking = global_report(values, transformed)
    print("\n=== SHAP global ranking (top 20) ===")
    print(ranking.head(20).round(3).to_string(index=False))

    for row in args.rows:
        base, contrib = local_report(values, transformed, row=row)
        pred = base + contrib["shap"].sum()
        print(f"\n=== SHAP local explanation, row {row} ===")
        print(f"base value (mean prediction) = {base:.2f}")
        print(f"model prediction             = {pred:.2f}")
        up = contrib[contrib["shap"] > 0].head(8)
        down = contrib[contrib["shap"] < 0].head(8)
        print("\nfeatures pushing the prediction UP:")
        print(up[["feature", "value", "shap"]].round(3).to_string(index=False))
        print("\nfeatures pushing the prediction DOWN:")
        print(down[["feature", "value", "shap"]].round(3).to_string(index=False))

    print(f"\nfigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
