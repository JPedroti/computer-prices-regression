"""SHAP explainability for the final model.

Produces the global and local analyses required by CLAUDE.md section 21.

SHAP attributes a model's prediction to its inputs; it describes what *the
model* does, not what causes prices in the world. No causal reading is implied
anywhere in this module or in the figures it writes.

A note on presentation. The final model uses a saturated representation, so one
conceptual feature such as ``brand`` is spread over many columns: a dummy per
level, sometimes the original numeric term, sometimes a missingness indicator.
Read column by column, the attribution is fragmented and hard to interpret.
SHAP values are additive, so this module also reports the values summed back
onto the source feature, which is the view a reader actually wants. Both views
are produced; the aggregated one is the headline.

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

from .config import FIGURES_DIR, RANDOM_SEED, REPORTS_DIR
from .data import load_dev_holdout
from .inference import load_model

DEFAULT_SAMPLE = 2000


# --------------------------------------------------------------------------- #
# Mapping transformed columns back to features
# --------------------------------------------------------------------------- #
def source_feature(name: str) -> str:
    """Map a transformed column back to the engineered feature it came from."""
    if "__lvl_" in name:
        return name.split("__lvl_")[0]
    for prefix in ("missingindicator_", "cat__", "num__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def aggregate_by_feature(values: np.ndarray, columns) -> pd.DataFrame:
    """Sum the SHAP values of every transformed column onto its source feature."""
    frame = pd.DataFrame(np.asarray(values), columns=list(columns))
    groups: dict[str, list[str]] = {}
    for col in frame.columns:
        groups.setdefault(source_feature(col), []).append(col)
    return pd.DataFrame({feat: frame[cols].sum(axis=1) for feat, cols in groups.items()})


# --------------------------------------------------------------------------- #
# Computing SHAP values
# --------------------------------------------------------------------------- #
def _split_pipeline(model):
    """Return ``(preprocessing, estimator)``, unwrapping any target transform."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline

    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    if isinstance(model, Pipeline):
        return model[:-1], model[-1]
    raise TypeError(f"cannot decompose model of type {type(model).__name__}")


def _engineered_frame(prep, raw_rows: pd.DataFrame) -> pd.DataFrame:
    """Values at the level SHAP names refer to, i.e. after feature engineering.

    ``source_feature`` maps transformed columns onto *engineered* names such as
    ``res_pixels``, which do not exist in the raw input, so the display frame has
    to be taken after the feature step and before the column encoding.
    """
    from .features import FeatureEngineer

    def find(obj):
        for _, step in getattr(obj, "steps", []):
            if isinstance(step, FeatureEngineer):
                return step
            found = find(step)
            if found is not None:
                return found
        return None

    engineer = find(prep)
    return engineer.transform(raw_rows) if engineer is not None else raw_rows


def _feature_names(prep, n_columns: int) -> list[str]:
    try:
        names = list(prep.get_feature_names_out())
        if len(names) == n_columns:
            return [str(n) for n in names]
    except Exception:
        pass
    return [f"f{i}" for i in range(n_columns)]


def compute_shap(model, X: pd.DataFrame, sample: int = DEFAULT_SAMPLE, seed: int = RANDOM_SEED):
    """Compute SHAP values on a random sample of rows.

    Returns ``(values, transformed, raw_rows, estimator)``. The explainer matches
    the estimator: exact for linear models, exact for tree ensembles.
    """
    import shap

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    raw_rows = X.iloc[idx].reset_index(drop=True)

    prep, estimator = _split_pipeline(model)
    transformed = prep.transform(raw_rows)
    if not isinstance(transformed, pd.DataFrame):
        transformed = pd.DataFrame(
            np.asarray(transformed), columns=_feature_names(prep, np.shape(transformed)[1])
        )
    transformed = transformed.reset_index(drop=True)
    display_rows = _engineered_frame(prep, raw_rows).reset_index(drop=True)

    name = type(estimator).__name__
    if name in {"Ridge", "LinearRegression", "Lasso", "ElasticNet"}:
        explainer = shap.LinearExplainer(estimator, transformed)
    else:
        explainer = shap.TreeExplainer(estimator)

    values = explainer(transformed)
    return values, transformed, display_rows, estimator


# --------------------------------------------------------------------------- #
# Global analysis
# --------------------------------------------------------------------------- #
def global_report(values, transformed: pd.DataFrame, top: int = 20, out_dir: Path = FIGURES_DIR):
    """Global feature ranking (aggregated and per-column) plus summary plots."""
    import shap

    out_dir.mkdir(parents=True, exist_ok=True)
    arr = values.values if hasattr(values, "values") else np.asarray(values)

    by_feature = aggregate_by_feature(arr, transformed.columns)
    ranking = (
        pd.DataFrame(
            {
                "feature": by_feature.columns,
                "mean_abs_shap": by_feature.abs().mean(axis=0).to_numpy(),
                "mean_shap": by_feature.mean(axis=0).to_numpy(),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Headline plot: aggregated importance, one bar per real feature.
    head = ranking.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(head))))
    ax.barh(head["feature"], head["mean_abs_shap"], color="steelblue")
    ax.set_xlabel("mean |SHAP| (currency units of price)")
    ax.set_title("SHAP global importance, summed per feature")
    plt.tight_layout()
    fig.savefig(out_dir / "shap_global_by_feature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-column beeswarm: shows the direction of each individual level's effect.
    plt.figure()
    shap.summary_plot(arr, transformed, max_display=top, show=False, plot_size=(10, 8))
    plt.title("SHAP beeswarm over transformed columns")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    column_ranking = (
        pd.DataFrame(
            {
                "column": transformed.columns,
                "mean_abs_shap": np.abs(arr).mean(axis=0),
                "mean_shap": arr.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(REPORTS_DIR / "shap_global_ranking.csv", index=False)
    column_ranking.to_csv(REPORTS_DIR / "shap_global_ranking_by_column.csv", index=False)
    return ranking, column_ranking


# --------------------------------------------------------------------------- #
# Local analysis
# --------------------------------------------------------------------------- #
def local_report(
    values,
    transformed: pd.DataFrame,
    raw_rows: pd.DataFrame,
    row: int = 0,
    out_dir: Path = FIGURES_DIR,
):
    """Explain one observation: contributions summed per feature, with raw values."""
    import shap

    out_dir.mkdir(parents=True, exist_ok=True)
    arr = values.values if hasattr(values, "values") else np.asarray(values)
    base = float(np.asarray(values.base_values).ravel()[row]) if hasattr(values, "base_values") else 0.0

    by_feature = aggregate_by_feature(arr[[row]], transformed.columns).iloc[0]
    raw = raw_rows.iloc[row]
    contrib = (
        pd.DataFrame(
            {
                "feature": by_feature.index,
                # The value the reader recognises, not the scaled one.
                "value": [raw.get(f, "-") for f in by_feature.index],
                "shap": by_feature.to_numpy(),
            }
        )
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    head = contrib.head(16).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.4 * len(head))))
    colors = ["indianred" if v < 0 else "steelblue" for v in head["shap"]]
    ax.barh([f"{f} = {v}" for f, v in zip(head["feature"], head["value"])], head["shap"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("SHAP contribution to the predicted price")
    ax.set_title(
        f"Observation {row}: base {base:,.0f} -> prediction {base + contrib['shap'].sum():,.0f}"
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"shap_local_row{row}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return base, contrib


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP analysis of the final model.")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--rows", type=int, nargs="*", default=[0])
    args = parser.parse_args()

    model = load_model()
    _, _, X_hold, y_hold = load_dev_holdout()
    print(f"explaining on {min(args.sample, len(X_hold))} holdout rows")

    values, transformed, raw_rows, estimator = compute_shap(model, X_hold, sample=args.sample)
    print(f"estimator: {type(estimator).__name__}   transformed columns: {transformed.shape[1]}")

    ranking, column_ranking = global_report(values, transformed)
    print("\n=== SHAP global ranking, summed per feature (top 20) ===")
    print(ranking.head(20).round(2).to_string(index=False))

    for row in args.rows:
        base, contrib = local_report(values, transformed, raw_rows, row=row)
        pred = base + contrib["shap"].sum()
        print(f"\n=== SHAP local explanation, row {row} ===")
        print(f"base value (mean prediction) = {base:.2f}")
        print(f"model prediction             = {pred:.2f}")
        print("\nfeatures pushing the prediction UP:")
        print(contrib[contrib["shap"] > 0].head(8)[["feature", "value", "shap"]].round(2).to_string(index=False))
        print("\nfeatures pushing the prediction DOWN:")
        print(contrib[contrib["shap"] < 0].head(8)[["feature", "value", "shap"]].round(2).to_string(index=False))

    print(f"\nfigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
