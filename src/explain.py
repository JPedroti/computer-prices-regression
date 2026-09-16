"""SHAP explainability for the final model.

Produces the global and local analyses required by CLAUDE.md section 21.

SHAP attributes a model's prediction to its inputs; it describes what *the
model* does, not what causes prices in the world. No causal reading is implied
anywhere in this module or in the figures it writes.

Two presentation decisions, both of which follow from SHAP values being additive:

* **Columns are summed back onto their source feature.** The saturated
  representation spreads one conceptual feature such as ``brand`` over a dummy
  per level, sometimes the original numeric term and sometimes a missingness
  indicator. Read column by column the attribution is fragmented; summed per
  feature it is readable, and the sum is exact.
* **A weighted ensemble is explained member by member.** Its output is a linear
  combination of its members, so the SHAP values of the ensemble are the same
  linear combination of the members' SHAP values. Each member is explained with
  the explainer that is exact for it -- linear for the Ridge, tree for CatBoost
  -- and the results are combined with the ensemble's own weights.

    python -m src.explain
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
_LINEAR = {"Ridge", "LinearRegression", "Lasso", "ElasticNet"}


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
    frame = pd.DataFrame(np.asarray(values), columns=[str(c) for c in columns])
    groups: dict[str, list[str]] = {}
    for col in frame.columns:
        groups.setdefault(source_feature(col), []).append(col)
    return pd.DataFrame({feat: frame[cols].sum(axis=1) for feat, cols in groups.items()})


@dataclass
class ShapResult:
    """SHAP values expressed per engineered feature, plus what is needed to plot."""

    by_feature: pd.DataFrame        # one column per engineered feature
    base_value: float
    display_rows: pd.DataFrame      # engineered feature values, for readable labels
    estimator_name: str
    per_column: tuple | None = None  # (array, frame) for single models, for the beeswarm

    def prediction(self, row: int) -> float:
        return float(self.base_value + self.by_feature.iloc[row].sum())


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
    """Values at the level SHAP feature names refer to, i.e. after engineering."""
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


def _explain_single(model, raw_rows: pd.DataFrame) -> ShapResult:
    import shap

    prep, estimator = _split_pipeline(model)
    transformed = prep.transform(raw_rows)
    if not isinstance(transformed, pd.DataFrame):
        transformed = pd.DataFrame(
            np.asarray(transformed), columns=_feature_names(prep, np.shape(transformed)[1])
        )
    transformed = transformed.reset_index(drop=True)

    name = type(estimator).__name__
    if name in _LINEAR:
        explainer = shap.LinearExplainer(estimator, transformed)
    elif name == "CatBoostNative":
        # The wrapper holds the real booster and the encoding it expects.
        transformed = estimator._prepare(transformed)
        explainer = shap.TreeExplainer(estimator.model_)
    else:
        explainer = shap.TreeExplainer(estimator)

    values = explainer(transformed)
    arr = values.values if hasattr(values, "values") else np.asarray(values)
    base = float(np.asarray(values.base_values).ravel()[0]) if hasattr(values, "base_values") else 0.0

    return ShapResult(
        by_feature=aggregate_by_feature(arr, transformed.columns),
        base_value=base,
        display_rows=_engineered_frame(prep, raw_rows).reset_index(drop=True),
        estimator_name=name,
        per_column=(arr, transformed),
    )


def compute_shap(
    model, X: pd.DataFrame, sample: int = DEFAULT_SAMPLE, seed: int = RANDOM_SEED
) -> ShapResult:
    """SHAP values for any model this project builds, expressed per feature."""
    from .ensemble import WeightedEnsemble

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    raw_rows = X.iloc[idx].reset_index(drop=True)

    if not isinstance(model, WeightedEnsemble):
        return _explain_single(model, raw_rows)

    # Ensemble: explain each member, then combine with the ensemble's weights.
    parts, bases, names = [], [], []
    for (label, _), fitted, weight in zip(model.members, model.fitted_, model.weights_):
        part = _explain_single(fitted, raw_rows)
        parts.append(part.by_feature * float(weight))
        bases.append(part.base_value * float(weight))
        names.append(f"{label}({float(weight):.3f})")

    features = sorted(set().union(*[set(p.columns) for p in parts]))
    combined = parts[0].reindex(columns=features, fill_value=0.0)
    for part in parts[1:]:
        combined = combined.add(part.reindex(columns=features, fill_value=0.0), fill_value=0.0)

    display = _engineered_frame(_split_pipeline(model.fitted_[0])[0], raw_rows).reset_index(drop=True)
    return ShapResult(
        by_feature=combined,
        base_value=float(sum(bases)),
        display_rows=display,
        estimator_name="WeightedEnsemble[" + ", ".join(names) + "]",
        per_column=None,
    )


# --------------------------------------------------------------------------- #
# Global analysis
# --------------------------------------------------------------------------- #
def global_report(result: ShapResult, top: int = 20, out_dir: Path = FIGURES_DIR):
    """Global feature ranking plus summary plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ranking = (
        pd.DataFrame(
            {
                "feature": result.by_feature.columns,
                "mean_abs_shap": result.by_feature.abs().mean(axis=0).to_numpy(),
                "mean_shap": result.by_feature.mean(axis=0).to_numpy(),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    head = ranking.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(head))))
    ax.barh(head["feature"], head["mean_abs_shap"], color="steelblue")
    ax.set_xlabel("mean |SHAP| (in price units)")
    ax.set_title("SHAP global importance, summed per feature")
    plt.tight_layout()
    fig.savefig(out_dir / "shap_global_by_feature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if result.per_column is not None:
        import shap

        arr, transformed = result.per_column
        plt.figure()
        shap.summary_plot(arr, transformed, max_display=top, show=False, plot_size=(10, 8))
        plt.title("SHAP beeswarm over transformed columns")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_global_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()

        pd.DataFrame(
            {
                "column": [str(c) for c in transformed.columns],
                "mean_abs_shap": np.abs(arr).mean(axis=0),
                "mean_shap": arr.mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False).to_csv(
            REPORTS_DIR / "shap_global_ranking_by_column.csv", index=False
        )

    ranking.to_csv(REPORTS_DIR / "shap_global_ranking.csv", index=False)
    return ranking


# --------------------------------------------------------------------------- #
# Local analysis
# --------------------------------------------------------------------------- #
def local_report(result: ShapResult, row: int = 0, out_dir: Path = FIGURES_DIR):
    """Explain one observation: contributions per feature, with readable values."""
    out_dir.mkdir(parents=True, exist_ok=True)

    values = result.by_feature.iloc[row]
    raw = result.display_rows.iloc[row]
    contrib = (
        pd.DataFrame(
            {
                "feature": values.index,
                "value": [raw.get(f, "-") for f in values.index],
                "shap": values.to_numpy(),
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
        f"Observation {row}: base {result.base_value:,.0f} -> prediction {result.prediction(row):,.0f}"
    )
    plt.tight_layout()
    fig.savefig(out_dir / f"shap_local_row{row}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return result.base_value, contrib


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP analysis of the final model.")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--rows", type=int, nargs="*", default=[0])
    args = parser.parse_args()

    model = load_model()
    _, _, X_hold, _ = load_dev_holdout()
    print(f"explaining on {min(args.sample, len(X_hold))} holdout rows")

    result = compute_shap(model, X_hold, sample=args.sample)
    print(f"estimator: {result.estimator_name}   features: {result.by_feature.shape[1]}")

    ranking = global_report(result)
    print("\n=== SHAP global ranking, summed per feature (top 20) ===")
    print(ranking.head(20).round(2).to_string(index=False))

    for row in args.rows:
        base, contrib = local_report(result, row=row)
        print(f"\n=== SHAP local explanation, row {row} ===")
        print(f"base value (mean prediction) = {base:.2f}")
        print(f"model prediction             = {result.prediction(row):.2f}")
        print("\nfeatures pushing the prediction UP:")
        print(contrib[contrib["shap"] > 0].head(8)[["feature", "value", "shap"]].round(2).to_string(index=False))
        print("\nfeatures pushing the prediction DOWN:")
        print(contrib[contrib["shap"] < 0].head(8)[["feature", "value", "shap"]].round(2).to_string(index=False))

    print(f"\nfigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
