"""Tests for the critical components: feature engineering, preprocessing,
model assembly and the inference contract (CLAUDE.md section 25).

Run with:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_raw, make_dev_holdout, split_features_target
from src.evaluate import bootstrap_rmse_ci, overfitting_report, rmse
from src.features import FeatureConfig, FeatureEngineer
from src.models import build_model
from src.preprocessing import build_preprocessor

SAMPLE_ROWS = 2000


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw().head(SAMPLE_ROWS)


@pytest.fixture(scope="module")
def xy(raw: pd.DataFrame):
    return split_features_target(raw)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_feature_engineer_drops_identifier_columns(xy):
    X, _ = xy
    out = FeatureEngineer(FeatureConfig()).fit_transform(X)
    for col in ("ID", "model", "cpu_model"):
        assert col not in out.columns, f"{col} must not reach the model"


def test_feature_engineer_preserves_row_count_and_order(xy):
    X, _ = xy
    out = FeatureEngineer(FeatureConfig()).fit_transform(X)
    assert len(out) == len(X)
    assert list(out.index) == list(X.index)


def test_model_and_cpu_decomposition(xy):
    X, _ = xy
    out = FeatureEngineer(FeatureConfig()).fit_transform(X)
    assert "model_line" in out.columns
    assert "cpu_family" in out.columns
    # "Acer Creator HGG" -> line "Creator"; "Intel i5-12462" -> family "Intel i5"
    first = X.iloc[0]
    assert out.iloc[0]["model_line"] == first["model"].split()[1]
    assert not any(ch.isdigit() for ch in str(out.iloc[0]["cpu_family"]).split()[-1])


def test_structural_zeros_become_missing(xy):
    X, _ = xy
    out = FeatureEngineer(FeatureConfig()).fit_transform(X)
    desktops = out["device_type"] == "Desktop"
    assert out.loc[desktops, "battery_wh"].isna().all()
    assert out.loc[~desktops, "psu_watts"].isna().all()
    # The applicability flags keep the information that was in the zeros.
    assert (out.loc[desktops, "has_battery"] == 0).all()
    assert (out.loc[~desktops, "has_psu"] == 0).all()


def test_resolution_decomposition(xy):
    X, _ = xy
    out = FeatureEngineer(FeatureConfig()).fit_transform(X)
    w, h = X.iloc[0]["resolution"].split("x")
    assert out.iloc[0]["res_width"] == float(w)
    assert out.iloc[0]["res_height"] == float(h)
    assert out["res_pixels"].gt(0).all()


def test_feature_families_are_switchable(xy):
    X, _ = xy
    base = FeatureEngineer(FeatureConfig()).fit_transform(X)
    rich = FeatureEngineer(FeatureConfig(capacity=True, premium=True)).fit_transform(X)
    assert rich.shape[1] > base.shape[1]
    assert set(base.columns).issubset(rich.columns)


def test_transform_schema_is_stable_for_unseen_rows(xy):
    X, _ = xy
    fe = FeatureEngineer(FeatureConfig(capacity=True, premium=True)).fit(X.head(500))
    out = fe.transform(X.tail(500))
    assert list(out.columns) == fe.feature_names_
    assert len(out) == 500


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["native", "onehot", "ordinal"])
def test_preprocessor_handles_unseen_categories(xy, kind):
    X, _ = xy
    prep = build_preprocessor(kind).fit(X.head(1000))
    unseen = X.tail(500).copy()
    unseen.loc[unseen.index[0], "brand"] = "BrandThatNeverExisted"
    out = prep.transform(unseen)
    assert out.shape[0] == 500


def test_onehot_output_has_no_missing_values(xy):
    X, _ = xy
    prep = build_preprocessor("onehot").fit(X)
    out = np.asarray(prep.transform(X), dtype=float)
    assert np.isfinite(out).all(), "linear models cannot consume NaN"


# --------------------------------------------------------------------------- #
# Model assembly and the inference contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["ridge", "lgbm", "hgb"])
def test_build_model_fits_and_predicts(xy, name):
    X, y = xy
    model = build_model(name)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()


def test_pipeline_accepts_raw_columns_without_manual_preprocessing(xy):
    """New data must flow in exactly as delivered, target column optional."""
    X, y = xy
    model = build_model("ridge")
    model.fit(X, y)
    fresh = X.head(50).copy()
    preds = model.predict(fresh)
    assert len(preds) == 50


def test_predictions_preserve_order(xy):
    X, y = xy
    model = build_model("lgbm", n_estimators=50)
    model.fit(X, y)
    subset = X.head(200)
    full = model.predict(subset)
    shuffled_idx = np.random.default_rng(0).permutation(200)
    shuffled = model.predict(subset.iloc[shuffled_idx])
    np.testing.assert_allclose(full[shuffled_idx], shuffled, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# Evaluation and the official overfitting rule
# --------------------------------------------------------------------------- #
def test_rmse_matches_manual_computation():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    assert rmse(y, p) == pytest.approx(np.sqrt(4 / 3))


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.normal(1000, 200, 3000)
    p = y + rng.normal(0, 50, 3000)
    ci = bootstrap_rmse_ci(y, p, n_boot=300, seed=1)
    assert ci["ci_low"] < ci["rmse"] < ci["ci_high"]


def test_overfitting_rule_detects_a_clearly_overfit_model():
    rng = np.random.default_rng(0)
    y_tr = rng.normal(1000, 200, 4000)
    y_va = rng.normal(1000, 200, 2000)
    tight = y_tr + rng.normal(0, 5, 4000)      # near-perfect on train
    loose = y_va + rng.normal(0, 200, 2000)    # poor on validation
    report = overfitting_report(y_tr, tight, y_va, loose, n_boot=300)
    assert report.intervals_overlap is False
    assert report.points == 0


def test_overfitting_rule_accepts_a_balanced_model():
    rng = np.random.default_rng(0)
    y_tr = rng.normal(1000, 200, 4000)
    y_va = rng.normal(1000, 200, 2000)
    report = overfitting_report(
        y_tr, y_tr + rng.normal(0, 100, 4000),
        y_va, y_va + rng.normal(0, 100, 2000),
        n_boot=300,
    )
    assert report.intervals_overlap is True
    assert report.points == 15


# --------------------------------------------------------------------------- #
# Split protocol
# --------------------------------------------------------------------------- #
def test_dev_and_holdout_are_disjoint_and_complete():
    df = load_raw()
    dev, holdout = make_dev_holdout(df)
    assert len(dev) + len(holdout) == len(df)
    assert set(dev["ID"]).isdisjoint(set(holdout["ID"]))


def test_split_is_deterministic():
    df = load_raw()
    a, _ = make_dev_holdout(df)
    b, _ = make_dev_holdout(df)
    assert a["ID"].tolist() == b["ID"].tolist()
