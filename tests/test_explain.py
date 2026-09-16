"""Tests for the SHAP layer (CLAUDE.md section 21).

The property that makes the presentation choices valid is additivity: the base
value plus the sum of the SHAP values must reproduce the model's prediction
exactly. That must survive both transformations this project applies — summing
transformed columns back onto their source feature, and combining a weighted
ensemble's members.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_raw, split_features_target
from src.ensemble import WeightedEnsemble
from src.explain import (
    aggregate_by_feature,
    compute_shap,
    global_report,
    local_report,
    source_feature,
)
from src.features import FeatureConfig
from src.models import build_model

TRAIN_ROWS = 4000
EXPLAIN_ROWS = 120


@pytest.fixture(scope="module")
def data():
    df = load_raw().head(TRAIN_ROWS + EXPLAIN_ROWS)
    X, y = split_features_target(df)
    return X.head(TRAIN_ROWS), y.head(TRAIN_ROWS), X.tail(EXPLAIN_ROWS).reset_index(drop=True)


@pytest.fixture(scope="module")
def ridge_model(data):
    X, y, _ = data
    return build_model("ridge", preprocessor="levels_plus", alpha=100.0).fit(X, y)


@pytest.fixture(scope="module")
def ensemble_model(data):
    X, y, _ = data
    cfg = FeatureConfig()
    ens = WeightedEnsemble(
        members=[
            ("ridge", build_model("ridge", feature_config=cfg, preprocessor="levels_plus", alpha=100.0)),
            ("cat", build_model("cat", feature_config=cfg, iterations=120, learning_rate=0.1, depth=4)),
        ],
        weights=[0.3717, 0.6283],
    )
    return ens.fit(X, y)


# --------------------------------------------------------------------------- #
# Column to feature mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "column,expected",
    [
        ("brand__lvl_Acer", "brand"),
        ("cpu_base_ghz__lvl_2.6", "cpu_base_ghz"),
        ("cpu_base_ghz", "cpu_base_ghz"),
        ("missingindicator_battery_wh", "battery_wh"),
        ("res_pixels", "res_pixels"),
    ],
)
def test_source_feature_mapping(column, expected):
    assert source_feature(column) == expected


def test_aggregation_preserves_the_total():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 4))
    columns = ["brand__lvl_A", "brand__lvl_B", "ram_gb", "missingindicator_ram_gb"]
    aggregated = aggregate_by_feature(values, columns)

    assert set(aggregated.columns) == {"brand", "ram_gb"}
    np.testing.assert_allclose(aggregated.sum(axis=1).to_numpy(), values.sum(axis=1), rtol=1e-12)


# --------------------------------------------------------------------------- #
# Additivity: the property everything else rests on
# --------------------------------------------------------------------------- #
def test_single_model_shap_reproduces_the_prediction(ridge_model, data):
    _, _, X_new = data
    result = compute_shap(ridge_model, X_new, sample=EXPLAIN_ROWS, seed=0)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_new), size=EXPLAIN_ROWS, replace=False)
    expected = ridge_model.predict(X_new.iloc[idx].reset_index(drop=True))
    reconstructed = result.base_value + result.by_feature.sum(axis=1).to_numpy()

    np.testing.assert_allclose(reconstructed, expected, rtol=1e-6, atol=1e-6)


def test_ensemble_shap_reproduces_the_prediction(ensemble_model, data):
    """A weighted ensemble is linear in its members, so its SHAP values are the
    same weighted combination of theirs."""
    _, _, X_new = data
    result = compute_shap(ensemble_model, X_new, sample=EXPLAIN_ROWS, seed=0)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_new), size=EXPLAIN_ROWS, replace=False)
    expected = ensemble_model.predict(X_new.iloc[idx].reset_index(drop=True))
    reconstructed = result.base_value + result.by_feature.sum(axis=1).to_numpy()

    np.testing.assert_allclose(reconstructed, expected, rtol=1e-6, atol=1e-6)


def test_ensemble_and_single_share_the_feature_space(ridge_model, ensemble_model, data):
    _, _, X_new = data
    single = compute_shap(ridge_model, X_new, sample=40, seed=0)
    ensemble = compute_shap(ensemble_model, X_new, sample=40, seed=0)
    assert set(single.by_feature.columns) == set(ensemble.by_feature.columns)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def test_global_report_ranks_every_feature(ridge_model, data, tmp_path):
    _, _, X_new = data
    result = compute_shap(ridge_model, X_new, sample=60, seed=0)
    ranking = global_report(result, out_dir=tmp_path)

    assert len(ranking) == result.by_feature.shape[1]
    assert ranking["mean_abs_shap"].is_monotonic_decreasing
    assert (ranking["mean_abs_shap"] >= 0).all()
    assert (tmp_path / "shap_global_by_feature.png").exists()


def test_local_report_shows_readable_values(ridge_model, data, tmp_path):
    """The local table must show the feature value a reader recognises, not the
    scaled number the estimator sees."""
    _, _, X_new = data
    result = compute_shap(ridge_model, X_new, sample=60, seed=0)
    base, contrib = local_report(result, row=0, out_dir=tmp_path)

    assert base == pytest.approx(result.base_value)
    assert contrib["shap"].sum() == pytest.approx(result.prediction(0) - base, rel=1e-6)
    assert (tmp_path / "shap_local_row0.png").exists()

    brand = contrib.loc[contrib["feature"] == "brand", "value"].iloc[0]
    assert isinstance(brand, str) and brand in set(load_raw()["brand"].unique())


def test_local_contributions_have_both_signs(ridge_model, data, tmp_path):
    _, _, X_new = data
    result = compute_shap(ridge_model, X_new, sample=60, seed=0)
    _, contrib = local_report(result, row=0, out_dir=tmp_path)
    assert (contrib["shap"] > 0).any(), "no feature pushed the prediction up"
    assert (contrib["shap"] < 0).any(), "no feature pushed the prediction down"
