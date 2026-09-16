"""Tests for the delivered inference contract (CLAUDE.md sections 20 and 25).

The promise being tested: a CSV with the feature columns goes in, one prediction
per row comes out, in the same order, with no manual preprocessing and no
retraining.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FINAL_MODEL_PATH, TARGET
from src.data import load_raw, make_dev_holdout
from src.inference import PREDICTION_COL, load_model, predict_csv, predict_frame
from src.models import build_model

SAMPLE_ROWS = 3000


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A small but complete pipeline, serialised exactly like the deliverable."""
    df = load_raw().head(SAMPLE_ROWS)
    model = build_model("ridge", preprocessor="levels_plus", alpha=100.0)
    model.fit(df.drop(columns=[TARGET]), df[TARGET])
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump(model, path)
    return path, df


@pytest.fixture(scope="module")
def new_data() -> pd.DataFrame:
    """Rows the model has never seen, with the target removed as in real use."""
    _, holdout = make_dev_holdout(load_raw())
    return holdout.head(500).drop(columns=[TARGET]).reset_index(drop=True)


def test_predicts_from_a_csv_without_the_target(trained, new_data, tmp_path):
    path, _ = trained
    csv = tmp_path / "new_machines.csv"
    new_data.to_csv(csv, index=False)

    out = predict_csv(csv, model_path=path)

    assert len(out) == len(new_data)
    assert PREDICTION_COL in out.columns
    assert out[PREDICTION_COL].notna().all()
    assert np.isfinite(out[PREDICTION_COL]).all()


def test_ids_are_carried_through_unchanged(trained, new_data, tmp_path):
    path, _ = trained
    csv = tmp_path / "new_machines.csv"
    new_data.to_csv(csv, index=False)

    out = predict_csv(csv, model_path=path)

    assert out["ID"].tolist() == new_data["ID"].tolist()


def test_target_column_is_tolerated_and_ignored(trained, tmp_path):
    """The same file works whether or not it still carries the true price."""
    path, _ = trained
    _, holdout = make_dev_holdout(load_raw())
    labelled = holdout.head(200).reset_index(drop=True)

    with_target = tmp_path / "with_target.csv"
    without_target = tmp_path / "without_target.csv"
    labelled.to_csv(with_target, index=False)
    labelled.drop(columns=[TARGET]).to_csv(without_target, index=False)

    a = predict_csv(with_target, model_path=path)[PREDICTION_COL].to_numpy()
    b = predict_csv(without_target, model_path=path)[PREDICTION_COL].to_numpy()

    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def test_row_order_is_preserved(trained, new_data):
    path, _ = trained
    model = load_model(path)

    straight = predict_frame(new_data, model=model)
    shuffled_frame = new_data.sample(frac=1.0, random_state=7)
    shuffled = predict_frame(shuffled_frame, model=model)

    np.testing.assert_allclose(
        straight.loc[shuffled_frame.index].to_numpy(), shuffled.to_numpy(), rtol=1e-10
    )


def test_single_row_works(trained, new_data):
    path, _ = trained
    preds = predict_frame(new_data.head(1), model=load_model(path))
    assert len(preds) == 1
    assert np.isfinite(preds.iloc[0])


def test_unseen_categories_do_not_break_inference(trained, new_data):
    """New data may contain brands or GPUs absent from training."""
    path, _ = trained
    odd = new_data.head(5).copy()
    odd.loc[odd.index[0], "brand"] = "BrandThatDidNotExistAtTrainingTime"
    odd.loc[odd.index[1], "gpu_model"] = "RTX 99 999"
    odd.loc[odd.index[2], "os"] = "TempleOS"

    preds = predict_frame(odd, model=load_model(path))

    assert len(preds) == 5
    assert np.isfinite(preds).all()


def test_no_retraining_is_needed(trained, new_data):
    """Loading and predicting must not refit: two loads give identical output."""
    path, _ = trained
    a = predict_frame(new_data, model=load_model(path)).to_numpy()
    b = predict_frame(new_data, model=load_model(path)).to_numpy()
    np.testing.assert_array_equal(a, b)


def test_missing_model_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="model artefact not found"):
        load_model(tmp_path / "does_not_exist.joblib")


@pytest.mark.skipif(not FINAL_MODEL_PATH.exists(), reason="final model not trained yet")
def test_delivered_artefact_predicts_the_holdout(new_data):
    """The artefact actually shipped in models/ satisfies the same contract."""
    preds = predict_frame(new_data)
    assert len(preds) == len(new_data)
    assert np.isfinite(preds).all()
    assert preds.between(100, 20000).all(), "predictions should be plausible prices"
