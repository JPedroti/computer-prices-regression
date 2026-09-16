"""End-to-end inference on new data.

The saved artefact is a complete pipeline: raw columns in, prices out. No
manual preprocessing, no feature building by hand and no retraining is needed
(CLAUDE.md section 20).

Command line
------------
    python -m src.inference --input new_data.csv --output predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import FINAL_MODEL_PATH, ID_COL, TARGET

PREDICTION_COL = "predicted_price"


def load_model(path: Path | str = FINAL_MODEL_PATH):
    """Load the serialized final pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"model artefact not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict_frame(df: pd.DataFrame, model=None, model_path: Path | str = FINAL_MODEL_PATH) -> pd.Series:
    """Predict prices for a raw feature frame, preserving row order and count.

    The target column is tolerated and ignored if present, so the same function
    works on labelled development data and on unlabelled new data.
    """
    if model is None:
        model = load_model(model_path)

    features = df.drop(columns=[c for c in (TARGET,) if c in df.columns])
    preds = np.asarray(model.predict(features), dtype=float)

    if len(preds) != len(df):
        raise RuntimeError(
            f"pipeline returned {len(preds)} predictions for {len(df)} input rows"
        )
    return pd.Series(preds, index=df.index, name=PREDICTION_COL)


def predict_csv(
    input_path: Path | str,
    output_path: Path | str | None = None,
    model_path: Path | str = FINAL_MODEL_PATH,
) -> pd.DataFrame:
    """Read a CSV, predict, and optionally write the predictions out."""
    df = pd.read_csv(input_path)
    preds = predict_frame(df, model_path=model_path)

    out = pd.DataFrame(index=df.index)
    if ID_COL in df.columns:
        out[ID_COL] = df[ID_COL]
    out[PREDICTION_COL] = preds.to_numpy()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict computer prices from a CSV file.")
    parser.add_argument("--input", required=True, help="CSV with the feature columns")
    parser.add_argument("--output", default=None, help="where to write predictions")
    parser.add_argument("--model", default=str(FINAL_MODEL_PATH), help="model artefact path")
    args = parser.parse_args()

    out = predict_csv(args.input, args.output, args.model)
    print(out.head(10).to_string(index=False))
    print(f"\n{len(out)} predictions"
          + (f" written to {args.output}" if args.output else " (not written; pass --output)"))


if __name__ == "__main__":
    main()
