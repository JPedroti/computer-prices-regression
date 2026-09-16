"""Dataset loading and the project's split protocol.

Split protocol (CLAUDE.md sections 12 and 17)
---------------------------------------------
The 80k development file is split once, with a fixed seed, into:

* ``dev``     (80%) -- every modelling decision is made here, via cross-validation;
* ``holdout`` (20%) -- touched only by the official overfitting rule at the end.

The holdout is never used to choose models, features or hyperparameters.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import HOLDOUT_FRACTION, RANDOM_SEED, TARGET, TRAIN_CSV


def load_raw(path=TRAIN_CSV) -> pd.DataFrame:
    """Load the development dataset exactly as delivered."""
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target column from the feature frame."""
    if TARGET not in df.columns:
        raise KeyError(f"target column {TARGET!r} not found; columns={list(df.columns)}")
    return df.drop(columns=[TARGET]), df[TARGET]


def make_dev_holdout(
    df: pd.DataFrame,
    holdout_fraction: float = HOLDOUT_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into the development frame and the sealed overfitting holdout.

    A plain random split is used because the secret test set is assumed to be
    drawn the same way as the development rows; this is treated as a working
    hypothesis, not a guarantee, so no split-specific tricks are applied.
    """
    dev, holdout = train_test_split(df, test_size=holdout_fraction, random_state=seed, shuffle=True)
    return dev.reset_index(drop=True), holdout.reset_index(drop=True)


def load_dev_holdout(
    path=TRAIN_CSV, holdout_fraction: float = HOLDOUT_FRACTION, seed: int = RANDOM_SEED
):
    """Convenience loader returning ``(X_dev, y_dev, X_holdout, y_holdout)``."""
    df = load_raw(path)
    dev, holdout = make_dev_holdout(df, holdout_fraction, seed)
    X_dev, y_dev = split_features_target(dev)
    X_hold, y_hold = split_features_target(holdout)
    return X_dev, y_dev, X_hold, y_hold
