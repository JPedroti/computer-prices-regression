"""Preprocessing strategies.

Three representations are provided because different model families need
different inputs:

``native``   categorical columns as pandas ``category`` dtype -- LightGBM,
             CatBoost and XGBoost consume these directly.
``onehot``   median imputation + one-hot + scaling -- required by linear models.
``ordinal``  integer-coded categories + imputation -- for tree models without
             native categorical support (RandomForest, ExtraTrees).

Every learned statistic (medians, category vocabularies, scaling parameters)
is fitted inside a scikit-learn pipeline, so it is re-fitted on each training
fold only and can never leak validation information (CLAUDE.md section 5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import FeatureConfig, FeatureEngineer

PREPROCESSORS = ("native", "onehot", "ordinal", "levels", "levels_plus")

# A column with at most this many distinct values is eligible to be expanded
# into one dummy per level by the saturated representations.
MAX_LEVELS = 60


class ToCategory(BaseEstimator, TransformerMixin):
    """Freeze categorical columns to a ``category`` dtype learned on train only.

    Categories seen at fit time define the vocabulary; unseen values at
    transform time become NaN, which the boosting libraries treat as missing
    rather than crashing.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ToCategory":
        self.columns_ = list(X.columns)
        self.categories_ = {
            c: pd.Index(sorted(X[c].dropna().astype(str).unique()))
            for c in X.columns
            if _is_object_like(X[c])
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()[self.columns_]
        for col, cats in self.categories_.items():
            values = out[col].astype(str).where(out[col].notna())
            # Map categories unseen during fit to NaN before constructing the
            # Categorical: the boosting libraries read that as missing, and
            # passing out-of-vocabulary values straight in is deprecated.
            values = values.where(values.isin(cats))
            out[col] = pd.Categorical(values, categories=cats)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.columns_, dtype=object)


def _is_object_like(series: pd.Series) -> bool:
    return series.dtype == object or str(series.dtype) in ("str", "string", "category")


def _column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in X.columns if _is_object_like(X[c])]
    num = [c for c in X.columns if c not in cat]
    return cat, num


class ColumnTyper(BaseEstimator, TransformerMixin):
    """Builds the requested representation, discovering column groups at fit time."""

    def __init__(self, kind: str = "native"):
        if kind not in PREPROCESSORS:
            raise ValueError(f"kind must be one of {PREPROCESSORS}, got {kind!r}")
        self.kind = kind

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnTyper":
        cat, num = _column_groups(X)
        self.cat_, self.num_ = cat, num
        self.inner_ = _build_inner(self.kind, cat, num)
        self.inner_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame):
        return self.inner_.transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self.kind == "native":
            return np.asarray(self.cat_ + self.num_, dtype=object)
        return np.asarray(self.inner_.get_feature_names_out(), dtype=object)


class LevelExpander(BaseEstimator, TransformerMixin):
    """Saturated additive representation: one dummy per observed level.

    Numeric columns with few distinct values (``ram_gb``, ``cpu_tier``, ...) are
    expanded into dummies rather than entering as a single linear term, so a
    linear model can fit an arbitrary shape per feature while staying additive.
    With ``keep_numeric`` the original numeric column is kept alongside its
    dummies, which lets the model extrapolate monotonically for levels that are
    thinly observed.

    The set of levels is learned at fit time -- on training folds only.
    """

    def __init__(self, max_levels: int = MAX_LEVELS, keep_numeric: bool = False):
        self.max_levels = max_levels
        self.keep_numeric = keep_numeric

    _SUFFIX = "__lvl"

    def fit(self, X: pd.DataFrame, y=None) -> "LevelExpander":
        # Routing is by type first, cardinality second. A text column can never
        # go down the numeric branch however many levels it has (interaction
        # columns, for instance, are text with hundreds of levels), and a
        # genuinely continuous numeric column is never expanded into dummies.
        self.level_cols_ = [
            c
            for c in X.columns
            if _is_object_like(X[c]) or X[c].nunique(dropna=True) <= self.max_levels
        ]
        self.wide_cols_ = [c for c in X.columns if c not in self.level_cols_]
        # Numeric columns that were expanded into dummies can additionally be
        # kept in their original numeric form, so the model retains a monotone
        # term for levels that are thinly observed.
        self.numeric_extra_ = (
            [c for c in self.level_cols_ if not _is_object_like(X[c])] if self.keep_numeric else []
        )

        dummy_cols = [c + self._SUFFIX for c in self.level_cols_]
        transformers = [
            ("levels", OneHotEncoder(handle_unknown="ignore", sparse_output=False), dummy_cols)
        ]
        passthrough = self.wide_cols_ + self.numeric_extra_
        if passthrough:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    passthrough,
                )
            )

        self.inner_ = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
        self.inner_.fit(self._expand(X), y)
        return self

    def _expand(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add a string-coded twin of every level column, leaving originals intact.

        Levels are compared as strings so that 8 and 8.0 collapse to one level;
        keeping the original column untouched lets the numeric branch of the
        ColumnTransformer still see real numbers.
        """
        out = X.copy()
        for col in self.level_cols_:
            out[col + self._SUFFIX] = (
                out[col].astype("string").fillna("__missing__").astype(str)
            )
        return out

    def transform(self, X: pd.DataFrame):
        return self.inner_.transform(self._expand(X))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.inner_.get_feature_names_out(), dtype=object)


def _build_inner(kind: str, cat: list[str], num: list[str]):
    if kind == "levels":
        return LevelExpander(keep_numeric=False)

    if kind == "levels_plus":
        return LevelExpander(keep_numeric=True)

    if kind == "native":
        # Keep a DataFrame; only pin the category vocabulary.
        return ToCategory()

    if kind == "onehot":
        return ColumnTransformer(
            [
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    cat,
                ),
                (
                    "num",
                    Pipeline(
                        [
                            # add_indicator keeps the "not applicable" signal of
                            # the structural zeros after imputation.
                            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    num,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                cat,
            ),
            (
                "num",
                SimpleImputer(strategy="median", add_indicator=True),
                num,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessor(kind: str = "native", feature_config: FeatureConfig | None = None) -> Pipeline:
    """Feature engineering followed by the chosen column representation."""
    return Pipeline(
        [
            ("features", FeatureEngineer(feature_config)),
            ("columns", ColumnTyper(kind)),
        ]
    )
