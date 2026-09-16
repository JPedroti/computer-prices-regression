"""Feature engineering.

Everything here is a *row-wise* derivation: no statistic is learned from the
data, so the transformer cannot leak information between train and validation.
Families are individually switchable so each one can be measured on its own
(CLAUDE.md section 10).

Structures recovered during the dataset audit
---------------------------------------------
* ``model`` = "<brand> <line> <code>"; the 3-character code has 38,215 distinct
  values over 80,000 rows and a between-group variance ratio of 0.99 (pure
  noise), while ``line`` has a ratio of 93 and is kept.
* ``cpu_model`` = "<family> <number>"; family has a ratio of 2345, the trailing
  number has a ratio of 0.54 once conditioned on family (noise).
* ``battery_wh``/``charger_watts`` are 0 exactly for Desktops and ``psu_watts``
  is 0 exactly for Laptops: these are "not applicable", not measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import RAW_DROP_COLS, STRUCTURAL_ZERO_COLS

# Apple CPU families that mark the top of the price distribution.
_APPLE_HIGH_END = ("Pro", "Max")


@dataclass(frozen=True)
class FeatureConfig:
    """Switches for each feature family, so families can be ablated."""

    model_decomp: bool = True       # brand line extracted from `model`
    cpu_decomp: bool = True         # cpu family extracted from `cpu_model`
    cpu_generation: bool = False    # audited as noise; off by default, still testable
    structural_zeros: bool = True   # not-applicable zeros -> NaN + flags
    resolution: bool = True         # width / height / pixels / aspect / ppi
    capacity: bool = False          # ratios and capacity interactions
    premium: bool = False           # tail-segment indicators

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)

    def enabled(self) -> list[str]:
        return [k for k, v in self.as_dict().items() if v]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless row-wise feature builder usable inside a scikit-learn pipeline."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    # fit learns nothing; it only records the output schema for validation.
    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        out = self._build(X)
        self.feature_names_ = list(out.columns)
        self.categorical_features_ = [c for c in out.columns if _is_categorical(out[c])]
        self.numeric_features_ = [c for c in out.columns if c not in self.categorical_features_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._build(X)
        if hasattr(self, "feature_names_"):
            # Guarantee identical columns and order at inference time.
            for col in self.feature_names_:
                if col not in out.columns:
                    out[col] = np.nan
            out = out[self.feature_names_]
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(getattr(self, "feature_names_", []), dtype=object)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = X.copy()

        if cfg.model_decomp and "model" in df.columns:
            parts = df["model"].astype("string").str.split(" ", n=2, expand=True)
            df["model_line"] = parts[1]

        if cfg.cpu_decomp and "cpu_model" in df.columns:
            cpu = df["cpu_model"].astype("string")
            df["cpu_family"] = cpu.str.replace(r"[-\s]?\d{3,}$", "", regex=True).str.strip()
            if cfg.cpu_generation:
                num = cpu.str.extract(r"(\d{3,})$")[0]
                # A 5-digit code carries a 2-digit generation, otherwise 1 digit.
                gen = num.str[:2].where(num.str.len() >= 5, num.str[:1])
                df["cpu_generation"] = pd.to_numeric(gen, errors="coerce")

        if cfg.structural_zeros:
            if "battery_wh" in df.columns:
                df["has_battery"] = (df["battery_wh"] > 0).astype("int8")
            if "psu_watts" in df.columns:
                df["has_psu"] = (df["psu_watts"] > 0).astype("int8")
            if {"charger_watts", "psu_watts"} <= set(df.columns):
                # The two power sources are mutually exclusive; one combined
                # column expresses "power envelope" without a spurious zero.
                df["power_watts"] = df["charger_watts"] + df["psu_watts"]
            for col in STRUCTURAL_ZERO_COLS:
                if col in df.columns:
                    df[col] = df[col].astype("float64").replace(0.0, np.nan)

        if cfg.resolution and "resolution" in df.columns:
            res = df["resolution"].astype("string").str.split("x", n=1, expand=True)
            df["res_width"] = pd.to_numeric(res[0], errors="coerce")
            df["res_height"] = pd.to_numeric(res[1], errors="coerce")
            df["res_pixels"] = df["res_width"] * df["res_height"]
            df["res_aspect"] = df["res_width"] / df["res_height"]
            if "display_size_in" in df.columns:
                diag_px = np.sqrt(df["res_width"] ** 2 + df["res_height"] ** 2)
                df["res_ppi"] = diag_px / df["display_size_in"].replace(0, np.nan)

        if cfg.capacity:
            df = self._add_capacity(df)

        if cfg.premium:
            df = self._add_premium(df)

        return df.drop(columns=[c for c in RAW_DROP_COLS if c in df.columns])

    @staticmethod
    def _add_capacity(df: pd.DataFrame) -> pd.DataFrame:
        """Ratios and products expressing total machine capability."""
        cols = set(df.columns)
        if {"cpu_cores", "cpu_boost_ghz"} <= cols:
            df["cpu_total_ghz"] = df["cpu_cores"] * df["cpu_boost_ghz"]
        if {"cpu_threads", "cpu_cores"} <= cols:
            df["threads_per_core"] = df["cpu_threads"] / df["cpu_cores"].replace(0, np.nan)
        if {"cpu_boost_ghz", "cpu_base_ghz"} <= cols:
            df["ghz_headroom"] = df["cpu_boost_ghz"] - df["cpu_base_ghz"]
        if {"ram_gb", "cpu_cores"} <= cols:
            df["ram_per_core"] = df["ram_gb"] / df["cpu_cores"].replace(0, np.nan)
        if {"storage_gb", "storage_drive_count"} <= cols:
            df["storage_total_gb"] = df["storage_gb"] * df["storage_drive_count"]
        if {"vram_gb", "gpu_tier"} <= cols:
            df["vram_x_gpu_tier"] = df["vram_gb"] * df["gpu_tier"]
        if {"cpu_tier", "gpu_tier"} <= cols:
            df["tier_sum"] = df["cpu_tier"] + df["gpu_tier"]
            df["tier_product"] = df["cpu_tier"] * df["gpu_tier"]
        if {"res_pixels", "refresh_hz"} <= cols:
            df["pixel_rate"] = df["res_pixels"] * df["refresh_hz"]
        if "release_year" in cols:
            # Fixed reference year keeps the feature stable across datasets.
            df["age_years"] = 2025 - df["release_year"]
        if {"battery_wh", "weight_kg"} <= cols:
            df["wh_per_kg"] = df["battery_wh"] / df["weight_kg"].replace(0, np.nan)
        return df

    @staticmethod
    def _add_premium(df: pd.DataFrame) -> pd.DataFrame:
        """Indicators for the expensive tail, where most squared error lives.

        Derived only from feature columns -- the target is never consulted.
        """
        cols = set(df.columns)
        if "cpu_family" in cols:
            fam = df["cpu_family"].astype("string").fillna("")
            is_apple = fam.str.startswith("Apple")
            df["is_apple_silicon"] = is_apple.astype("int8")
            df["is_apple_high_end"] = (is_apple & fam.str.endswith(_APPLE_HIGH_END)).astype("int8")
        if "ram_gb" in cols:
            df["ram_log2"] = np.log2(df["ram_gb"].clip(lower=1))
            df["is_high_ram"] = (df["ram_gb"] >= 128).astype("int8")
        if "gpu_tier" in cols:
            df["is_top_gpu"] = (df["gpu_tier"] >= 6).astype("int8")
        if {"cpu_tier", "gpu_tier", "ram_gb"} <= cols:
            df["premium_score"] = (
                df["cpu_tier"] * 2.0 + df["gpu_tier"] * 2.0 + np.log2(df["ram_gb"].clip(lower=1))
            )
        if "storage_gb" in cols:
            df["storage_log2"] = np.log2(df["storage_gb"].clip(lower=1))
        return df


def _is_categorical(series: pd.Series) -> bool:
    return series.dtype == object or str(series.dtype) in ("str", "string", "category")


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ``(categorical_columns, numeric_columns)`` for a transformed frame."""
    cat = [c for c in df.columns if _is_categorical(df[c])]
    num = [c for c in df.columns if c not in cat]
    return cat, num
