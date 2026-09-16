"""Reproducibility helpers and experiment tracking."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .config import RESULTS_CSV

# Schema required by CLAUDE.md section 23.
RESULT_COLUMNS: list[str] = [
    "experiment_id",
    "timestamp",
    "model",
    "features",
    "preprocessing",
    "hyperparameters",
    "seed",
    "validation_strategy",
    "train_rmse",
    "validation_rmse",
    "train_mae",
    "validation_mae",
    "validation_rmse_std",
    "fit_seconds",
    "notes",
]


def set_seed(seed: int) -> None:
    """Seed every RNG this project can reach."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def timer() -> Iterator[list[float]]:
    """Context manager yielding a one-element list that ends up holding elapsed seconds."""
    holder: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start


def _stringify(value: Any) -> str:
    """Render dicts/lists compactly and deterministically for CSV storage."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return str(value)


@dataclass
class ExperimentTracker:
    """Append-only experiment log backed by a CSV file.

    Results are flushed on every ``log`` call so an interrupted session never
    loses history.
    """

    path: Path = RESULTS_CSV
    rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _next_id(self) -> str:
        existing = 0
        if self.path.exists():
            existing = len(pd.read_csv(self.path))
        return f"exp_{existing + 1:04d}"

    def log(
        self,
        model: str,
        features: Any,
        preprocessing: str,
        hyperparameters: Any,
        seed: int,
        validation_strategy: str,
        train_rmse: float,
        validation_rmse: float,
        train_mae: float,
        validation_mae: float,
        validation_rmse_std: float | None = None,
        fit_seconds: float | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        row = {
            "experiment_id": self._next_id(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "features": _stringify(features),
            "preprocessing": preprocessing,
            "hyperparameters": _stringify(hyperparameters),
            "seed": seed,
            "validation_strategy": validation_strategy,
            "train_rmse": round(float(train_rmse), 4),
            "validation_rmse": round(float(validation_rmse), 4),
            "train_mae": round(float(train_mae), 4),
            "validation_mae": round(float(validation_mae), 4),
            "validation_rmse_std": (
                None if validation_rmse_std is None else round(float(validation_rmse_std), 4)
            ),
            "fit_seconds": None if fit_seconds is None else round(float(fit_seconds), 2),
            "notes": notes,
        }
        self.rows.append(row)
        frame = pd.DataFrame([row], columns=RESULT_COLUMNS)
        header = not self.path.exists()
        frame.to_csv(self.path, mode="a", header=header, index=False)
        return row

    @staticmethod
    def load(path: Path = RESULTS_CSV) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=RESULT_COLUMNS)
        return pd.read_csv(path)

    @staticmethod
    def leaderboard(path: Path = RESULTS_CSV, top: int = 15) -> pd.DataFrame:
        df = ExperimentTracker.load(path)
        if df.empty:
            return df
        cols = [
            "experiment_id", "model", "preprocessing", "validation_rmse",
            "validation_rmse_std", "train_rmse", "validation_mae", "notes",
        ]
        return df.sort_values("validation_rmse").head(top)[cols].reset_index(drop=True)
