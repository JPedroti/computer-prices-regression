"""Central configuration: paths, seeds and column groups.

Every constant that more than one module needs lives here so that experiments,
training and inference cannot silently drift apart.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

TRAIN_CSV: Path = DATA_DIR / "computer_prices_train_80.csv"
RESULTS_CSV: Path = EXPERIMENTS_DIR / "results.csv"
FINAL_MODEL_PATH: Path = MODELS_DIR / "final_model.joblib"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42
# Seeds used for multi-seed robustness checks.
ROBUSTNESS_SEEDS: tuple[int, ...] = (42, 7, 2024, 1337, 99)

# --------------------------------------------------------------------------- #
# Dataset semantics
# --------------------------------------------------------------------------- #
TARGET: str = "price"
ID_COL: str = "ID"

# Raw columns dropped before modelling.
#   ID       -> row identifier, Spearman vs price = 0.003 (audited noise)
#   model    -> decomposed into brand/line/code by the feature engineer
#   cpu_model-> decomposed into family/number by the feature engineer
RAW_DROP_COLS: tuple[str, ...] = (ID_COL, "model", "cpu_model")

# Columns whose zero means "not applicable" rather than a measured zero.
# Audited: battery_wh == 0 and charger_watts == 0 exactly for Desktops,
#          psu_watts == 0 exactly for Laptops.
STRUCTURAL_ZERO_COLS: tuple[str, ...] = ("battery_wh", "charger_watts", "psu_watts")

# Validation protocol
HOLDOUT_FRACTION: float = 0.20
CV_FOLDS: int = 5
CV_REPEATS: int = 2
BOOTSTRAP_N: int = 2000
BOOTSTRAP_SEED: int = 12345
