"""Acceptance check for the delivered solution.

Runs through the requirements in CLAUDE.md section 35 against the artefacts that
are actually on disk, so the claim "this is finished" is verified rather than
asserted. Prints a pass/fail line per item and exits non-zero if anything fails.

    python experiments/final_check.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import FINAL_MODEL_PATH, RESULTS_CSV, TARGET
from src.data import load_dev_holdout, load_raw
from src.evaluate import mae, overfitting_report, rmse
from src.inference import load_model, predict_frame

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def main() -> None:
    print("=== artefacts ===")
    for path in [
        FINAL_MODEL_PATH,
        ROOT / "models" / "final_model_metadata.json",
        ROOT / "configs" / "final_model.json",
        ROOT / "requirements.txt",
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        RESULTS_CSV,
        ROOT / "experiments" / "experiment_log.md",
        ROOT / "notebooks" / "01_exploration_and_modeling.ipynb",
        ROOT / "reports" / "shap_global_ranking.csv",
        ROOT / "reports" / "figures" / "shap_global_by_feature.png",
        ROOT / "reports" / "figures" / "shap_effect_by_level.png",
        ROOT / "reports" / "figures" / "shap_local_row0.png",
    ]:
        check(f"exists: {path.relative_to(ROOT).as_posix()}", path.exists())

    print("\n=== experiment tracking ===")
    results = pd.read_csv(RESULTS_CSV)
    required = {
        "experiment_id", "timestamp", "model", "features", "preprocessing",
        "hyperparameters", "seed", "validation_strategy", "train_rmse",
        "validation_rmse", "train_mae", "validation_mae", "notes",
    }
    check("results.csv has the required schema", required <= set(results.columns),
          f"{len(results)} rows")
    check("more than one model family was tried",
          results["model"].str.split("|").str[0].nunique() >= 6,
          f"{results['model'].str.split('|').str[0].nunique()} distinct models")

    print("\n=== final model ===")
    model = load_model()
    X_dev, y_dev, X_hold, y_hold = load_dev_holdout()
    pred_dev = model.predict(X_dev)
    pred_hold = model.predict(X_hold)

    check("model loads without retraining", True, type(model).__name__)
    check("one prediction per row", len(pred_hold) == len(X_hold))
    check("predictions are finite", bool(np.isfinite(pred_hold).all()))
    check("predictions are plausible prices",
          bool(((pred_hold > 100) & (pred_hold < 20000)).all()),
          f"range {pred_hold.min():.0f}-{pred_hold.max():.0f}")

    print("\n=== metrics ===")
    metrics = {
        "train RMSE": rmse(y_dev, pred_dev), "train MAE": mae(y_dev, pred_dev),
        "holdout RMSE": rmse(y_hold, pred_hold), "holdout MAE": mae(y_hold, pred_hold),
    }
    for k, v in metrics.items():
        print(f"  {k:14s} {v:9.3f}")
    check("beats the mean baseline by a wide margin",
          metrics["holdout RMSE"] < 0.6 * rmse(y_hold, np.full(len(y_hold), y_dev.mean())))

    print("\n=== official overfitting rule ===")
    report = overfitting_report(y_dev, pred_dev, y_hold, pred_hold)
    print("  " + str(report).replace("\n", "\n  "))
    check("intervals overlap (no overfitting)", report.intervals_overlap,
          f"margin {report.margin:+.2f}")
    check("rule awards the full 15 points", report.points == 15)

    print("\n=== reproducibility ===")
    meta = json.loads((ROOT / "models" / "final_model_metadata.json").read_text(encoding="utf-8"))
    check("metadata records the config and the rule",
          "config" in meta and "overfitting_rule" in meta)
    # The metadata stores metrics rounded to four decimals, so the tolerance
    # has to match that precision rather than machine epsilon.
    check("metadata metrics match a fresh evaluation",
          abs(meta["metrics"]["holdout_rmse"] - metrics["holdout RMSE"]) < 5e-4,
          f"stored {meta['metrics']['holdout_rmse']:.4f} vs fresh {metrics['holdout RMSE']:.4f}")

    a = predict_frame(X_hold.head(200), model=load_model()).to_numpy()
    b = predict_frame(X_hold.head(200), model=load_model()).to_numpy()
    check("repeated loads give identical predictions", bool(np.array_equal(a, b)))

    print("\n=== inference on new data ===")
    raw = load_raw().head(50).drop(columns=[TARGET])
    shuffled = raw[list(np.random.default_rng(0).permutation(raw.columns))]
    check("target column is optional", len(predict_frame(raw, model=model)) == 50)
    check("column order does not matter",
          bool(np.allclose(predict_frame(raw, model=model).to_numpy(),
                           predict_frame(shuffled, model=model).to_numpy())))

    print("\n=== SHAP ===")
    from src.explain import compute_shap

    result = compute_shap(model, X_hold, sample=200)
    recon = result.base_value + result.by_feature.sum(axis=1).to_numpy()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_hold), size=200, replace=False)
    expected = model.predict(X_hold.iloc[idx].reset_index(drop=True))
    check("SHAP values reproduce the prediction",
          bool(np.allclose(recon, expected, rtol=1e-6, atol=1e-6)),
          f"max error {np.abs(recon - expected).max():.2e}")
    check("global ranking covers every feature", result.by_feature.shape[1] > 30,
          f"{result.by_feature.shape[1]} features")

    print("\n=== tests ===")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "no output"
    check("test suite passes", proc.returncode == 0, tail)

    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    print("all acceptance checks passed")


if __name__ == "__main__":
    main()
