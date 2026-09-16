"""Train and serialize the final model.

The model specification lives in ``configs/final_model.json`` so that training
is driven by configuration rather than by edits to the code.

Protocol (CLAUDE.md section 17)
-------------------------------
The final model is fitted on the development split only. The sealed holdout is
then used exactly once, to apply the official overfitting rule, and never to
choose anything.

    python -m src.train
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    FINAL_MODEL_PATH,
    HOLDOUT_FRACTION,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from .data import load_dev_holdout
from .evaluate import mae, overfitting_report, rmse
from .features import FeatureConfig
from .models import build_model
from .utils import set_seed

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "final_model.json"
METADATA_PATH = PROJECT_ROOT / "models" / "final_model_metadata.json"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_from_config(cfg: dict[str, Any]):
    """Instantiate the model described by a configuration dictionary."""
    kind = cfg.get("kind", "single")
    feature_config = FeatureConfig(**cfg.get("feature_config", {}))
    seed = cfg.get("seed", RANDOM_SEED)

    if kind == "single":
        return build_model(
            cfg["model"],
            feature_config=feature_config,
            preprocessor=cfg.get("preprocessor"),
            target_transform=cfg.get("target_transform"),
            seed=seed,
            **cfg.get("hyperparameters", {}),
        )

    if kind == "ensemble":
        from .ensemble import WeightedEnsemble

        members = []
        for spec in cfg["members"]:
            member_features = FeatureConfig(**spec.get("feature_config", cfg.get("feature_config", {})))
            members.append(
                (
                    spec.get("name", spec["model"]),
                    build_model(
                        spec["model"],
                        feature_config=member_features,
                        preprocessor=spec.get("preprocessor"),
                        target_transform=spec.get("target_transform"),
                        seed=spec.get("seed", seed),
                        **spec.get("hyperparameters", {}),
                    ),
                )
            )
        return WeightedEnsemble(members=members, weights=cfg.get("weights"))

    raise ValueError(f"unknown config kind {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the final model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default=str(FINAL_MODEL_PATH))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", RANDOM_SEED))

    X_dev, y_dev, X_hold, y_hold = load_dev_holdout()
    print(f"config      : {cfg.get('name', 'unnamed')}")
    print(f"development : {len(X_dev)} rows   holdout: {len(X_hold)} rows")

    model = build_from_config(cfg)
    print("fitting on the development split ...")
    model.fit(X_dev, y_dev)

    pred_dev = model.predict(X_dev)
    pred_hold = model.predict(X_hold)

    metrics = {
        "train_rmse": rmse(y_dev, pred_dev),
        "train_mae": mae(y_dev, pred_dev),
        "holdout_rmse": rmse(y_hold, pred_hold),
        "holdout_mae": mae(y_hold, pred_hold),
    }
    print(
        "\ntrain   RMSE={train_rmse:8.3f}  MAE={train_mae:8.3f}\n"
        "holdout RMSE={holdout_rmse:8.3f}  MAE={holdout_mae:8.3f}".format(**metrics)
    )

    print("\n=== official overfitting rule ===")
    report = overfitting_report(
        y_dev, pred_dev, y_hold, pred_hold, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED
    )
    print(report)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"\nsaved model -> {args.output}  ({size_mb:.1f} MB)")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "holdout_fraction": HOLDOUT_FRACTION,
        "split_seed": RANDOM_SEED,
        "n_train_rows": int(len(X_dev)),
        "n_holdout_rows": int(len(X_hold)),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "overfitting_rule": report.to_dict(),
        "artifact_mb": round(size_mb, 2),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"saved metadata -> {METADATA_PATH.name}")


if __name__ == "__main__":
    main()
