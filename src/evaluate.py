"""Metrics, cross-validation and the official overfitting rule.

The overfitting procedure implemented in :func:`overfitting_report` follows
CLAUDE.md section 17 literally:

1. split the data into train and validation;
2. train the final model on train only;
3. compute train RMSE;
4. compute validation RMSE;
5. bootstrap the train observations;
6. obtain the 95% CI of the train RMSE;
7. bootstrap the validation observations separately;
8. obtain the 95% CI of the validation RMSE;
9. compare the intervals.

Overlapping intervals -> no overfitting (15 points). Disjoint intervals with a
larger validation error -> overfitting (0 points).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from .config import BOOTSTRAP_N, BOOTSTRAP_SEED, CV_FOLDS, CV_REPEATS, RANDOM_SEED


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
@dataclass
class CVResult:
    """Aggregated cross-validation outcome plus out-of-fold predictions."""

    model_name: str
    train_rmse: float
    val_rmse: float
    train_mae: float
    val_mae: float
    val_rmse_std: float
    fold_val_rmse: list[float] = field(default_factory=list)
    oof_pred: np.ndarray | None = None
    fit_seconds: float = 0.0

    @property
    def gap(self) -> float:
        """Validation minus train RMSE: the overfitting pressure indicator."""
        return self.val_rmse - self.train_rmse

    def summary(self) -> str:
        return (
            f"{self.model_name:34s} valRMSE={self.val_rmse:8.3f} (+-{self.val_rmse_std:5.3f}) "
            f"trainRMSE={self.train_rmse:8.3f} gap={self.gap:+7.3f} "
            f"valMAE={self.val_mae:7.3f} [{self.fit_seconds:.1f}s]"
        )


def cross_validate_model(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    name: str = "model",
    n_splits: int = CV_FOLDS,
    n_repeats: int = CV_REPEATS,
    seed: int = RANDOM_SEED,
    return_oof: bool = True,
) -> CVResult:
    """Repeated K-Fold CV returning train and validation metrics plus OOF predictions.

    The estimator is cloned for every fold, so any statistic it learns comes
    from that fold's training rows only.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)

    oof_sum = np.zeros(len(X), dtype=float)
    oof_count = np.zeros(len(X), dtype=float)
    tr_rmse, va_rmse, tr_mae, va_mae = [], [], [], []

    start = time.perf_counter()
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for tr_idx, va_idx in kf.split(X):
            model = clone(estimator)
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            model.fit(X_tr, y_tr)
            p_tr = model.predict(X_tr)
            p_va = model.predict(X_va)
            tr_rmse.append(rmse(y_tr, p_tr))
            va_rmse.append(rmse(y_va, p_va))
            tr_mae.append(mae(y_tr, p_tr))
            va_mae.append(mae(y_va, p_va))
            if return_oof:
                oof_sum[va_idx] += p_va
                oof_count[va_idx] += 1
    elapsed = time.perf_counter() - start

    oof = None
    if return_oof:
        with np.errstate(invalid="ignore"):
            oof = np.where(oof_count > 0, oof_sum / np.maximum(oof_count, 1), np.nan)

    return CVResult(
        model_name=name,
        train_rmse=float(np.mean(tr_rmse)),
        val_rmse=float(np.mean(va_rmse)),
        train_mae=float(np.mean(tr_mae)),
        val_mae=float(np.mean(va_mae)),
        val_rmse_std=float(np.std(va_rmse, ddof=1)) if len(va_rmse) > 1 else 0.0,
        fold_val_rmse=[float(v) for v in va_rmse],
        oof_pred=oof,
        fit_seconds=elapsed,
    )


# --------------------------------------------------------------------------- #
# Bootstrap and the official overfitting rule
# --------------------------------------------------------------------------- #
def bootstrap_rmse_ci(
    y_true,
    y_pred,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
    batch: int = 250,
) -> dict[str, Any]:
    """Percentile bootstrap CI for the RMSE.

    Observations (not residuals) are resampled with replacement: each replicate
    draws ``n`` row indices from the ``n`` available rows, recomputes the RMSE,
    and the interval is taken from the empirical percentiles of those replicate
    RMSEs. Resampling is done in batches to bound memory use.
    """
    err_sq = (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2
    n = err_sq.size
    rng = np.random.default_rng(seed)

    replicates = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        size = min(batch, n_boot - done)
        idx = rng.integers(0, n, size=(size, n))
        replicates[done : done + size] = np.sqrt(err_sq[idx].mean(axis=1))
        done += size

    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "rmse": float(np.sqrt(err_sq.mean())),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_obs": int(n),
        "method": "percentile bootstrap over observations",
    }


@dataclass
class OverfittingReport:
    """Result of the official overfitting rule."""

    train: dict[str, Any]
    validation: dict[str, Any]
    intervals_overlap: bool
    verdict: str
    points: int
    margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_rmse": self.train["rmse"],
            "train_ci_low": self.train["ci_low"],
            "train_ci_high": self.train["ci_high"],
            "validation_rmse": self.validation["rmse"],
            "validation_ci_low": self.validation["ci_low"],
            "validation_ci_high": self.validation["ci_high"],
            "intervals_overlap": self.intervals_overlap,
            "verdict": self.verdict,
            "points": self.points,
            "margin": self.margin,
            "n_boot": self.train["n_boot"],
            "bootstrap_seed": self.train["seed"],
        }

    def __str__(self) -> str:
        t, v = self.train, self.validation
        return (
            f"train RMSE      = {t['rmse']:8.3f}   95% CI [{t['ci_low']:8.3f}, {t['ci_high']:8.3f}]\n"
            f"validation RMSE = {v['rmse']:8.3f}   95% CI [{v['ci_low']:8.3f}, {v['ci_high']:8.3f}]\n"
            f"overlap = {self.intervals_overlap}  margin = {self.margin:+.3f}\n"
            f"verdict: {self.verdict}  ->  {self.points} points"
        )


def overfitting_report(
    y_train,
    pred_train,
    y_val,
    pred_val,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> OverfittingReport:
    """Apply the official rule to predictions of a model trained on train only."""
    train = bootstrap_rmse_ci(y_train, pred_train, n_boot=n_boot, seed=seed)
    validation = bootstrap_rmse_ci(y_val, pred_val, n_boot=n_boot, seed=seed + 1)

    # Intervals overlap unless one lies entirely above the other.
    overlap = not (train["ci_high"] < validation["ci_low"] or validation["ci_high"] < train["ci_low"])
    # Positive margin = how far the train CI upper bound reaches past the
    # validation CI lower bound; the safety buffer we want to keep.
    margin = float(train["ci_high"] - validation["ci_low"])

    if overlap:
        verdict, points = "no overfitting (CIs overlap)", 15
    elif validation["rmse"] > train["rmse"]:
        verdict, points = "overfitting (disjoint CIs, validation error larger)", 0
    else:
        verdict, points = "disjoint CIs but validation error smaller (not overfitting)", 15

    return OverfittingReport(train, validation, overlap, verdict, points, margin)


def error_concentration(y_true, y_pred, quantiles=(0.001, 0.01, 0.05, 0.10)) -> pd.DataFrame:
    """How much of the total squared error the worst rows account for."""
    err_sq = np.sort((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)[::-1]
    total = err_sq.sum()
    rows = []
    for q in quantiles:
        k = max(1, int(round(q * err_sq.size)))
        rows.append(
            {
                "top_fraction": q,
                "n_rows": k,
                "share_of_squared_error": err_sq[:k].sum() / total,
                "rmse_excluding_them": float(np.sqrt(err_sq[k:].sum() / err_sq.size)),
            }
        )
    return pd.DataFrame(rows)


def residual_by_decile(y_true, y_pred) -> pd.DataFrame:
    """Bias and RMSE per decile of the true target -- exposes tail shrinkage."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    decile = pd.qcut(y_true, 10, labels=False, duplicates="drop")
    frame = pd.DataFrame({"y": y_true, "pred": y_pred, "err": y_true - y_pred, "decile": decile})
    return frame.groupby("decile").agg(
        n=("y", "size"),
        y_mean=("y", "mean"),
        pred_mean=("pred", "mean"),
        bias=("err", "mean"),
        rmse=("err", lambda s: float(np.sqrt(np.mean(s**2)))),
    )
