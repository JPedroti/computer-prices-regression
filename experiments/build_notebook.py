"""Generate ``notebooks/01_exploration_and_modeling.ipynb``.

The notebook is the narrative of the project and must run top to bottom. Heavy
sweeps stay in the ``experiments/*.py`` scripts; the notebook re-runs a compact,
representative subset live and reads ``experiments/results.csv`` for the full
history, so nothing important is hidden but execution stays reasonable.

    python experiments/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_exploration_and_modeling.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


CELLS = [
    md("""
# Computer Prices — Regression Challenge

**Goal.** Predict `price` from hardware specifications, minimising RMSE on a
secret test set, while satisfying the challenge's overfitting, reproducibility
and explainability criteria.

**How to read this notebook.** It tells the story end to end: audit, then
exploration, then the experiments that decided the model, then the final model,
the official overfitting rule and the SHAP analysis.

The exhaustive sweeps live in `experiments/exp*.py` and their results in
`experiments/results.csv`; this notebook re-runs a representative subset live
so every claim can be checked, and loads the full history where re-running all
of it would take an hour.
"""),
    code("""
import sys, warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

from src.config import RANDOM_SEED, TARGET
from src.utils import set_seed

set_seed(RANDOM_SEED)
print("seed fixed at", RANDOM_SEED)
"""),
    md("""
## 1. Loading the data
"""),
    code("""
from src.data import load_raw

df = load_raw()
print(f"shape: {df.shape}")
df.head()
"""),
    md("""
## 2. Dataset audit

Before any modelling: types, missing values, cardinality, duplicates.
"""),
    code("""
audit = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "n_missing": df.isna().sum(),
    "nunique": df.nunique(dropna=True),
})
audit["pct_unique"] = (audit["nunique"] / len(df) * 100).round(2)
audit
"""),
    code("""
print("full-row duplicates          :", df.duplicated().sum())
print("duplicates ignoring ID       :", df.drop(columns=['ID']).duplicated().sum())
feat = [c for c in df.columns if c not in ('ID', TARGET)]
print("duplicates on features only  :", df[feat].duplicated().sum())
print("ID unique                    :", df['ID'].is_unique, "| range:", df['ID'].min(), "-", df['ID'].max())
print("total missing cells          :", int(df.isna().sum().sum()))
"""),
    md("""
No missing values and no duplicates at all. Every row is a unique combination of
features, which also means the irreducible noise cannot be estimated by
comparing repeated specifications — it has to be inferred from model behaviour
instead (section 8).

### Is `ID` informative?

`ID` is unique and spans 0–99,999 while only 80,000 rows are present. That is
consistent with the development set being a random subset of a larger file, but
that remains a **hypothesis about the data, not a fact about the challenge**, and
no modelling decision below depends on it. What matters is the check that `ID`
carries no signal, so it can be dropped rather than exploited.
"""),
    code("""
print("Spearman(ID, price):", round(df[['ID', TARGET]].corr(method='spearman').iloc[0, 1], 5))
df.groupby(pd.qcut(df['ID'], 10, labels=False))[TARGET].mean().round(1).rename("mean price by ID decile")
"""),
    md("""
## 3. The target
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(df[TARGET], bins=80, ax=axes[0], color="steelblue")
axes[0].set_title("price")
sns.histplot(np.log(df[TARGET]), bins=80, ax=axes[1], color="darkorange")
axes[1].set_title("log(price)")
plt.tight_layout(); plt.show()

print(df[TARGET].describe(percentiles=[.01, .25, .5, .75, .95, .99]).round(2).to_string())
print("\\nskew:", round(df[TARGET].skew(), 3))
cents = (df[TARGET] * 100 % 100).round().astype(int)
print("distinct cent endings:", cents.unique(), "-> every price ends in .99")
"""),
    md("""
The target is right-skewed (skew 0.96) with a long upper tail reaching 9,772.99
against a median of 1,863.99. Every single price ends in `.99`, so the
underlying quantity is a whole number of currency units. Snapping predictions
back to `.99` would change RMSE by about 0.0002 — far too small to be worth the
extra machinery, so it is not done.

## 4. Hidden structure in the text columns

`model` and `cpu_model` look high-cardinality but are compositions of an
informative part and a random part. The test below compares the variance
explained by grouping on a column against what a purely random grouping of the
same size would explain: a ratio near 1 means noise.
"""),
    code("""
work = df.copy()
work['model_line'] = work['model'].str.split().str[1]
work['model_code'] = work['model'].str.split().str[2]
work['cpu_family'] = work['cpu_model'].str.replace(r'[-\\s]?\\d{3,}$', '', regex=True).str.strip()

overall_var = work[TARGET].var()
rows = []
for col in ['model_code', 'model_line', 'cpu_family', 'brand', 'gpu_model']:
    g = work.groupby(col)[TARGET].agg(['size', 'mean'])
    between = np.average((g['mean'] - work[TARGET].mean())**2, weights=g['size'])
    expected_if_random = overall_var * (len(g) - 1) / len(work)
    rows.append({"column": col, "n_levels": len(g),
                 "between_var": between, "expected_if_noise": expected_if_random,
                 "ratio": between / expected_if_random})
pd.DataFrame(rows).round(2)
"""),
    md("""
`model_code` sits at a ratio of ~1.0: the three-character suffix is a random
identifier and is dropped. `model_line`, `cpu_family`, `brand` and `gpu_model`
carry real signal and are kept.

The same check on the trailing number of `cpu_model`, **conditioned on the CPU
family**, gives a ratio around 0.5 — it looks informative only because it is
confounded with the family. It is dropped too.

## 5. Structural zeros

Some zeros are not measurements but "not applicable".
"""),
    code("""
print(pd.crosstab(df['device_type'], df['battery_wh'].eq(0)).rename(columns={False: 'battery>0', True: 'battery==0'}))
print()
print(pd.crosstab(df['device_type'], df['psu_watts'].eq(0)).rename(columns={False: 'psu>0', True: 'psu==0'}))
"""),
    md("""
`battery_wh` and `charger_watts` are zero for exactly every Desktop, and
`psu_watts` is zero for exactly every Laptop. Treating those zeros as numbers
would tell a linear model that a desktop has a zero-capacity battery. They are
encoded as missing plus an applicability flag.

## 6. Relationships with the target
"""),
    code("""
num_cols = df.select_dtypes(include=[np.number]).columns.drop(['ID', TARGET])
corr = df[list(num_cols) + [TARGET]].corr(method='spearman')[TARGET].drop(TARGET)
corr = corr.sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(8, 6))
corr.plot.barh(ax=ax, color=np.where(corr > 0, "steelblue", "indianred"))
ax.invert_yaxis(); ax.set_title("Spearman correlation with price"); ax.set_xlabel("rho")
plt.tight_layout(); plt.show()
corr.round(3)
"""),
    code("""
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), ['ram_gb', 'cpu_tier', 'gpu_tier', 'brand', 'device_type', 'storage_type']):
    order = None
    if df[col].dtype != object and str(df[col].dtype) not in ('str', 'string'):
        order = sorted(df[col].unique())
    sns.boxplot(data=df, x=col, y=TARGET, ax=ax, order=order, showfliers=False)
    ax.set_title(f"price by {col}")
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.show()
"""),
    md("""
`ram_gb`, `gpu_tier` and `cpu_tier` dominate. Note that the response to `ram_gb`
is **not** linear in the raw value — which is why the modelling section below
compares a representation that enters numbers as single linear terms against one
that gives every observed level its own coefficient.

## 7. Validation protocol

The 80,000 rows are split once, with a fixed seed:

* **development, 64,000 rows** — every decision is made here, by 5-fold CV with
  shared folds so model comparisons are *paired*;
* **holdout, 16,000 rows** — sealed, used once, only for the official
  overfitting rule.
"""),
    code("""
from src.data import load_dev_holdout

X_dev, y_dev, X_hold, y_hold = load_dev_holdout()
print(f"development: {X_dev.shape}   holdout: {X_hold.shape}")
print(f"mean price  dev={y_dev.mean():.2f}  holdout={y_hold.mean():.2f}")
"""),
    md("""
## 8. Feature engineering and baselines

Feature families are switchable so each can be measured separately.
"""),
    code("""
from src.features import FeatureConfig, FeatureEngineer

cfg = FeatureConfig()
print("enabled families:", cfg.enabled())

engineered = FeatureEngineer(cfg).fit_transform(X_dev)
print(f"{X_dev.shape[1]} raw columns -> {engineered.shape[1]} engineered columns")
engineered.head(3)
"""),
    code("""
from src.evaluate import cross_validate_model
from src.models import build_model
from src.config import CV_FOLDS

# A compact, representative re-run. The full sweep lives in experiments/exp01.
LIVE = [
    ("mean baseline",        build_model("mean")),
    ("ridge | onehot",       build_model("ridge", preprocessor="onehot", alpha=10.0)),
    ("ridge | levels_plus",  build_model("ridge", preprocessor="levels_plus", alpha=100.0)),
    ("lightgbm",             build_model("lgbm", n_estimators=400, learning_rate=0.06)),
]

live_results = []
for label, model in LIVE:
    res = cross_validate_model(model, X_dev, y_dev, name=label,
                               n_splits=CV_FOLDS, n_repeats=1, seed=RANDOM_SEED, return_oof=False)
    live_results.append({"model": label, "val_rmse": res.val_rmse, "train_rmse": res.train_rmse,
                         "gap": res.gap, "val_mae": res.val_mae})
    print(res.summary())

pd.DataFrame(live_results).round(3)
"""),
    md("""
Two results drive everything that follows.

1. A **linear** model is competitive with gradient boosting on this data.
2. Expanding every low-cardinality column into one dummy per level (the
   `levels_plus` representation) beats both — and does so with a train/validation
   gap of about one point, whereas LightGBM's gap is over thirty.

The full model sweep, including random forests, extra trees, XGBoost and
CatBoost, is in the experiment log.
"""),
    code("""
from src.utils import ExperimentTracker

history = ExperimentTracker.load()
print(f"{len(history)} logged experiments")
(history.sort_values("validation_rmse")
        [["experiment_id", "model", "preprocessing", "hyperparameters",
          "train_rmse", "validation_rmse", "validation_mae", "notes"]]
        .head(15).reset_index(drop=True))
"""),
    md("""
## 9. What shape is the generating process?

Three questions were asked directly (`experiments/exp02_structure.py`,
`exp06_interactions.py`, `exp08_additive_boosting.py`):

* **Is it multiplicative?** A log-space fit, with a Duan smearing correction so
  the comparison is fair, was clearly worse. The noise is additive on the price
  scale.
* **Are there interactions?** Boosting the additive model's residual made
  predictions *worse*, and the residual importances were flat and diffuse —
  the signature of noise. Explicit interaction terms were then added to the
  linear model, which is a direct test that cannot be confounded by a booster
  overfitting; see the log for the outcome.
* **Does a booster help if forced to be additive?** Tested with depth-1 trees
  and with per-feature interaction constraints.

The cell below reproduces the most important of these checks at small scale.
"""),
    code("""
from src.evaluate import rmse
from sklearn.model_selection import train_test_split

idx_tr, idx_va = train_test_split(np.arange(len(X_dev)), test_size=0.25, random_state=RANDOM_SEED)
Xtr, Xva = X_dev.iloc[idx_tr], X_dev.iloc[idx_va]
ytr, yva = y_dev.iloc[idx_tr], y_dev.iloc[idx_va]

additive = build_model("ridge", preprocessor="levels_plus", alpha=100.0).fit(Xtr, ytr)
p_add = additive.predict(Xva)
print(f"additive Ridge            valRMSE={rmse(yva, p_add):8.3f}")

log_model = build_model("ridge", preprocessor="levels_plus", alpha=100.0, target_transform="log").fit(Xtr, ytr)
print(f"multiplicative (log space) valRMSE={rmse(yva, log_model.predict(Xva)):8.3f}")

booster = build_model("lgbm", n_estimators=1500, learning_rate=0.05, num_leaves=63).fit(Xtr, ytr)
print(f"unconstrained LightGBM     valRMSE={rmse(yva, booster.predict(Xva)):8.3f}")
"""),
    md("""
## 10. The error tail

A small number of rows carry a large share of the squared error. The important
statistical caveat: a positive bias in the top decile of the **true** price is
expected even from a perfect model, because conditioning on a high `y` also
selects rows with a large positive noise draw. Calibration must be read on
deciles of the **prediction**.
"""),
    code("""
from src.evaluate import error_concentration, residual_by_decile

pred_dev = additive.predict(Xva)
print("=== share of squared error carried by the worst rows ===")
display(error_concentration(yva, pred_dev).round(4))

print("\\n=== by decile of TRUE price (selection artefact expected) ===")
display(residual_by_decile(yva, pred_dev).round(2))

resid = yva.to_numpy() - pred_dev
d = pd.qcut(pred_dev, 10, labels=False, duplicates="drop")
by_pred = pd.DataFrame({"pred": pred_dev, "err": resid, "d": d}).groupby("d").agg(
    n=("err", "size"), pred_level=("pred", "mean"),
    bias=("err", "mean"), resid_sd=("err", "std"))
print("\\n=== by decile of PREDICTED price (true calibration check) ===")
display(by_pred.round(2))
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(pred_dev, yva, s=4, alpha=0.15, color="steelblue")
lims = [min(pred_dev.min(), yva.min()), max(pred_dev.max(), yva.max())]
axes[0].plot(lims, lims, "k--", lw=1)
axes[0].set_xlabel("predicted"); axes[0].set_ylabel("actual"); axes[0].set_title("predicted vs actual")

axes[1].scatter(pred_dev, resid, s=4, alpha=0.15, color="indianred")
axes[1].axhline(0, color="k", ls="--", lw=1)
axes[1].set_xlabel("predicted"); axes[1].set_ylabel("residual"); axes[1].set_title("residuals vs prediction")
plt.tight_layout(); plt.show()
"""),
    md("""
## 11. Choosing the final model

Three candidates survived the screening. The choice between them weighs
predictive accuracy, overfitting margin, stability, complexity and
explainability together, as the challenge's own criteria require — not RMSE
alone.

| | Ridge `levels_plus` | CatBoost | **blend of the two** |
|---|---|---|---|
| pooled out-of-fold RMSE | 212.34 | 211.69 | **211.33** |
| overfitting margin, mean over inner splits | +27.7 | +20.1 | +22.9 |
| overfitting margin, worst split | +25.2 | +17.5 | +20.4 |
| deterministic across refits | yes | yes | yes |
| artefact size | 24 KB | ~2 MB | ~2 MB |
| SHAP | exact (linear) | exact (tree) | exact (weighted sum) |

For contrast, an ordinary LightGBM reaches 215.5 and **fails the overfitting
rule on every inner split** (margin −13), and HistGradientBoosting passes with a
worst-case margin of only 7.5. The models that fit this data well are also the
ones that satisfy the constraint; that is not a coincidence, it is what happens
when the model form matches the process.

The blend takes 0.372 of the Ridge and 0.628 of CatBoost. Those weights were
fitted by non-negative least squares on out-of-fold predictions, and the gain
survives fitting them on rows they are not scored on.

Adding HistGradientBoosting, LightGBM and the one-hot Ridge to the blend buys
0.025 RMSE, so they are left out.

## 12. The official overfitting rule

The rule, implemented exactly as specified in `src.evaluate.overfitting_report`:
train the final model on the training split only; compute RMSE on train and on
validation; give each a 95% percentile bootstrap confidence interval by
resampling **observations** (2,000 replicates, fixed seed); compare the
intervals. Overlap means no overfitting.
"""),
    code("""
import json
from src.train import build_from_config, load_config
from src.evaluate import overfitting_report, mae

config = load_config()
print(json.dumps(config, indent=2))

final_model = build_from_config(config)
final_model.fit(X_dev, y_dev)

pred_train = final_model.predict(X_dev)
pred_hold = final_model.predict(X_hold)

print(f"\\ntrain   RMSE={rmse(y_dev, pred_train):8.3f}  MAE={mae(y_dev, pred_train):8.3f}")
print(f"holdout RMSE={rmse(y_hold, pred_hold):8.3f}  MAE={mae(y_hold, pred_hold):8.3f}")
"""),
    code("""
report = overfitting_report(y_dev, pred_train, y_hold, pred_hold)
print(report)
"""),
    code("""
fig, ax = plt.subplots(figsize=(9, 3))
for i, (label, d, color) in enumerate([("train", report.train, "steelblue"),
                                       ("validation", report.validation, "darkorange")]):
    ax.plot([d["ci_low"], d["ci_high"]], [i, i], lw=8, color=color, solid_capstyle="round", alpha=.8)
    ax.plot(d["rmse"], i, "o", color="black", zorder=3)
    ax.text(d["ci_high"] + 0.4, i, f"{d['rmse']:.2f}", va="center")
ax.set_yticks([0, 1]); ax.set_yticklabels(["train", "validation"])
ax.set_xlabel("RMSE"); ax.set_title("95% bootstrap confidence intervals — official overfitting rule")
plt.tight_layout(); plt.show()
"""),
    md("""
## 13. SHAP — global

SHAP attributes the model's output to its inputs. It describes **what the model
does**, not what causes prices in the world; nothing here is a causal claim.
"""),
    code("""
from src.explain import compute_shap, global_report, local_report

# The saturated representation spreads one feature over many columns, and a
# weighted ensemble spreads it over members too. SHAP values are additive, so
# both are summed back to one number per engineered feature.
shap_result = compute_shap(final_model, X_hold, sample=1500)
print("estimator:", shap_result.estimator_name)
print("features :", shap_result.by_feature.shape[1])

ranking = global_report(shap_result)
ranking.head(20).round(2)
"""),
    code("""
top = ranking.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top['feature'], top['mean_abs_shap'], color="steelblue")
ax.set_xlabel("mean |SHAP| (in price units)")
ax.set_title("Which features move this model's predictions most")
plt.tight_layout(); plt.show()
"""),
    md("""
The ranking is led by the component specifications — GPU tier, display type,
CPU tier and clock — which is what one would expect to drive a machine's price.

One contrast is worth pausing on. In section 6, `ram_gb` had the **highest**
rank correlation with price (0.79), yet here it sits well below `gpu_tier`.
There is no contradiction: a correlation is *marginal*, measuring `ram_gb`
against price while ignoring everything else, and machines with a lot of RAM
also tend to have a strong GPU and a fast CPU. SHAP is *conditional*: it asks
what `ram_gb` adds once the rest of the specification is already known, and the
answer is much less. The two numbers answer different questions.

Read all of this as a description of the fitted model's behaviour — which inputs
it leans on — not as evidence about what causes prices in the market.
"""),
    md("""
## 14. SHAP — local

One individual prediction, decomposed into the features that pushed it up and
those that pushed it down.
"""),
    code("""
ROW = 0
base, contrib = local_report(shap_values, transformed, display_rows, row=ROW)
print(f"base value (mean prediction) = {base:.2f}")
print(f"model prediction             = {base + contrib['shap'].sum():.2f}\\n")

print("pushing the prediction UP:")
display(contrib[contrib['shap'] > 0].head(8)[['feature', 'value', 'shap']].round(2))
print("\\npushing the prediction DOWN:")
display(contrib[contrib['shap'] < 0].head(8)[['feature', 'value', 'shap']].round(2))
"""),
    code("""
head = contrib.head(14).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh([f"{f} = {v}" for f, v in zip(head['feature'], head['value'])],
        head['shap'], color=np.where(head['shap'] < 0, "indianred", "steelblue"))
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("contribution to the predicted price")
ax.set_title(f"Why the model predicted {shap_result.prediction(ROW):,.0f} for observation {ROW}")
plt.tight_layout(); plt.show()
"""),
    md("""
Reading the example above: the model starts from the average prediction and then
adds or subtracts for each specification. A high GPU tier and large storage push
the estimate up; a mechanical hard drive and a plain IPS panel pull it down.

Again, this is the model's internal accounting, not a causal statement about the
computer market.
"""),
    md("""
## 15. Inference on new data

The delivered artefact is a single pipeline: raw columns in, predictions out.
No manual preprocessing, no retraining.
"""),
    code("""
import joblib
from src.config import FINAL_MODEL_PATH
from src.inference import predict_frame

joblib.dump(final_model, FINAL_MODEL_PATH)
reloaded = joblib.load(FINAL_MODEL_PATH)

# Simulate "new data": holdout rows, target column dropped, order shuffled.
new_data = X_hold.sample(frac=1.0, random_state=0).reset_index(drop=True)
preds = predict_frame(new_data, model=reloaded)

print(f"{len(preds)} predictions for {len(new_data)} input rows")
print(f"order preserved: {len(preds) == len(new_data) and list(preds.index) == list(new_data.index)}")
preds.head()
"""),
    md("""
## 16. Conclusion

**What the data turned out to be.** The price is close to a sum of
per-component contributions. Three independent tests agree: boosting the
additive model's residual makes it worse with flat, diffuse importances;
explicit interaction terms produce changes of at most 0.13 RMSE and hurt when
combined; and forcing a booster to be additive improves it by 5.3 points. A
log-space fit is clearly worse, so the noise is additive on the price scale.

**What that implied for modelling.** A saturated additive linear model — one
coefficient per observed level of each feature — outperformed random forests,
extra trees, XGBoost, LightGBM and HistGradientBoosting, all of which spend
capacity on interactions the data does not contain. CatBoost was the exception,
and blending the two is better than either alone, by a margin that survives
fitting the weights out of sample.

**Why the same choice wins on the other criteria.** Because the model form
matches the process, training and validation errors nearly coincide, so the
bootstrap intervals overlap with room to spare rather than marginally. The
contrast is stark: an ordinary LightGBM, only 4 RMSE points behind, **fails the
overfitting criterion on every split tested**. Here there was no trade-off to
make between accuracy and the constraint — but there would have been, had the
data been different, and that is why the margin was measured rather than
assumed.

**What was ruled out, with evidence.** Multiplicative structure; interactions;
a tail correction — the apparent bias among expensive machines is a selection
artefact of conditioning on the true target, and the calibration measured on
deciles of the *prediction* is flat. All of the engineered feature families
turned out to be redundant under the saturated representation, which is worth
saying plainly: the gain came from the representation, not from the feature
engineering.

**Honest limitations.** The remaining error is, as far as these experiments can
tell, irreducible noise. Its spread is proportional to price at about 11% with
an extremely heavy right tail, the worst 0.1% of rows carry 38% of all squared
error while belonging to no identifiable segment, and the model's RMSE already
equals the RMSE that this noise alone implies. A flexible booster drives
training error far below the additive model's while making validation error
worse — the signature of fitting noise.

The secret test set is assumed to share the development distribution. That is a
working hypothesis, not a guarantee, and no modelling choice here depends on how
the data was split.
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(NOTEBOOK, fh, indent=1)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
