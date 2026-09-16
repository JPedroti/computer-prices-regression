# Computer Prices — Regression Challenge

Solution for Projeto 03 of the Data Science mentorship: predict the price of a
computer from its specifications, minimising RMSE on a secret test set while
satisfying the challenge's overfitting, reproducibility and explainability
criteria.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Train the final model (fits on the development split, applies the official
overfitting rule, writes the artefact):

```bash
python -m src.train
```

Predict on new data — a CSV with the same feature columns, with or without the
`price` column:

```bash
python -m src.inference --input data/new_machines.csv --output predictions.csv
```

Run the tests:

```bash
python -m pytest tests -q
```

Produce the SHAP analysis:

```bash
python -m src.explain
```

### Reproducing the investigation

Each experiment is a standalone script and appends to `experiments/results.csv`.
They can be run in any order, except that `exp13` needs the out-of-fold cache
that `exp07` writes. The full set takes a few hours, dominated by CatBoost:

```bash
python experiments/exp01_baselines.py
```

Regenerate the narrative notebook from its source:

```bash
python experiments/build_notebook.py
```

---

## Project structure

```text
computer-prices-regression/
├── data/
│   └── computer_prices_train_80.csv   development data (80,000 rows)
├── notebooks/
│   └── 01_exploration_and_modeling.ipynb   the full story, start to finish
├── src/
│   ├── config.py        paths, seeds, column semantics, protocol constants
│   ├── data.py          loading and the dev / sealed-holdout split
│   ├── features.py      row-wise feature engineering, one switch per family
│   ├── preprocessing.py the five column representations (native, onehot,
│   │                    ordinal, levels, levels_plus)
│   ├── models.py        model factory: every model is a full pipeline
│   ├── ensemble.py      weighted blending and stacking
│   ├── evaluate.py      metrics, cross-validation, bootstrap overfitting rule
│   ├── train.py         fits and serialises the final model
│   ├── explain.py       SHAP global and local analysis
│   ├── inference.py     end-to-end prediction on new CSVs
│   └── utils.py         seeding and experiment tracking
├── experiments/
│   ├── exp01_baselines.py            every model family, one screening pass
│   ├── exp02_structure.py            what shape is the generating process?
│   ├── exp03_tail_and_noise.py       the error tail and the noise floor
│   ├── exp04_additive_spec.py        choosing the additive specification
│   ├── exp05_feature_ablation.py     one run per feature family
│   ├── exp06_interactions.py         interactions, tested directly
│   ├── exp07_ensembles.py            blending and stacking over OOF predictions
│   ├── exp08_additive_boosting.py    boosting constrained to be additive
│   ├── exp09_robustness.py           finalists across five fold partitions
│   ├── exp10_catboost_tuning.py      targeted tuning of the strongest booster
│   ├── exp11_overfitting_margin.py   rehearsing the official rule on inner splits
│   ├── exp12_learning_curve.py       what the train-only protocol costs
│   ├── build_notebook.py             generates the narrative notebook
│   ├── results.csv                   every run, with the full schema
│   └── experiment_log.md             hypotheses, findings and decisions
├── models/
│   ├── final_model.joblib           the deliverable
│   └── final_model_metadata.json    metrics and provenance
├── reports/figures/     SHAP and diagnostic plots
├── tests/               pipeline, inference contract and rule tests
└── configs/
    └── final_model.json the final model specification
```

---

## Method

### Validation protocol

The 80,000 development rows are split once, with seed 42, into:

* **development** (64,000 rows) — every modelling decision is made here, by
  5-fold cross-validation with fixed folds so that model comparisons are
  paired;
* **holdout** (16,000 rows) — sealed, and used only to apply the official
  overfitting rule.

The holdout is never used to select a model, a feature or a hyperparameter.
The secret test set is never accessed, inspected or reasoned about beyond the
assumption that it shares the development distribution.

### What the data turned out to be

The full investigation is in `experiments/experiment_log.md`; the short version:

1. **The price is close to a sum of per-component contributions.** Three
   independent tests agree. Boosting the additive model's residual makes it
   *worse*, with flat, diffuse importances. Explicit interaction terms move the
   RMSE by at most 0.13 points and hurt when combined. And forcing a booster to
   be additive improves it by 5.3 points.
2. **The noise is additive on the price scale**, so a log-space fit loses 6–9
   points: exponentiating it targets the conditional median, and the right tail
   is heavy enough that the mean sits well above it.
3. **The remaining error is irreducible.** Its spread is proportional to price
   at roughly 11% with a very heavy right tail; the worst 0.1% of rows carry 38%
   of all squared error and belong to no identifiable segment; and the model's
   RMSE already equals what that noise alone implies.
4. **The feature engineering was not what produced the gain** — the saturated
   representation was. Under it, every engineered family is redundant, and the
   ablation says so (no family moves the RMSE by more than 0.14).

### The final model

A fixed-weight blend, 0.372 of a saturated additive Ridge and 0.628 of CatBoost.

| Candidate | pooled OOF RMSE | overfitting margin (mean / worst) |
|---|---|---|
| Ridge `levels_plus` | 212.34 | +27.7 / +25.2 |
| CatBoost | 211.69 | +20.1 / +17.5 |
| **blend of the two** | **211.33** | **+22.9 / +20.4** |
| HistGradientBoosting | 215.06 | +9.4 / +7.5 |
| LightGBM | 215.49 | −13.0 — **fails on every split** |

The blend wins on the metric that carries 55 points, and the gain survives
fitting its weights on rows they are not scored on. Adding
HistGradientBoosting, LightGBM or the one-hot Ridge to it buys 0.025 RMSE, so
they are left out — and LightGBM would bring real risk for nothing.

That last row is the point worth keeping: a perfectly ordinary, well-performing
LightGBM is only 4 RMSE points behind and would score **zero** of the 15
overfitting points.

### Result on the sealed holdout

Fitted on the 64,000 development rows, evaluated once on the 16,000 sealed rows:

```text
train RMSE      =  206.614   95% CI [196.987, 216.383]     MAE 135.72
holdout RMSE    =  232.946   95% CI [209.123, 259.959]     MAE 139.82
overlap = True   margin = +7.26   ->  no overfitting, 15 points
```

Two honest notes. The holdout RMSE (232.9) is higher than the cross-validated
estimate (211.0) because this particular block drew more of the expensive tail —
the same swing appears across inner splits, where validation RMSE ranged from
211.7 to 219.5 by seed. And the margin of +7.26 sits at the low end of what the
inner-split rehearsals predicted (+20.4 to +24.9); it is positive, so the rule
awards the full 15 points, but the spread is real and worth stating.

### Overfitting rule

Implemented exactly as specified, in `src.evaluate.overfitting_report`:
the model is trained on the training split only; RMSE is computed on train and
on validation; each is given a 95% percentile-bootstrap confidence interval
built by resampling **observations** (2,000 replicates, seed 12345); the
intervals are then compared. Overlapping intervals mean no overfitting.

Candidate models were compared using this rule on *inner* splits of the
development data (`exp11`, `exp13`), never on the sealed holdout — using the
holdout to choose between candidates would have made it a selection set and
voided the final number. The holdout is read exactly once, by `src/train.py`.

Reported values are in `models/final_model_metadata.json`.

---

## Reproducibility

Developed and tested with Python 3.11.9 and:

```text
numpy 2.4.6        pandas 3.0.5       scikit-learn 1.9.0   scipy 1.17.1
lightgbm 4.7.0     xgboost 3.2.0      catboost 1.2.10      shap 0.51.0
optuna 5.0.0       joblib 1.5.3       matplotlib 3.11.1    seaborn 0.13.2
```

* All seeds are fixed in `src/config.py` and applied through `src.utils.set_seed`.
* Every learned transformation (imputation, encoding, scaling) lives inside a
  scikit-learn pipeline, so it is re-fitted per fold and cannot leak.
* The delivered artefact is a single pipeline: raw CSV in, predictions out, no
  manual preprocessing and no retraining.
* `requirements.txt` pins the minimum versions used.

---

## Notes and limitations

* **The error left is very likely irreducible.** Its spread is proportional to
  price at about 11% with an extremely heavy right tail — the worst 0.1% of rows
  carry 38% of all squared error while belonging to no identifiable segment — and
  the model's RMSE already equals what that noise alone implies. A flexible
  booster drives training error far below the additive model's while making
  validation error worse. Further gains here are likely to be fractions of a
  point, not points.
* **The feature engineering is not load-bearing.** Under the saturated
  representation every engineered family is redundant (no family moves the RMSE
  by more than 0.14). The engineered columns are kept because they cost nothing
  measurable and make the SHAP output readable, not because they were shown to
  help. The gain over the boosters came from the representation.
* **SHAP describes what the model does, not what causes prices.** No causal claim
  is made anywhere in this repository.
* **The secret test set is assumed to share the development distribution.** This
  is a working hypothesis, not a guarantee, so no modelling choice depends on the
  particular way the data was split. `ID` was checked and dropped rather than
  exploited.
* **The holdout was read once.** Candidate models were compared with the
  overfitting rule on inner splits of the development data; the sealed holdout is
  touched only by `src/train.py`, after the model was already chosen.
