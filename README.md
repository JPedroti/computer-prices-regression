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
│   ├── preprocessing.py the three column representations
│   ├── models.py        model factory: every model is a full pipeline
│   ├── ensemble.py      weighted blending and stacking
│   ├── evaluate.py      metrics, cross-validation, bootstrap overfitting rule
│   ├── train.py         fits and serialises the final model
│   ├── explain.py       SHAP global and local analysis
│   ├── inference.py     end-to-end prediction on new CSVs
│   └── utils.py         seeding and experiment tracking
├── experiments/
│   ├── exp01..exp09_*.py   one script per investigation
│   ├── results.csv         every run, with the full schema
│   └── experiment_log.md   hypotheses, findings and decisions
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

### Why the model looks the way it does

The investigation is recorded in `experiments/experiment_log.md`. The short
version:

1. The dataset is close to a **sum of per-component contributions**. A
   saturated additive model beat a 1,500-tree LightGBM by 7 RMSE points on the
   same split.
2. A log-space (multiplicative) fit was clearly worse, so the noise is additive
   on the price scale.
3. **No exploitable interactions were found.** Boosting the additive model's
   residual made it worse, and explicit interaction terms did not produce a
   paired improvement across folds.
4. Consequently the final model is an additive one. That choice also removes
   almost all of the train/validation gap, which is what the overfitting
   criterion rewards.

### Overfitting rule

Implemented exactly as specified, in `src.evaluate.overfitting_report`:
the model is trained on the training split only; RMSE is computed on train and
on validation; each is given a 95% percentile-bootstrap confidence interval
built by resampling **observations** (2,000 replicates, seed 12345); the
intervals are then compared. Overlapping intervals mean no overfitting.

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

* SHAP describes what the model does, not what causes prices. No causal claim
  is made anywhere in this repository.
* The secret test set is assumed to be drawn from the same distribution as the
  development data. This is a working hypothesis, not a guarantee, so no
  modelling choice depends on the particular way the data was split.
