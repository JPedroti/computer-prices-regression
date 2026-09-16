# Experiment log

Narrative record of the investigation. Quantitative results for every run live
in `results.csv`; this file records the *hypotheses*, what the numbers meant and
what was decided as a consequence.

Validation protocol used throughout:

* **Screening**: 5-fold CV, seed 42, on the development split (64,000 rows).
  Every experiment uses these same folds, so model-to-model comparisons are
  paired. This matters here: the fold-to-fold spread of the RMSE is large
  (±14) because a few extreme prices dominate the squared error, while paired
  differences between models on the same folds are far more precise.
* **Overfitting holdout**: 16,000 rows, split off once with seed 42 and sealed.
  Used only for the official bootstrap rule, never to select anything.

---

## Phase 0 — Dataset audit

80,000 rows, 34 columns, target `price`. No missing values, no duplicates
(neither full-row, nor ignoring `ID`, nor on features alone).

**Findings that shaped everything else:**

| Finding | Evidence | Consequence |
|---|---|---|
| `ID` is a pure identifier | Spearman vs price = 0.003; flat mean price across ID deciles | Dropped |
| Secret test is plausibly the complement of a 100k row set | `ID` unique in 0–99999, 80,000 present | Treated as a *hypothesis only*; no split-specific tricks used |
| `model` = brand + line + random code | Line has between-group variance ratio 93; the 3-char code has ratio 0.99 over 38,215 values | Keep the line, drop the code |
| `cpu_model` = family + random number | Family ratio 2345; the trailing number's ratio is 0.54 *conditioned on family* | Keep the family, drop the number |
| Structural zeros | `battery_wh`/`charger_watts` are 0 for exactly the 32,127 Desktops; `psu_watts` is 0 for exactly the 47,873 Laptops | Encoded as not-applicable (NaN) plus applicability flags |
| Target granularity | All 80,000 prices end in `.99` | Noted; snapping predictions would change RMSE by ~0.0002, so not done |
| Error is concentrated | On a probe model, 10 validation rows (0.06%) carried 30.8% of total squared error | Motivated the dedicated tail investigation (exp 03) |

No feature was found to carry target information directly or indirectly; the
leakage audit came back clean.

---

## exp01 — Baselines and one pass over every model family

**Hypothesis.** The audit suggested a nearly additive process, so linear models
should be competitive with gradient boosting rather than far behind it.

**Result.** Confirmed. With only the structural feature decompositions in place:

| Model | valRMSE | train/val gap |
|---|---|---|
| mean baseline | 578.06 | 0.00 |
| median baseline | 581.66 | -0.02 |
| Linear regression (one-hot) | 217.25 | +0.29 |
| **Ridge one-hot, alpha=10** | **217.24** | **+0.28** |
| Ridge one-hot, alpha=100 | 217.25 | +0.22 |
| Lasso, alpha=1 | 218.20 | -0.09 |
| ElasticNet, alpha=1 | 274.97 | +0.02 |
| Decision tree, depth 12 | 262.04 | +14.06 |
| Random forest, 200 trees | 235.70 | +66.82 |
| Extra trees, 200 trees | 235.96 | +83.44 |
| **HistGradientBoosting** | **214.68** | +15.48 |
| LightGBM, 400 trees | 214.89 | +32.52 |
| XGBoost, 400 trees | 217.61 | +57.92 |

Two things stand out. First, the *linear* model is within 2.6 points of the best
booster while having essentially **no train/validation gap at all** — the
boosters buy their small advantage with gaps of 15 to 58 points, which is
exactly what the overfitting criterion penalises. Second, the bagged forests are
far behind, which is what one expects when the truth is additive and smooth:
they fit deep interactions that do not exist.

Two practical notes:

* Lasso and ElasticNet were first run with alpha=0.1. Because alphas act on the
  price scale (~2000), that is nearly unpenalised, and coordinate descent failed
  to converge in reasonable time. Rerun with alpha=1 and a looser tolerance.
* The CatBoost wrapper was not clonable by scikit-learn (its constructor
  modified the stored parameter, which `clone` checks by identity), so the first
  sweep aborted there. Fixed and completed in exp01b.

---

## exp02 — What shape is the generating process?

**Hypothesis.** If price is a sum of component contributions, a *saturated
additive* model — one dummy per observed level of every feature, no
interactions — should approach the achievable floor, and whatever a booster
adds on top is interaction structure.

**Results** (single 48k/16k split inside the development data, so absolute
numbers are not comparable to CV figures; the comparisons *within* the block
are valid because they share the split):

| Model | valRMSE |
|---|---|
| Saturated additive Ridge | **225.80** |
| LightGBM, 1500 trees, 63 leaves | 232.82 |
| Log-space (multiplicative) Ridge, Duan smearing | 233.13 |
| Saturated additive + LightGBM on its residual | 230.92 |

**Three conclusions.**

1. **The process is additive, not multiplicative.** The log-space fit is 7
   points worse, so the noise is additive on the price scale.
2. **Gradient boosting is not the right tool here.** The additive model beats a
   large LightGBM by 7 points; the booster is spending capacity on noise.
3. **There are no exploitable interactions.** Boosting the additive model's
   residual made it *worse* (230.9 vs 225.8), and the residual gain
   importances are flat and diffuse (7.8%, 7.3%, 6.1%, 6.0%, …) rather than
   dominated by a few features. That is the signature of noise, not of
   structure left on the table.

**Decision.** Stop trying to out-muscle the problem with model capacity. The
remaining question is how best to *specify* the additive model.

---

## exp03 — The error tail and the noise floor

**The tail bias was an artefact.** The audit showed a +189 bias in the top
decile of the **true** price, which looks like systematic under-prediction of
expensive machines. It is not. If `y = f(x) + noise`, selecting rows with a high
`y` also selects rows with a large positive noise draw, so even a perfect
predictor of `E[y|x]` shows that bias. Judged on deciles of the **prediction**,
which is the correct calibration check, the bias is between −7.9 and +11.9
across all ten deciles — the model is well calibrated and there is no tail
correction to make.

| decile of *true* price | bias | | decile of *predicted* price | bias |
|---|---|---|---|---|
| 0 (cheapest) | −75.7 | | 0 | −0.2 |
| 9 (most expensive) | **+161.8** | | 9 | **−7.9** |

**The noise is proportional and extremely heavy-tailed.** Fitting the residual
spread against the price level: a proportional model (sd ≈ 11.4% of price) fits
far better than a constant one (SSE 9,144 vs 50,309). The relative residual has
a skew of 6.7, and the absolute residual a skew of 8.4.

**The extreme rows are noise, not missing structure.** The worst 16 rows (0.1%)
carry 37% of all squared error, but they are not concentrated in any segment —
lifts are only 2.9× for Apple and 1.8× for macOS — and they make no physical
sense as a group: one is an i5 with 16 GB of RAM priced at 5,031.99 against a
prediction of 1,545. These are large multiplicative noise draws. There is
nothing to model there, and attempts to chase them would fit noise.

**Decision.** No tail-specific treatment, no weighting, no segmentation, no
calibration layer. None of them can help against label noise, and each would
add complexity with no measurable benefit.

---

## exp05 — Feature family ablation

Every family was added or removed from the winning `levels_plus` specification
and scored on shared folds.

**Result: no family moves the RMSE by more than 0.06 points**, against a
fold-to-fold spread of ±15. Under the saturated representation, all of the
engineered features are redundant:

* `resolution` decomposition — the raw `resolution` column has only 6 levels, so
  one dummy per level already encodes width, height, pixels and aspect exactly;
* `structural_zeros` — a structural zero is simply its own level, so recoding it
  as missing changes nothing;
* `cpu_decomp` — the CPU family is fully determined by the brand, tier, cores,
  threads and clock columns that are already present;
* `model_decomp` — likewise for the product line given brand and form factor.

This is worth stating plainly: the feature engineering was **not** what produced
the gain over the boosters. The saturated additive *representation* was. The
engineered columns are kept because they cost nothing and make the model easier
to read, not because they were shown to help.

---

## exp04 — Best additive specification

**Hypothesis.** The onehot Ridge enters each numeric column as a single linear
term, which cannot represent a non-linear price response to, say, `ram_gb`.
Expanding every low-cardinality column into one dummy per level should recover
that shape while staying additive.

Note on why exp02's saturated model looked *worse* (225.8) than exp01's onehot
Ridge (217.2): those were different splits. The fold-to-fold spread here is ±14,
so single-split comparisons across experiments are meaningless. Under shared
folds the picture reverses.

| Representation | best alpha | valRMSE | gap |
|---|---|---|---|
| `onehot` (numerics linear) | 10 | 217.24 | +0.28 |
| `levels` (dummy per level) | 10 | 211.99 | +1.07 |
| **`levels_plus`** (dummies + numeric term) | **100** | **211.92** | **+0.90** |

**Result.** The saturated representation is worth **5.3 RMSE points** over the
linear-in-numerics one, and it beats every gradient booster from exp01 while
keeping the train/validation gap under one point.

`levels` and `levels_plus` are within 0.07 of each other — far inside the noise
— so the choice between them is settled on robustness grounds in exp09 rather
than on this single number.

Regularisation is flat between alpha 1 and 100 and only starts to hurt past 300,
which confirms that exp02's alpha ≤ 10 scan was simply too narrow to matter
either way.

---

## exp05 — Feature family ablation

*(in progress)*
