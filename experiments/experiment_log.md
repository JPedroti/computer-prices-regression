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

**Result.** Confirmed, and more strongly than expected. With only the structural
feature decompositions in place, a plain Ridge on one-hot features reached
**valRMSE 217.24** with a train/validation gap of **+0.28** — essentially no
overfitting at all. Mean baseline: 578.06.

The single decision tree (262.0, gap +14.1) and ElasticNet at alpha=1
(275.0 — over-regularised) confirm the metric behaves sensibly.

A note on cost: Lasso and ElasticNet were first run with alpha=0.1. Because the
alphas act on the price scale (~2000), that is nearly unpenalised, and
coordinate descent failed to converge within any reasonable time. Rerun with
alpha=1 and a looser tolerance.

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

*(in progress — see below once complete)*

Statistical caution recorded up front: the +189 bias observed in the top decile
of the **true** price during the audit is expected even for a perfect predictor.
Conditioning on a high `y` also selects rows with a large positive noise draw.
Calibration must therefore be judged on deciles of the **prediction**. The
experiment reports both so the artefact stays visible.

---

## exp04 — Best additive specification

*(in progress)*

Compares `onehot` (numerics as single linear terms), `levels` (a dummy per
level) and `levels_plus` (dummies *and* the numeric term) across a wide
regularisation range. Experiment 02 used alphas ≤ 10, which are negligible
against a 415-column design and 48k rows, so the scan here goes to 3000.

---

## exp05 — Feature family ablation

*(in progress)*
