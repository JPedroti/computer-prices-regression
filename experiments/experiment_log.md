# Experiment log

Narrative record of the investigation. Quantitative results for every run live in
`results.csv`; this file records the *hypotheses*, what the numbers meant, and
what was decided as a consequence.

**Validation protocol used throughout**

* **Screening** — 5-fold CV, seed 42, on the development split (64,000 rows).
  Every experiment uses the same folds, so model-to-model comparisons are
  *paired*. This matters here: the fold-to-fold spread of the RMSE is about ±15,
  because a handful of extreme prices dominate the squared error, while paired
  differences between models on shared folds have a standard error near 0.1.
  Absolute RMSEs from different splits are therefore **not** comparable; paired
  differences are.
* **Robustness** — finalists re-scored over three fold partitions (15 folds),
  with the seed driving both the partition and the model's own randomness.
* **Holdout** — 16,000 rows, split off once with seed 42 and never used to fit a
  model. It is where the official bootstrap rule is applied to the final model.
  No model, hyperparameter, feature family or blend weight was chosen by
  comparing scores on it, but it was not read only once — see *How the holdout
  was actually used* at the end of this log.

**How to read `results.csv`.** Sorting it by `validation_rmse` gives a rough
leaderboard, but the rows are not all measured the same way — always read
`validation_strategy` alongside. Most rows are the mean of 5 fold RMSEs under
the screening protocol; exp09's rows are means over 15 folds; and exp07's blend
rows are the RMSE of the pooled out-of-fold vector, which is a different
statistic again (see exp07 for why that distinction cost an experiment).
Comparisons within a strategy are meaningful; across strategies they differ by a
few tenths for reasons that have nothing to do with model quality.

---

## Phase 0 — Dataset audit

80,000 rows, 34 columns, target `price`. No missing values and no duplicates
(neither full-row, nor ignoring `ID`, nor on features alone). Every row is a
unique combination of features, so the irreducible noise cannot be estimated by
comparing repeated specifications; it had to be inferred from model behaviour
(exp03).

| Finding | Evidence | Consequence |
|---|---|---|
| `ID` is a pure identifier | Spearman vs price 0.003; flat mean price across ID deciles | Dropped |
| Development data is a subset of a larger ID range | `ID` unique within 0–99,999, 80,000 present | Treated as a **hypothesis about the data only**; no modelling choice depends on it |
| `model` = brand + line + random code | Line has between-group variance ratio 93; the 3-character code has ratio 0.99 over 38,215 values | Keep the line, drop the code |
| `cpu_model` = family + random number | Family ratio 2345; the trailing number's ratio is 0.54 *conditioned on family* | Keep the family, drop the number |
| Structural zeros | `battery_wh`/`charger_watts` are 0 for exactly the 32,127 Desktops; `psu_watts` for exactly the 47,873 Laptops | Encoded as not-applicable plus applicability flags |
| Target granularity | All 80,000 prices end in `.99` | Snapping predictions would change RMSE by ~0.0002; not done |

The leakage audit came back clean: no feature carries target information
directly or indirectly.

---

## exp01 — Baselines and one pass over every model family

**Hypothesis.** The audit suggested a nearly additive process, so linear models
should be competitive with gradient boosting rather than far behind it.

| Model | valRMSE | train/val gap |
|---|---|---|
| mean baseline | 578.06 | 0.00 |
| median baseline | 581.66 | −0.02 |
| Linear regression (one-hot) | 217.25 | +0.29 |
| Ridge one-hot, alpha=10 | 217.24 | +0.28 |
| Lasso, alpha=1 | 218.20 | −0.09 |
| ElasticNet, alpha=1 | 274.97 | +0.02 |
| Decision tree, depth 12 | 262.04 | +14.06 |
| Random forest, 200 trees | 235.70 | +66.82 |
| Extra trees, 200 trees | 235.96 | +83.44 |
| HistGradientBoosting | 214.68 | +15.48 |
| LightGBM, 400 trees | 214.89 | +32.52 |
| XGBoost, 400 trees | 217.61 | +57.92 |
| **CatBoost, 600 iterations** | **211.38** | +5.72 |

**Result.** Confirmed. The *linear* model is within 2.6 points of the best
tree ensemble while having essentially **no train/validation gap**; the boosters
buy their small advantage with gaps of 15 to 58 points, which is exactly what
the overfitting criterion penalises. CatBoost is the exception and the strongest
single model out of the box — symmetric trees and ordered target statistics
regularise it far better than the others.

The bagged forests are far behind, which is what one expects when the truth is
additive: they fit deep interactions that do not exist.

**Target transforms** all lost ground: LightGBM on log 214.06, Ridge `levels` on
log 220.47, Ridge one-hot on log 222.72. See exp03 for why a log fit loses even
though the noise really is multiplicative.

**Two practical notes.** Lasso and ElasticNet were first run with alpha=0.1;
since alphas act on the price scale (~2000) that is nearly unpenalised and
coordinate descent would not converge in reasonable time. And the CatBoost
wrapper was not clonable by scikit-learn — its constructor modified the stored
parameter, which `clone` checks by identity — so the first sweep aborted there
and was completed in exp01b once fixed.

---

## exp02 — What shape is the generating process?

**Hypothesis.** If price is a sum of component contributions, a *saturated
additive* model — one dummy per observed level of every feature, no interactions
— should approach the achievable floor, and whatever a booster adds on top is
interaction structure.

Results on one 48k/16k split (absolute numbers not comparable with CV figures;
the comparisons *within* this block share the split and are valid):

| Model | valRMSE |
|---|---|
| Saturated additive Ridge | **225.80** |
| LightGBM, 1500 trees, 63 leaves | 232.82 |
| Log-space (multiplicative) Ridge, Duan smearing | 233.13 |
| Saturated additive + LightGBM on its residual | 230.92 |

**Three conclusions.**

1. **Additive, not multiplicative** in the sense that matters for fitting: the
   log-space model is 7 points worse.
2. **Gradient boosting is not the right tool here** — the additive model beats a
   large LightGBM by 7 points on identical data.
3. **No exploitable interactions.** Boosting the additive model's residual made
   it *worse*, and the residual gain importances were flat and diffuse (7.8%,
   7.3%, 6.1%, 6.0%, …) rather than dominated by a few features. That is the
   signature of noise, not of structure left on the table.

**Decision.** Stop trying to out-muscle the problem with model capacity; the
remaining question is how best to *specify* the additive model.

---

## exp03 — The error tail and the noise floor

**The tail bias was an artefact.** The audit showed a +162 bias in the top decile
of the **true** price, which looks like systematic under-prediction of expensive
machines. It is not. If `y = f(x) + noise`, selecting rows with a high `y` also
selects rows with a large positive noise draw, so even a perfect predictor of
`E[y|x]` shows that bias. Judged on deciles of the **prediction**, which is the
correct calibration check, the bias ranges from −7.9 to +11.9 across all ten
deciles: the model is well calibrated and there is no tail correction to make.

| decile | bias by *true* price | bias by *predicted* price |
|---|---|---|
| 0 (cheapest) | −75.7 | −0.2 |
| 9 (most expensive) | **+161.8** | **−7.9** |

**The noise is proportional and extremely heavy-tailed.** Fitting the residual
spread against the price level, a proportional model (sd ≈ 11% of price) fits far
better than a constant one (SSE 7,244 vs 45,490). The relative residual has skew
6.9, the absolute residual skew 8.7.

**Why a log-space fit still loses.** Multiplicative noise does *not* imply that
fitting in log space minimises RMSE. Exponentiating a log-space fit returns a
conditional *median*; with a strong right tail the mean sits well above it, so
the predictions are systematically low. A constant Duan smearing factor cannot
repair that, because the gap varies with the price level. Fitting in price space
with squared error targets the conditional mean directly, which is exactly what
RMSE rewards. This is consistent with exp01's result that every log variant lost
6–9 points.

**The extreme rows are noise, not missing structure.** The worst 16 rows (0.1%)
carry 38% of all squared error, but they belong to no segment — lifts are only
2.9× for Apple and 1.8× for macOS — and make no sense as a group: one is an i5
with 16 GB of RAM priced at 5,031.99 against a prediction of 1,545.

**The model is at the noise floor.** With the best additive model, the RMSE
implied by the fitted proportional noise alone (225.86 on that split) matches the
model's actual RMSE (225.69). The remaining error is consistent with pure label
noise.

**Decision.** No tail-specific treatment, no weighting, no segmentation, no
calibration layer. None of them can help against label noise.

---

## exp04 — The best additive specification

**Hypothesis.** The one-hot Ridge enters each numeric column as a single linear
term, which cannot represent a non-linear price response to, say, `ram_gb`.
Expanding every low-cardinality column into one dummy per level should recover
that shape while staying additive.

| Representation | best alpha | valRMSE | gap |
|---|---|---|---|
| `onehot` (numerics linear) | 10 | 217.24 | +0.28 |
| `levels` (dummy per level) | 10 | 211.99 | +1.07 |
| **`levels_plus`** (dummies + numeric term) | **100** | **211.92** | **+0.90** |

**Result.** The saturated representation is worth **5.3 RMSE points** over
linear-in-numerics, and beats every gradient booster except CatBoost while
keeping the train/validation gap under one point.

Note on why exp02's saturated model looked *worse* (225.8) than exp01's one-hot
Ridge (217.2): those were different splits, and the fold spread is ±15. Under
shared folds the ordering reverses. This is the reason every comparison in this
project is paired.

Regularisation is flat between alpha 1 and 100 and only hurts past 300, which
confirms exp02's alpha ≤ 10 scan was simply too narrow to matter either way.

`levels` and `levels_plus` sit within 0.07 of each other — far inside the noise —
so the choice between them is settled on robustness grounds in exp09.

---

## exp05 — Feature family ablation

Every family was added to or removed from the winning specification and scored on
shared folds.

| Variant | valRMSE | paired diff vs base |
|---|---|---|
| −resolution | 211.868 | −0.052 |
| −cpu_decomp | 211.894 | −0.026 |
| −model_decomp | 211.900 | −0.020 |
| **base** | **211.920** | 0.000 |
| −structural_zeros | 211.921 | +0.001 |
| +premium | 211.927 | +0.007 |
| +cpu_generation | 211.958 | +0.038 |
| +capacity | 211.991 | +0.071 |
| +capacity+premium | 212.022 | +0.102 |
| all families on | 212.059 | +0.139 |

**Result: no family moves the RMSE by more than 0.14 points**, against a fold
spread of ±15. Under the saturated representation every engineered feature turns
out to be redundant:

* `resolution` decomposition — the raw column has only 6 levels, so one dummy per
  level already encodes width, height, pixels and aspect exactly;
* `structural_zeros` — a structural zero is simply its own level;
* `cpu_decomp` — the CPU family is fully determined by the brand, tier, cores,
  threads and clock columns already present;
* `model_decomp` — likewise for the product line, given brand and form factor.

Worth stating plainly: **the feature engineering was not what produced the gain
over the boosters — the saturated representation was.** The engineered columns
are kept because they cost nothing measurable and make SHAP output easier to
read, not because they were shown to help. The candidate families that did *not*
pay their way (`capacity`, `premium`, `cpu_generation`) are left switched off.

---

## exp06 — Interactions, tested directly

exp02 concluded "no interactions" indirectly, from a booster failing on the
residual. That test can fail for the wrong reason, so the question was asked
directly: append a concrete interaction term to the winning additive design and
see whether the cross-validated RMSE falls. A linear model cannot overfit its way
into a paired improvement across shared folds.

Ten pairs chosen from domain structure — a brand premium applied to components, a
form-factor dependent GPU price, laptop-vs-desktop treatment of shared parts.

| Interaction added | valRMSE | paired diff | verdict |
|---|---|---|---|
| brand × cpu_family | 211.789 | −0.131 ±0.030 | detectable |
| brand × device_type | 211.826 | −0.094 ±0.031 | detectable |
| device_type × cpu_family | 211.826 | −0.094 ±0.025 | detectable |
| device_type × gpu_model | 211.843 | −0.077 ±0.033 | detectable |
| *(none — additive only)* | *211.920* | — | — |
| cpu_family × gpu_model | 212.060 | +0.140 ±0.027 | hurts |
| **all ten pairs together** | **212.275** | **+0.355 ±0.116** | **hurts** |

Four pairs are *statistically* detectable, because paired folds make the standard
error about 0.03. But the largest effect is **0.13 RMSE points — 0.06%** — and
adding all ten together makes the model *worse* (+0.355) while tripling the train
gap. Statistical detectability and practical value are not the same thing here,
and nothing is adopted on the strength of a 0.06% move.

**exp06b — combining only the four that helped.** Adding all ten is not a fair
test of combination, since it includes pairs that hurt on their own. Combining
only the four with a negative effect gives **−0.237 ±0.032**, against a sum of
individual effects of −0.396: they overlap, so they do not add up.

Not adopted, for three reasons. The gain is 0.11% of RMSE. Those four pairs were
*selected* by their measured effect on these very folds, so −0.237 is an
optimistic estimate of a real effect. And the train gap rises from +0.90 to
+1.34, spending overfitting margin — the constraint actually worth protecting —
to buy a fifth of an RMSE point.

**Conclusion.** Three independent lines of evidence — residual boosting (exp02),
explicit interaction terms (exp06) and interaction-constrained boosting (exp08) —
all say the same thing: the process is additive and there is no interaction
structure to exploit.

---

## exp08 — Boosting constrained to be additive

**Hypothesis.** The Ridge and the boosters differ in something besides
interactions: the Ridge estimates each level's effect independently, while a
booster shrinks neighbouring levels together through stagewise fitting. A booster
*forced* to be additive might get the best of both.

| Model | valRMSE | gap | paired diff vs Ridge |
|---|---|---|---|
| **Ridge, `levels_plus`, alpha=100** | **211.92** | +0.90 | — |
| stumps (depth 1), lr 0.05, 5000 trees | 212.40 | +0.47 | +0.48 ±0.10 |
| constrained, 63 leaves, 2000 trees | 212.66 | +0.71 | +0.74 ±0.13 |
| constrained, 31 leaves, 3000 trees | 212.74 | +0.67 | +0.82 ±0.14 |
| stumps, lr 0.02, 8000 trees | 213.00 | +0.37 | +1.08 ±0.13 |
| stumps, lr 0.05, 2000 trees | 214.21 | +0.25 | +2.29 ±0.18 |
| unconstrained LightGBM, 31 leaves | 217.21 | +67.67 | +5.29 ±0.40 |

**Hypothesis not confirmed.** Every additive booster lands behind the additive
Ridge. The reason is visible in the data: with 51k training rows and at most 49
levels in any column, even the rarest level has hundreds of observations, so
there is nothing for shrinkage to rescue, and the shrinkage costs a little bias.

The valuable part is the confirmation from a third direction: **forcing a booster
to be additive improves it by 5.3 points** (217.21 → 212.40 at 31 leaves). The
boosters' deficit really was interaction capacity spent on noise.

---

## exp10 — CatBoost tuning

CatBoost was the only booster to beat the additive Ridge, so it was worth tuning
— but along the directions the additive finding predicts should matter, not as a
blind grid. Each configuration costs about six minutes, so the grid is eight
entries (CLAUDE.md section 31).

| Configuration | valRMSE | gap | paired diff vs reference |
|---|---|---|---|
| **depth 6, 1500 iters, lr 0.03** | **211.190** | +6.76 | **−0.193 ±0.031** |
| depth 4, 1000 iters, lr 0.06 | 211.327 | +3.96 | −0.055 ±0.110 |
| depth 6, 600 iters, lr 0.06 *(reference)* | 211.382 | +5.72 | — |
| depth 5, 800 iters | 211.461 | +5.06 | +0.078 ±0.074 |
| depth 4, 2000 iters, lr 0.03, l2 = 10 | 211.462 | +2.57 | +0.080 ±0.110 |
| depth 6, l2 = 10 | 211.604 | +3.88 | +0.221 ±0.093 |
| depth 6, one_hot_max_size = 64 | 211.615 | **+16.72** | +0.233 ±0.145 |
| depth 6, l2 = 30 | 211.862 | +2.80 | +0.479 ±0.163 |

Two things are worth reading off this table.

**Depth barely matters, which is the additive result again.** Going from depth 6
to depth 4 costs nothing measurable (−0.055 ±0.110) while nearly halving the
train gap. A model that could use five-way interactions gains nothing from being
allowed to.

**The ordered target statistics are doing the work.** Forcing plain one-hot
encoding of the categoricals (`one_hot_max_size = 64`) leaves the RMSE almost
unchanged but triples the train gap, from +5.7 to +16.7, and makes the fit 15×
faster. CatBoost's advantage over LightGBM here is its categorical handling and
its symmetric trees, not extra capacity.

Explicit regularisation (`l2_leaf_reg`) shrinks the gap but costs RMSE, so it is
not used: the gap is already comfortable.

**The tuned configuration was evaluated and not adopted.** The −0.193 above is
measured against this experiment's own 600-iteration reference, but the blend's
CatBoost member is the 800-iteration model, which already scores 211.253 on the
same protocol. Against *that*, the tuned configuration is worth −0.063, which at
a blend weight of 0.628 comes to **−0.040 RMSE, or 0.019%** — while nearly
doubling the fit time and forcing every downstream measurement (weights, margin,
robustness) to be redone. The blend keeps the 800-iteration member, which is the
one validated over 15 folds in exp09. The comparison is left in
`exp07_ensembles.py`, commented out, so it can be reproduced.

This is the same judgement applied in exp06b, stated once more: a statistically
real effect is not automatically worth adopting.

---

## exp07 — Ensembles

Out-of-fold predictions were generated once per candidate on the shared folds and
then combined offline.

**A measurement bug worth recording.** The first version of this experiment
compared a blend's *pooled* out-of-fold RMSE against a single model's *mean of
per-fold RMSEs*. Those are different aggregations and differ by a few tenths
here, because fold RMSEs are dominated by how many extreme prices each fold
happens to receive. The comparison made the blend look worse than the best single
model, which is impossible for a non-negative least squares blend fitted on those
same predictions. Everything below is the pooled out-of-fold RMSE.

| Model | pooled OOF RMSE | mean of fold RMSEs |
|---|---|---|
| CatBoost | **211.686** | 211.253 |
| Ridge `levels_plus` | 212.341 | 211.920 |
| HistGradientBoosting | 215.056 | 214.675 |
| LightGBM | 215.494 | 215.114 |
| Ridge `onehot` | 217.623 | 217.240 |

| Blend | pooled OOF RMSE | gain vs best single |
|---|---|---|
| NNLS over all five | 211.308 | +0.378 |
| **NNLS over Ridge + CatBoost** | **211.333** | **+0.353** |
| simple average of Ridge + CatBoost | 211.375 | +0.311 |

**Result.** Blending helps, but only a little, and essentially all of it comes
from combining the additive Ridge with CatBoost: adding the other three members
buys 0.025 RMSE. That is unsurprising given the residual correlations — Ridge and
CatBoost correlate at 0.988, and every pair is above 0.94.

Weights fitted and scored on the same vector are optimistic, so the check was
repeated honestly: fit the weights on one half of the rows, score on the other.
For the two-member blend the gain over CatBoost alone was **+0.42** in one
direction and **+0.27** in the other, with weights of 0.34/0.66 and 0.40/0.60 —
stable, and positive both ways.

---

## exp11 — How much overfitting margin does each candidate have?

The official rule is binary and worth 15 points, so what matters is not only
whether a candidate passes but by how much. A candidate that passes by a hair is
a bad bet, because the margin depends on which rows land in validation.

**The holdout cannot be used for this comparison** — using it to choose
between candidates would make it a selection set and void the final number. The
rule is therefore rehearsed on *inner* splits of the development data (same 80/20
shape, same bootstrap), repeated over three seeds.

| Model | mean gap | mean margin | worst margin | passes |
|---|---|---|---|---|
| **Ridge `levels_plus`** | 5.48 | **+27.74** | **+25.20** | 3/3 |
| Ridge `levels` | 5.63 | +27.63 | +25.05 | 3/3 |
| CatBoost | 11.96 | +20.11 | +17.53 | 3/3 |
| HistGradientBoosting | 21.24 | +9.40 | +7.50 | 3/3 |
| **LightGBM** | 41.55 | **−12.96** | −15.49 | **0/3 — fails** |

Margin = upper end of the train CI minus the lower end of the validation CI;
positive means the intervals overlap.

**This is the sharpest result in the project.** A perfectly ordinary,
well-performing LightGBM — 215.11 RMSE, only 3 points behind the best model —
**fails the overfitting criterion on every split** and would score 0 of those 15
points. HistGradientBoosting passes, but with a margin of 7.5 on its worst split
it is one unlucky partition away from failing.

The additive Ridge passes with roughly three times HistGradientBoosting's
buffer, and it does so *because* it is the right model for this data rather than
by being deliberately hobbled. That is the happy case: the specification that
minimises the error is also the one that satisfies the constraint.

---

## exp12 — What does the train-only protocol cost?

Section 17 requires the final model to be fitted on the training split alone, so
the delivered model sees 64,000 rows instead of all 80,000. A learning curve with
a fixed validation block measures what that costs rather than assuming it away.

The curve is essentially flat at this end: doubling the training data from 16,000
to 32,000 rows buys about 1 RMSE point, and the last increments buy a few
hundredths. Extrapolating `rmse² = a + b/n`, the difference between training on
64,000 and on 80,000 rows is a fraction of a point.

**Decision.** Follow section 17 literally. The protocol costs almost nothing and
is the defensible choice; the alternative — shipping a model trained on all the
data while reporting the rule from a split — buys a rounding error and invites a
fair objection.

---

## exp09 — Robustness across seeds

Each finalist was re-scored over three fold partitions (15 folds in total). The
blend is a fixed-weight combination of its members, so its fold predictions are
the weighted sum of theirs — fitting each member once per fold gives all three
candidates for the cost of two.

| Model | mean RMSE | sd across folds | sd across seed means | worst seed | mean gap |
|---|---|---|---|---|---|
| **blend** | **210.981** | 11.74 | 0.072 | 211.040 | +5.18 |
| CatBoost | 211.335 | 11.69 | 0.076 | 211.403 | +7.52 |
| Ridge `levels_plus` | 211.988 | 11.71 | 0.109 | 212.113 | +0.96 |

**Paired per-fold differences against the blend:**

| Model | difference | folds where it wins |
|---|---|---|
| CatBoost | +0.354 ±0.034 | **0 of 15** |
| Ridge `levels_plus` | +1.007 ±0.048 | **0 of 15** |

The ordering is not a fluke of one partition: the blend is better on **every one
of the fifteen folds**, against both alternatives. The seed-to-seed spread of
each candidate's mean is under 0.11 RMSE, so the ranking is stable in exactly the
sense CLAUDE.md section 15 asks about — the absolute spread across folds is large
(±11.7), but that is the tail lottery, and it cancels in the paired comparison.

---

## exp13 — Choosing the final model

Three candidates survived, and the choice between them is made on the joint
criteria of section 33, not on RMSE alone.

| | Ridge `levels_plus` | CatBoost | **Ridge + CatBoost blend** |
|---|---|---|---|
| pooled OOF RMSE | 212.341 | 211.686 | **211.333** |
| overfitting margin, worst inner split | +25.20 | +17.53 | see below |
| deterministic across refits | yes | yes | yes |
| artefact size | 24 KB | a few MB | a few MB |
| SHAP | exact (linear) | exact (tree) | exact (weighted sum of the two) |

The blend's weights are 0.372 on the Ridge and 0.628 on CatBoost.

Overfitting margins on the same three inner splits used in exp11:

| Candidate | margin per split | mean | worst | passes |
|---|---|---|---|---|
| Ridge `levels_plus` | +25.20, +27.24, +30.76 | +27.74 | +25.20 | 3/3 |
| **Ridge + CatBoost blend** | +20.41, +23.43, +24.92 | **+22.92** | **+20.41** | 3/3 |
| CatBoost | +17.53, +21.17, +21.62 | +20.11 | +17.53 | 3/3 |

**Decision: the two-member blend is the final model.**

It is the best of the three on the metric that carries 55 points, by 1.01 RMSE
over the Ridge and 0.35 over CatBoost, and that gain was confirmed with the
weights fitted on rows they were not scored on, in both directions of the split.
It passes the overfitting rule on every inner split with a margin around 23 —
less buffer than the Ridge alone, but far more than the 7.5 that
HistGradientBoosting would bring and unlike LightGBM, which fails outright. Both
members are deterministic across refits, the artefact is a couple of megabytes,
and SHAP is exact for it: the ensemble is linear in its members, so its SHAP
values are the weighted sum of theirs, which a test verifies reproduces the
prediction to within 1e-6.

The complexity added is two models with fixed weights, and it is paid for by a
measured, out-of-sample gain — which is the standard section 16 asks for.

---

## Final result on the holdout

The blend was fitted on the 64,000 development rows and the official rule
applied to the 16,000 holdout rows (commit `10452c2`). This was not the first
time the holdout's prices were scored — see the next section.

```text
train RMSE      =  206.614   95% CI [196.987, 216.383]
holdout RMSE    =  232.946   95% CI [209.123, 259.959]
overlap = True   margin = +7.26
verdict: no overfitting  ->  15 points
```

train MAE 135.72, holdout MAE 139.82. Artefact: 3.8 MB.

**On that margin, honestly.** The inner-split rehearsals in exp13 put the blend's
margin at +20.4 to +24.9. On the split that actually counts it is +7.26 —
positive, so the rule awards the full 15 points, but at the low end of what the
rehearsals suggested. That is the variation the rehearsals existed to measure,
and it is worth stating rather than glossing: with a heavy-tailed target, how far
the intervals overlap depends on how many extreme prices land in validation.

For reference, the Ridge alone scored 233.795 on the same holdout with a margin
of +11.60, in the first end-to-end run of `src/train.py` (commit `cc5ed3f`). The
blend is 0.85 RMSE better and keeps a smaller but still positive buffer. The
choice between them was made on cross-validation and inner splits, not on these
two numbers, and it is not revisited now on the strength of one number — doing
so would turn the holdout into a selection set.

The holdout RMSE (232.9) is higher than the cross-validated estimate (211.0)
because this particular 16,000-row block drew more of the expensive tail; the
same pattern showed up across the inner splits, where validation RMSE ranged from
211.7 to 219.5 depending on the seed.

---

## How the holdout was actually used

Earlier versions of this log and of the README said the holdout was read only
once. That was wrong. This is the chronology, reconstructed from the commit
history and the development session.

Three kinds of data play different roles and are kept apart below:

| Data | Role |
|---|---|
| CV folds and inner splits of the 64,000 development rows | every choice of model, hyperparameter and blend weight, and the feature-family ablation (exp01–exp13) |
| holdout, the other 16,000 rows | never used to fit a model; scored at the points listed below |
| the mentor's secret test set | never accessed, inspected or reconstructed |

**1. Initial audit, before the split was formalised.** The dataset audit computed
statistics involving the price on all 80,000 rows, including the between-group
variance ratios used to decide which parts of `model` and `cpu_model` to keep.
Its exploratory probes (LightGBM and Ridge fits, and a first bootstrap check of
the overfitting rule) were trained on 80% of the rows and scored on the other 20%
with `train_test_split(test_size=0.2, random_state=42)` — exactly the partition
`src/data.py` later adopted, so the scored rows are the holdout rows. These
observations motivated the hypotheses investigated afterwards, each of which was
then tested on development data. The notebook's EDA likewise describes all
80,000 rows and prints the holdout's mean price.

**2. Pipeline sanity checks during development.** While the code was being built,
ad-hoc scripts outside the repository fitted models on subsets of the development
rows and printed their RMSE on the holdout, to confirm that a code path worked:

| Check | Fitted on | Holdout RMSE |
|---|---|---|
| `levels` / `levels_plus` encoder fix | 20,000 development rows | 234.32 / 234.28 |
| interaction pipeline | 20,000 development rows | 234.29 |
| refit determinism, Ridge / CatBoost | 25,000 development rows | 234.38 / 234.39 |
| ensemble configuration path (Ridge + HistGradientBoosting), learned / fixed weights | 12,000 development rows | 235.43 / 235.79 |

**3. First end-to-end run of `src/train.py`** (commit `cc5ed3f`), with the Ridge
`levels_plus` configuration then in `configs/final_model.json`: holdout RMSE
233.80, margin +11.60. An inference check on that same model reproduced 233.80.

**4. Final run of `src/train.py` with the blend** (commit `10452c2`): holdout RMSE
232.95, margin +7.26. The executed notebook, `experiments/final_check.py` and the
final audit re-evaluate this same fixed model and reproduce those numbers.

**5. Feature rows without prices.** Holdout rows are the unseen inputs for the
SHAP explanations and for the inference-contract tests; one test loads them with
the price column present only to check that it is ignored.

**What this means.** No model, hyperparameter, feature family or blend weight was
selected by comparing scores on the holdout: the Ridge-versus-blend decision
rests on exp07, exp09, exp11 and exp13, and on the holdout the Ridge actually had
the larger margin. But the holdout's prices were seen repeatedly, and the audit
that shaped the early hypotheses used them, so the holdout figures are a check on
a model chosen on development data rather than a strictly blind estimate of
generalisation. The mentor's secret test set was never accessed.
