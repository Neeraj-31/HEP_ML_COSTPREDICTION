# Dam Cost-Overrun Model — Data Analysis & Improvement Report

## 1. Dataset

`dam_ml_ready_cleaned.csv` — **45 historical hydropower/dam projects**, 51 columns
(1 identifier, 1 target, 49 candidate features: geometry, geography, climate
extremes, socio-economic, contract type).

## 2. Target distribution: `cost_overrun_pct`

| Stat | Value |
|---|---|
| Mean | 107.1% |
| Median | 71.6% |
| Std | 118.3% |
| Min | -59.1% (finished under budget) |
| Max | 551.7% |
| Skew | **+1.78** (strongly right-skewed) |

**5 extreme outlier projects** (IQR rule) sit at 300–552% overrun: `nathpa jhakri`,
`larji`, `tiloth`, `chilla`, `maneri_bhali_I`. With only 45 rows, these five points
dominate the loss function of any model trained on raw or naively log-transformed
targets — this is the direct cause of the negative bias (PBIAS ≈ -20%) you saw
earlier: the model hedges toward the bulk of the distribution and systematically
under-predicts the very projects a risk tool most needs to flag.

## 3. Feature signal (correlation with target, full data)

Strongest linear correlates: `cost_per_mw` (-0.60), `CWD_mean` (+0.50, consecutive
wet days), `contract_IR` (+0.41), `contract_EPC` (-0.40), `initial_cost` (-0.36),
`catchment_radius_km` (+0.34), `slope_mean_deg` (-0.31), `steep_frac_25` (-0.30).

RF feature-importance (full-data fit, for guidance only) ranked `cost_per_mw` far
above everything else (0.42), then `initial_cost`, `CWD_mean`, `R30mm_total`,
`landowner_displaced`, `dist_to_nearest_resort_km` — three of which (`CWD_mean`,
`landowner_displaced`, `dist_to_nearest_resort_km`) were **not** in the original
9-feature model, plus `PRCPTOT_max` (extreme annual rainfall). Adding these four
gave a real, LOOCV-verified improvement (below), so they were kept in the new
13-feature set alongside the original 9.

## 4. Honest evaluation methodology

At n=45, a single train/test split is noise, not signal — different splits swing
R² wildly. **Leave-One-Out CV (LOOCV)** is the right protocol here: every project
is held out and predicted from a model trained on the other 44, so the reported
metrics are a genuine estimate of real-world performance.

| Model / feature set | LOOCV RMSE | MAE | Bias | PBIAS | R² |
|---|---|---|---|---|---|
| Original 9 features, RF (as shipped) | 103.2 | 71.0 | -23.3 | -21.7% | 0.22 |
| GradientBoosting, Ridge, Huber, SVR (9 feat.) | 117–124 | 80–90 | -8 to -18 | -7 to -17% | ≤0 |
| **Extended 13 features, RF (raw)** | 99.0 | 68.3 | -27.2 | -25.4% | 0.28 |
| **Extended 13 features, RF + bias correction** | **95.2** | **66.7** | **-0.0** | **-0.0%** | **0.34** |

Random Forest beat every simpler/other model tried (Ridge, Huber, SVR,
GradientBoosting) under honest LOOCV — it wasn't the algorithm that was the
problem, it was the missing bias correction and the narrower feature set.

**Bottom line: R² ≈ 0.3 is the honest ceiling for this dataset today.** That's a
real, useful signal for *relative* risk ranking and contingency sizing, not a
precise forecasting tool — with 45 data points and a target this skewed, no
model will do dramatically better without more data (see recommendations).

## 5. What changed in the rebuilt pipeline

`dam_train.py` (new):
- Extended feature set (13 vs 9), justified above.
- LOOCV used for all reported metrics — no more misleadingly optimistic
  single-split numbers.
- **Bias-correction calibrator**: a linear model of `residual ~ prediction`,
  fit on out-of-fold predictions (not in-sample, so no leakage) and bundled
  with the model. Removes the -25% systematic under-prediction.
- Saves per-tree prediction spread (`avg_tree_log_std`) — a measure of how much
  the model itself disagrees with itself — for use as a model-uncertainty term.

`dam_rf2_v2.py` (new, replaces `dam_rf2.py`):
- **Correlated resampling**: bootstraps whole historical *deviation vectors*
  for the uncertain features (rainfall days, duration, schedule overrun)
  instead of sampling each independently. A wet year historically drives both
  rainfall counts *and* schedule slip together — independent sampling was
  silently understating joint/tail risk. A small Gaussian jitter (smoothed
  bootstrap) avoids only ever replaying 45 discrete historical outcomes.
- **Model uncertainty**: adds noise from the forest's across-tree spread to
  every simulated draw, so the Monte Carlo bands reflect "the model doesn't
  fully know" on top of "the inputs are uncertain" — appropriate given n=45.
- **Bias correction applied** to every simulated draw and the point estimate.
- **Non-interactive `--config config.json` mode** for scripting/batch runs,
  alongside the original interactive prompts.
- Output plot and CSV now show raw vs. bias-corrected distributions side by
  side for transparency.

## 6. Recommendations going forward

1. **More data is the single highest-leverage fix.** Every method above tops
   out around R²≈0.3–0.35 because 45 rows of a right-skewed target can't
   support much more. If more completed-project records (even partial feature
   sets) can be sourced, retrain — expect the biggest gains here.
2. Treat the 5 extreme-overrun projects as a flag for **root-cause review**,
   not just a modeling nuisance — understanding *why* those specific projects
   blew out (litigation? geology surprise? contract type?) may surface a
   feature that explains them structurally rather than statistically.
3. Consider **quantile regression** (e.g. `GradientBoostingRegressor(loss="quantile")`
   or `lightgbm`'s native quantile objective) trained directly on P10/P50/P90 of
   overrun, as a complement to the mean-based RF + Monte Carlo approach —
   quantile models are often more robust with skewed, small-n targets since
   they don't rely on a single symmetric loss function.
4. Re-run `dam_train.py` any time the CSV is updated; the bias correction and
   feature set are re-fit automatically, not hand-tuned constants.
