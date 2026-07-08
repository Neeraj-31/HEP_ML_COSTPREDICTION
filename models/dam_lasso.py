"""
dam_lasso.py
─────────────
Runs FIRST, before dam_rf.py / dam_xgb.py / dam_catboost.py.

Does two jobs:
  1. FEATURE SELECTION — fits LassoCV (L1-regularized regression) on all
     48 candidate features (47 raw + engineered cost_per_mw), using the
     whole cleaned dataset. Features with a non-zero coefficient are kept;
     this is essential here because n=45 rows and p=48 features is a
     1:1 ratio that would badly overfit any of the four models. The
     selected feature list is saved to dam_selected_features.json so
     dam_rf.py, dam_xgb.py, and dam_catboost.py all train on the exact
     same reduced feature set — this keeps the model comparison fair.

     CAVEAT: selection uses the full dataset (not just a training fold),
     so downstream CV/LOOCV metrics are mildly optimistic — a known,
     common simplification for very small-n problems. Worth keeping in
     mind when comparing these numbers to the paper benchmarks.

  2. MODEL EVALUATION — evaluates Lasso itself (on the selected features)
     using the same 75/25, 65/35, LOOCV protocol as the other 3 scripts.

Run: python dam_lasso.py
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler

import dam_common as C

MODEL_TAG = "Lasso"
MAX_SELECTED_FEATURES = 15
MIN_SELECTED_FEATURES = 10

# ── 1. Load data ─────────────────────────────────────────────────────────
df = C.load_cleaned()
candidate_features = C.get_all_candidate_features(df)
X_full, y_full = C.get_X_y(df, candidate_features)
y_log, y_shift = C.make_log_transform(y_full)

print(f"Dataset shape : {X_full.shape}  (n={len(y_full)} samples, "
      f"{len(candidate_features)} candidate features)")
print(f"Target range  : [{y_full.min():.1f}%, {y_full.max():.1f}%]  "
      f"mean={y_full.mean():.1f}%")
print()

# ── 2. Feature selection via LassoCV on the FULL dataset ──────────────────
print(f"{'='*65}")
print("FEATURE SELECTION — LassoCV on all candidate features")
print(f"{'='*65}")

scaler_fs = RobustScaler()
X_full_scaled = scaler_fs.fit_transform(X_full)

lasso_cv = LassoCV(
    cv=5, random_state=C.RANDOM_SEED, max_iter=50000,
    n_alphas=100
)
lasso_cv.fit(X_full_scaled, y_log)
print(f"Selected alpha (LassoCV): {lasso_cv.alpha_:.5f}")

coefs = pd.Series(lasso_cv.coef_, index=candidate_features)
nonzero = coefs[coefs.abs() > 1e-6].sort_values(key=np.abs, ascending=False)

print(f"\nNon-zero coefficients ({len(nonzero)} of {len(candidate_features)} features):")
print(nonzero.to_string())

selected_features = list(nonzero.index)

# Guardrails: keep the selection in a sane range for a 45-row dataset
if len(selected_features) > MAX_SELECTED_FEATURES:
    print(f"\n{len(selected_features)} features selected — capping to top "
          f"{MAX_SELECTED_FEATURES} by |coefficient|")
    selected_features = list(nonzero.index[:MAX_SELECTED_FEATURES])
elif len(selected_features) < MIN_SELECTED_FEATURES:
    print(f"\nOnly {len(selected_features)} features selected — LassoCV's "
          "cross-validated MAE is essentially flat (or worsening) for any "
          "amount of added complexity, meaning no linear combination of "
          "these features clearly beats a near-intercept-only fit. Backing "
          f"off to the top {MIN_SELECTED_FEATURES} features by "
          "|correlation| with the log-target so the tree-based models "
          "(RF/XGBoost/CatBoost) still have something reasonable to work "
          "with — but treat this as weak-signal data, not a strong linear "
          "predictor set.")
    corr = pd.Series(
        [abs(np.corrcoef(X_full[:, i], y_log)[0, 1]) for i in range(X_full.shape[1])],
        index=candidate_features
    ).sort_values(ascending=False)
    selected_features = list(corr.index[:MIN_SELECTED_FEATURES])

print(f"\nFinal selected feature set ({len(selected_features)}):")
for f in selected_features:
    print(f"  - {f}")

with open(C.SELECTED_FEATURES_JSON, "w") as f:
    json.dump({"selected_features": selected_features,
               "lasso_alpha": lasso_cv.alpha_}, f, indent=2)
print(f"\nSaved feature list -> {C.SELECTED_FEATURES_JSON}")

# ── 3. Re-load X restricted to the selected features for model evaluation ─
X_sel, y_full = C.get_X_y(df, selected_features)

LASSO_GRID = {"alpha": np.logspace(-3, 1, 30)}
results = []


def make_search_fn():
    def _fn(X_tr_s, y_tr_log):
        search = GridSearchCV(
            Lasso(random_state=C.RANDOM_SEED, max_iter=50000),
            LASSO_GRID, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        search.fit(X_tr_s, y_tr_log)
        print(f"[Lasso] Best alpha: {search.best_params_['alpha']:.5f}")
        return search.best_estimator_
    return _fn


# ── 4. 75/25 split ──────────────────────────────────────────────────────
X_tr75, X_te25, yl_tr75, yl_te25, y_tr75, y_te25 = train_test_split(
    X_sel, y_log, y_full, test_size=0.25, random_state=C.RANDOM_SEED
)
lasso_75, sc_75, pred_25, m = C.run_split(
    make_search_fn(), X_tr75, X_te25, yl_tr75, y_te25, y_shift, "75/25", MODEL_TAG
)
results.append(m)

# ── 5. 65/35 split ──────────────────────────────────────────────────────
X_tr65, X_te35, yl_tr65, yl_te35, y_tr65, y_te35 = train_test_split(
    X_sel, y_log, y_full, test_size=0.35, random_state=C.RANDOM_SEED
)
lasso_65, sc_65, pred_35, m = C.run_split(
    make_search_fn(), X_tr65, X_te35, yl_tr65, y_te35, y_shift, "65/35", MODEL_TAG
)
results.append(m)

# ── 6. Leave-One-Out CV (reuse best alpha from the 65/35 model) ──────────
best_alpha = lasso_65.alpha


def clone_fn():
    return Lasso(alpha=best_alpha, random_state=C.RANDOM_SEED, max_iter=50000)


y_loo_pred, m = C.run_loo(clone_fn, X_sel, y_log, y_full, y_shift, MODEL_TAG)
results.append(m)

# ── 7. Summary ─────────────────────────────────────────────────────────
df_res = C.summarize(results, "Lasso (all splits)")

# ── 8. Coefficients (on the 65/35-fit model) ──────────────────────────────
coef_65 = pd.Series(lasso_65.coef_, index=selected_features).sort_values(
    key=np.abs, ascending=False)
print("\nLasso coefficients (65/35 model, standardized features, log-target):")
print(coef_65.to_string())

# ── 9. Plots ───────────────────────────────────────────────────────────
C.plot_predictions(y_te25, pred_25, y_te35, pred_35, y_full, y_loo_pred,
                    MODEL_TAG, "dam_lasso_predictions.png")
C.plot_importances(coef_65.abs(), MODEL_TAG, "dam_lasso_coefficients.png",
                    ylabel="|Standardized Coefficient|")

# ── 10. Save model ────────────────────────────────────────────────────────
C.save_model({"model": lasso_65, "scaler": sc_65, "y_shift": y_shift,
              "features": selected_features}, "dam_best_lasso.pkl")

best_row = df_res.iloc[0]
print(f"\nBest Lasso configuration overall: {best_row['label']}  "
      f"(MAPE={best_row['MAPE']:.3f})")
