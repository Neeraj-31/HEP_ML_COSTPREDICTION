import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error, max_error
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

import dam_common as C

MODEL_TAG = "Random Forest"

# ── 1. Load data + selected features ───────────────────────────────────────
df = C.load_cleaned()
features = C.load_selected_features()
X_full, y_full = C.get_X_y(df, features)
y_log, y_shift = C.make_log_transform(y_full)

print(f"Dataset shape : {X_full.shape}  (n={len(y_full)} samples, "
      f"{len(features)} features)")
print(f"Features      : {features}")
print()

RF_GRID = {
    "n_estimators"     : [100, 200, 300],
    "max_depth"        : [None, 3, 5],
    "min_samples_leaf" : [1, 2, 3],
    "max_features"     : ["sqrt", "log2", 0.5],
}
results = []


def make_search_fn():
    def _fn(X_tr_s, y_tr_log):
        search = GridSearchCV(
            RandomForestRegressor(random_state=C.RANDOM_SEED, n_jobs=1),
            RF_GRID, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1, refit=True
        )
        search.fit(X_tr_s, y_tr_log)
        print(f"[RF] Best params: {search.best_params_}")
        return search.best_estimator_
    return _fn


# ── 2. 75/25 split ──────────────────────────────────────────────────────
X_tr75, X_te25, yl_tr75, yl_te25, y_tr75, y_te25 = train_test_split(
    X_full, y_log, y_full, test_size=0.25, random_state=C.RANDOM_SEED
)
rf_75, sc_75, pred_25, m = C.run_split(
    make_search_fn(), X_tr75, X_te25, yl_tr75, y_te25, y_shift, "75/25", MODEL_TAG
)
results.append(m)

# ── 3. 65/35 split ──────────────────────────────────────────────────────
X_tr65, X_te35, yl_tr65, yl_te35, y_tr65, y_te35 = train_test_split(
    X_full, y_log, y_full, test_size=0.35, random_state=C.RANDOM_SEED
)
rf_65, sc_65, pred_35, m = C.run_split(
    make_search_fn(), X_tr65, X_te35, yl_tr65, y_te35, y_shift, "65/35", MODEL_TAG
)
results.append(m)

# ── 4. Leave-One-Out CV (reuse best params from the 65/35 model) ─────────
best_params = rf_65.get_params()


def clone_fn():
    return RandomForestRegressor(**best_params)


y_loo_pred, m = C.run_loo(clone_fn, X_full, y_log, y_full, y_shift, MODEL_TAG)
results.append(m)

# ── 5. Summary ────────────────────────────────────────────────────────────
df_res = C.summarize(results, "Random Forest (all splits)")

# ── 6. Feature importances (65/35 model) ──────────────────────────────────
importances = pd.Series(rf_65.feature_importances_, index=features).sort_values(
    ascending=False)
print("\nFeature importances (Random Forest 65/35):")
print(importances.to_string())

# ── 7. Plots ───────────────────────────────────────────────────────────
C.plot_predictions(y_te25, pred_25, y_te35, pred_35, y_full, y_loo_pred,
                    MODEL_TAG, "dam_rf_predictions.png")
C.plot_importances(importances, MODEL_TAG, "dam_rf_feature_importance.png",
                    ylabel="Feature Importance (mean impurity decrease)")

# ── 8. Save model ────────────────────────────────────────────────────────
C.save_model({"model": rf_65, "scaler": sc_65, "y_shift": y_shift,
              "features": features}, "dam_best_rf.pkl")

best_row = df_res.iloc[0]
print(f"\nBest RF configuration overall: {best_row['label']}  "
      f"(MAPE={best_row['MAPE']:.3f})")
def compute_extended_metrics(y_true, y_pred, k_features):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_rho, _ = stats.spearmanr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    exp_var = explained_variance_score(y_true, y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nrmse_pct = (rmse / (np.max(y_true) - np.min(y_true))) * 100 if (np.max(y_true) - np.min(y_true)) != 0 else np.nan
    
    mae = mean_absolute_error(y_true, y_pred)
    mape_pct = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape_pct = (100 / n) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    mx_err = max_error(y_true, y_pred)
    
    num = np.sqrt(np.mean((y_true - y_pred)**2))
    den = np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2))
    tci_theil = num / den if den != 0 else np.nan
    
    bias_mbe = np.mean(y_pred - y_true)
    pbias_pct = 100 * np.sum(y_pred - y_true) / np.sum(y_true)
    std_resid = np.std(y_pred - y_true)
    
    var_y = np.sum((y_true - np.mean(y_true))**2)
    nse = 1 - (np.sum((y_true - y_pred)**2) / var_y) if var_y != 0 else np.nan
    
    wd_den = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true)))**2)
    willmott_d = 1 - (np.sum((y_true - y_pred)**2) / wd_den) if wd_den != 0 else np.nan
    
    sum_abs_err = np.sum(np.abs(y_true - y_pred))
    sum_abs_dev = np.sum(np.abs(y_true - np.mean(y_true)))
    if sum_abs_err <= 2 * sum_abs_dev:
        rioa = 1 - (sum_abs_err / (2 * sum_abs_dev)) if sum_abs_dev != 0 else np.nan
    else:
        rioa = (2 * sum_abs_dev / sum_abs_err) - 1 if sum_abs_err != 0 else np.nan
        
    cv_true = np.std(y_true) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    cv_pred = np.std(y_pred) / np.mean(y_pred) if np.mean(y_pred) != 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    gamma = cv_pred / cv_true if cv_true != 0 and not np.isnan(cv_true) else np.nan
    mkge = 1 - np.sqrt((pearson_r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2) if not np.isnan(gamma) else np.nan
    
    mtss = np.mean((y_true - np.mean(y_true))**2)
    nmse = np.sum((y_true - y_pred)**2) / var_y if var_y != 0 else np.nan
    aic = n * np.log(mse) + 2 * k_features if mse > 0 else np.nan

    return {
        "n": n, "pearson_r": pearson_r, "spearman_rho": spearman_rho, "R2": r2,
        "Explained_Variance": exp_var, "RMSE": rmse, "NRMSE_pct_of_range": nrmse_pct,
        "MAE": mae, "MAPE_pct": mape_pct, "SMAPE_pct": smape_pct, "Max_Error": mx_err,
        "TCI_Theil": tci_theil, "Bias_MBE": bias_mbe, "PBIAS_pct": pbias_pct,
        "Std_of_residuals": std_resid, "NSE": nse, "Willmott_d": willmott_d,
        "RIoA": rioa, "mKGE": mkge, "MTSS": mtss, "NMSE": nmse, "AIC": aic
    }

print("\n" + "--"*65)
print(f"Extended Metrics (Leave-One-Out CV) - {MODEL_TAG}")
print("--"*65)
ext_metrics = compute_extended_metrics(y_full, y_loo_pred, len(features))
for k, v in ext_metrics.items():
    if isinstance(v, int):
        print(f"{k+':':<22} {v}")
    else:
        print(f"{k+':':<22} {v:.4f}")
print("--"*65 + "\n")