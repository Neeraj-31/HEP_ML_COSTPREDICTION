import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy.stats as stats
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error, max_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

import dam_common as C

MODEL_TAG = "GPR"
Z_95 = 1.96   # for a ~95% predictive interval in log-space

# ── 1. Load data + selected features ───────────────────────────────────────
df = C.load_cleaned()
features = C.load_selected_features()
X_full, y_full = C.get_X_y(df, features)
y_log, y_shift = C.make_log_transform(y_full)

print(f"Dataset shape : {X_full.shape}  (n={len(y_full)} samples, "
      f"{len(features)} features)")
print(f"Features      : {features}")
print()

# ── 2. Kernel candidates + grid ─────────────────────────────────────────────
def make_kernels():
    return [
        ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0),
        ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0),
    ]

GPR_GRID = {
    "kernel": make_kernels(),
    "alpha": [1e-10, 1e-3, 1e-2],
}
results = []


def make_search_fn():
    def _fn(X_tr_s, y_tr_log):
        search = GridSearchCV(
            GaussianProcessRegressor(random_state=C.RANDOM_SEED,
                                      normalize_y=True, n_restarts_optimizer=5),
            GPR_GRID, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1, refit=True
        )
        search.fit(X_tr_s, y_tr_log)
        print(f"[GPR] Best params: alpha={search.best_params_['alpha']}, "
              f"kernel={search.best_estimator_.kernel_}")
        return search.best_estimator_
    return _fn


def predict_with_ci(model, X_s, y_shift):
    """Predict in log-space with std, then map mean/lo/hi into original space."""
    mean_log, std_log = model.predict(X_s, return_std=True)
    lo_log, hi_log = mean_log - Z_95 * std_log, mean_log + Z_95 * std_log
    mean_orig = C.inv_log(mean_log, y_shift)
    lo_orig = C.inv_log(lo_log, y_shift)
    hi_orig = C.inv_log(hi_log, y_shift)
    return mean_orig, lo_orig, hi_orig, std_log


# ── 3. 75/25 split ──────────────────────────────────────────────────────
X_tr75, X_te25, yl_tr75, yl_te25, y_tr75, y_te25 = train_test_split(
    X_full, y_log, y_full, test_size=0.25, random_state=C.RANDOM_SEED
)
gpr_75, sc_75, pred_25, m = C.run_split(
    make_search_fn(), X_tr75, X_te25, yl_tr75, y_te25, y_shift, "75/25", MODEL_TAG
)
results.append(m)
_, lo_25, hi_25, _ = predict_with_ci(gpr_75, sc_75.transform(X_te25), y_shift)

# ── 4. 65/35 split ──────────────────────────────────────────────────────
X_tr65, X_te35, yl_tr65, yl_te35, y_tr65, y_te35 = train_test_split(
    X_full, y_log, y_full, test_size=0.35, random_state=C.RANDOM_SEED
)
gpr_65, sc_65, pred_35, m = C.run_split(
    make_search_fn(), X_tr65, X_te35, yl_tr65, y_te35, y_shift, "65/35", MODEL_TAG
)
results.append(m)
_, lo_35, hi_35, _ = predict_with_ci(gpr_65, sc_65.transform(X_te35), y_shift)

# ── 5. Leave-One-Out CV (reuse best kernel/alpha from the 65/35 model) ────
best_kernel = gpr_65.kernel_
best_alpha = gpr_65.alpha

print(f"\n{'-'*60}")
print(f"Leave-One-Out CV (LOO) — full dataset — {MODEL_TAG}")
print(f"{'-'*60}")

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import RobustScaler

loo = LeaveOneOut()
y_loo_pred = np.empty_like(y_full, dtype=float)
y_loo_lo = np.empty_like(y_full, dtype=float)
y_loo_hi = np.empty_like(y_full, dtype=float)

for train_idx, test_idx in loo.split(X_full):
    scaler_loo = RobustScaler()
    X_tr_s = scaler_loo.fit_transform(X_full[train_idx])
    X_te_s = scaler_loo.transform(X_full[test_idx])

    m_clone = GaussianProcessRegressor(
        kernel=best_kernel, alpha=best_alpha, random_state=C.RANDOM_SEED,
        normalize_y=True, n_restarts_optimizer=2, optimizer=None
        # optimizer=None: reuse the already-optimized 65/35 kernel
        # hyperparameters as-is rather than re-fitting theta on n-1=44
        # points per fold — keeps LOOCV fast and stable for this model.
    )
    m_clone.fit(X_tr_s, y_log[train_idx])
    mean_orig, lo_orig, hi_orig, _ = predict_with_ci(m_clone, X_te_s, y_shift)
    y_loo_pred[test_idx] = mean_orig
    y_loo_lo[test_idx] = lo_orig
    y_loo_hi[test_idx] = hi_orig

m = C.evaluate(y_full, y_loo_pred, label=f"{MODEL_TAG} (LOO)")
results.append(m)

# ── 6. Summary ────────────────────────────────────────────────────────────
df_res = C.summarize(results, "GPR (all splits)")



# ── 8. Plots — predictions with 95% confidence bands ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Dam Cost Overrun — GPR Predictions (with 95% CI)",
              fontsize=14, fontweight="bold")


def scatter_ci(ax, y_true, y_pred, lo, hi, title):
    yerr = np.vstack([np.clip(y_pred - lo, 0, None), np.clip(hi - y_pred, 0, None)])
    ax.errorbar(y_true, y_pred, yerr=yerr, fmt="o", alpha=0.7, ecolor="gray",
                elinewidth=1, capsize=2, markeredgecolor="k", markersize=6)
    lo_ax = min(y_true.min(), y_pred.min()); hi_ax = max(y_true.max(), y_pred.max())
    ax.plot([lo_ax, hi_ax], [lo_ax, hi_ax], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual % Cost Overrun"); ax.set_ylabel("Predicted % Cost Overrun")
    ax.set_title(title); ax.legend(fontsize=8)


scatter_ci(axes[0], y_te25, pred_25, lo_25, hi_25, "GPR — 75/25 split")
scatter_ci(axes[1], y_te35, pred_35, lo_35, hi_35, "GPR — 65/35 split")
scatter_ci(axes[2], y_full, y_loo_pred, y_loo_lo, y_loo_hi, "GPR — LOO")
plt.tight_layout()
plt.savefig("C:\\Users\\User\\.vscode\\HEP_ML\\modelwise_prediction\\dam_gpr_predictions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nPrediction scatter plots (with 95% CI) -> dam_gpr_predictions.png")

# ── 9. Save model ────────────────────────────────────────────────────────
C.save_model({"model": gpr_65, "scaler": sc_65, "y_shift": y_shift,
              "features": features}, "dam_best_gpr.pkl")

best_row = df_res.iloc[0]
print(f"\nBest GPR configuration overall: {best_row['label']}  "
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

print("\n" + "-"*65)
print(f"Extended Metrics (Leave-One-Out CV) - {MODEL_TAG}")
print("-"*65)
ext_metrics = compute_extended_metrics(y_full, y_loo_pred, len(features))
for k, v in ext_metrics.items():
    if isinstance(v, int):
        print(f"{k+':':<22} {v}")
    else:
        print(f"{k+':':<22} {v:.4f}")
print("-"*65 + "\n")