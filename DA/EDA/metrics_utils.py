"""
metrics_utils.py
=================
One shared, consistently-defined metrics suite used by all three model
scripts (01_ridge_lasso_regression.py, 02_random_forest.py,
03_xgboost.py) so results are directly comparable.

All metrics are computed on the ORIGINAL %-scale predictions (after
inverse-transforming out of log space), from Leave-One-Out CV
predictions (y_true = actual cost_overrun_pct, y_pred = LOOCV
prediction for that held-out project).

Notes on a few less-standard metrics, since definitions vary in the
literature -- these are the conventions used here:

- NRMSE_pct_of_range: RMSE / (max(y_true) - min(y_true)) * 100
- MAPE / SMAPE: with a target that can be ~0 or negative (this one
  ranges from -59% to +3366%), MAPE is numerically unstable near
  zero. It's still reported (for comparability) but SMAPE is the
  more trustworthy of the two here.
- Theil's U (TCI_Theil): Theil's U1, bounded [0,1], 0 = perfect.
  U1 = RMSE / (sqrt(mean(y_true^2)) + sqrt(mean(y_pred^2)))
- PBIAS: percent bias, negative = model over-predicts on average.
- NSE: Nash-Sutcliffe Efficiency (hydrology standard). 1 = perfect,
  0 = no better than predicting the mean, <0 = worse than the mean.
- Willmott_d: original (1981) index of agreement, [0,1].
- RIoA: Refined Index of Agreement (Willmott et al. 2012), [-1,1],
  less sensitive to outliers than d.
- mKGE: modified Kling-Gupta Efficiency (Kling et al. 2012), uses the
  coefficient of variation ratio (gamma) instead of the raw std ratio,
  which is the standard fix for mean values near zero.
- MTSS: Taylor Skill Score (Taylor, 2001), using R0=1 (max attainable
  correlation), based on correlation + relative variability.
- NMSE: mean squared error normalized by variance of y_true.
  (Note: NMSE = 1 - NSE algebraically -- both are reported because
  you asked for both, but they carry the same information.)
- AIC: computed from LOOCV residuals as
  n*ln(RSS/n) + 2k. This is a NON-STANDARD use of AIC (textbook AIC
  assumes in-sample residuals from a single fitted model, not
  out-of-sample LOOCV residuals from n different refits). It's
  included because it was requested, but for RF/XGBoost 'k' has no
  clean definition (trees aren't parametric); we approximate k as the
  number of input features for linear models and as a fixed small
  effective-parameter count for tree ensembles, and flag this loudly
  in the output. Do NOT use these AIC values to justify one model
  class over another -- use LOOCV RMSE / NSE / mKGE for that instead.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, \
    explained_variance_score


def compute_full_metrics(y_true, y_pred, k_params=None, model_name=""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    resid = y_pred - y_true          # positive = over-prediction
    abs_resid = np.abs(resid)
    sq_resid = resid ** 2

    mean_y = y_true.mean()
    var_y = y_true.var(ddof=0)
    range_y = y_true.max() - y_true.min()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    max_err = abs_resid.max()

    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_rho, _ = spearmanr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    expl_var = explained_variance_score(y_true, y_pred)

    nrmse_pct = 100 * rmse / range_y if range_y != 0 else np.nan

    # MAPE / SMAPE -- guard against div-by-zero (target crosses zero here)
    nonzero = y_true != 0
    mape_pct = 100 * np.mean(abs_resid[nonzero] / np.abs(y_true[nonzero])) if nonzero.any() else np.nan
    denom_smape = (np.abs(y_true) + np.abs(y_pred))
    smape_mask = denom_smape != 0
    smape_pct = 100 * np.mean(2 * abs_resid[smape_mask] / denom_smape[smape_mask]) if smape_mask.any() else np.nan

    # Theil's U1
    theil_denom = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    tci_theil = rmse / theil_denom if theil_denom != 0 else np.nan

    bias_mbe = resid.mean()  # mean(pred - actual)
    pbias_pct = 100 * resid.sum() / y_true.sum() if y_true.sum() != 0 else np.nan
    std_resid = resid.std(ddof=1) if n > 1 else np.nan

    # NSE (Nash-Sutcliffe)
    ss_res = sq_resid.sum()
    ss_tot = ((y_true - mean_y) ** 2).sum()
    nse = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # Willmott's original d (1981)
    d_denom = np.sum((np.abs(y_pred - mean_y) + np.abs(y_true - mean_y)) ** 2)
    willmott_d = 1 - ss_res / d_denom if d_denom != 0 else np.nan

    # Refined Index of Agreement (Willmott et al. 2012)
    num_c = np.sum(abs_resid)
    denom_c = 2 * np.sum(np.abs(y_true - mean_y))
    if denom_c != 0:
        rioa = 1 - num_c / denom_c if num_c <= denom_c else denom_c / num_c - 1
    else:
        rioa = np.nan

    # modified KGE (Kling et al. 2012) -- uses CV ratio (gamma), not raw std ratio
    mean_p = y_pred.mean()
    std_y = y_true.std(ddof=0)
    std_p = y_pred.std(ddof=0)
    beta = mean_p / mean_y if mean_y != 0 else np.nan
    cv_y = std_y / mean_y if mean_y != 0 else np.nan
    cv_p = std_p / mean_p if mean_p != 0 else np.nan
    gamma = cv_p / cv_y if (cv_y not in (0, np.nan) and not np.isnan(cv_y)) else np.nan
    if np.isnan(beta) or np.isnan(gamma):
        mkge = np.nan
    else:
        mkge = 1 - np.sqrt((pearson_r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

    # Taylor Skill Score (MTSS), R0 = 1
    sigma_hat = std_p / std_y if std_y != 0 else np.nan
    if np.isnan(sigma_hat) or sigma_hat == 0:
        mtss = np.nan
    else:
        mtss = ((1 + pearson_r) ** 4) / (4 * (sigma_hat + 1 / sigma_hat) ** 2)

    # NMSE (normalized MSE by variance of y_true) -- algebraically = 1 - NSE
    nmse = ss_res / (n * var_y) if var_y != 0 else np.nan

    # AIC (see caveats in module docstring) -- computed from LOOCV residuals
    if k_params is None:
        k_params = 1
    rss = ss_res
    if rss <= 0:
        aic = np.nan
    else:
        aic = n * np.log(rss / n) + 2 * (k_params + 1)  # +1 for estimated residual variance

    return {
        "model": model_name,
        "n": n,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "R2": r2,
        "Explained_Variance": expl_var,
        "RMSE": rmse,
        "NRMSE_pct_of_range": nrmse_pct,
        "MAE": mae,
        "MAPE_pct": mape_pct,
        "SMAPE_pct": smape_pct,
        "Max_Error": max_err,
        "TCI_Theil": tci_theil,
        "Bias_MBE": bias_mbe,
        "PBIAS_pct": pbias_pct,
        "Std_of_residuals": std_resid,
        "NSE": nse,
        "Willmott_d": willmott_d,
        "RIoA": rioa,
        "mKGE": mkge,
        "MTSS": mtss,
        "NMSE": nmse,
        "AIC": aic,
        "AIC_k_used": k_params,
    }


def print_metrics(metrics_dict):
    order = [
        "model", "n", "pearson_r", "spearman_rho", "R2", "Explained_Variance",
        "RMSE", "NRMSE_pct_of_range", "MAE", "MAPE_pct", "SMAPE_pct", "Max_Error",
        "TCI_Theil", "Bias_MBE", "PBIAS_pct", "Std_of_residuals",
        "NSE", "Willmott_d", "RIoA", "mKGE", "MTSS", "NMSE", "AIC", "AIC_k_used",
    ]
    print(f"\n{'='*60}")
    print(f"Full metrics suite: {metrics_dict.get('model','')}")
    print(f"{'='*60}")
    for k in order:
        v = metrics_dict.get(k)
        if isinstance(v, float):
            print(f"  {k:22s} = {v:.4f}")
        else:
            print(f"  {k:22s} = {v}")
