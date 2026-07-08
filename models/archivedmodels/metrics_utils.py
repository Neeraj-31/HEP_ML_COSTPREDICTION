"""
metrics_utils.py
==================
Shared code imported by every train_<model>.py script in this folder,
so all models are scored with the exact same 22-metric battery and the
exact same leakage-free evaluation protocol (feature selection + target
transform refit INSIDE every LOOCV fold via sklearn Pipeline).

Why this matters for n=50, p=48
---------------------------------
With 48 candidate features and 50 rows, doing feature selection once
on the whole dataset and then cross-validating leaks information and
inflates every score. Every function here that touches the data is
designed to be wrapped inside an sklearn Pipeline / TransformedTargetRegressor
so cross_val_predict(..., cv=LeaveOneOut()) refits imputation, scaling,
feature selection, and the target power-transform separately per fold.

Target transform: cost_overrun_pct is heavily right-skewed (skew~3.8)
and includes negative values, so a Yeo-Johnson PowerTransformer is used
(handles negatives; log1p does not). All metrics are computed on
inverse-transformed predictions, i.e. in the original % units.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score, max_error,
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

TARGET = "cost_overrun_pct"
ID_COL = "project_name"
RANDOM_STATE = 42


class PermutationImportanceSelector(BaseEstimator, TransformerMixin):
    """Robust, model-agnostic feature selection using permutation importance.

    This is more stable than univariate correlation-based selectors on mixed
    tabular data and works well for tree-based regressors such as Random Forest,
    XGBoost, and HistGradientBoosting.
    """

    def __init__(self, k=10, n_repeats=10, random_state=RANDOM_STATE):
        self.k = k
        self.n_repeats = n_repeats
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        self.k_ = min(max(1, int(self.k)), self.n_features_in_)

        surrogate = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
        )
        surrogate.fit(X, y)

        importances = permutation_importance(
            surrogate,
            X,
            y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        self.importances_ = importances.importances_mean

        top_idx = np.argsort(self.importances_)[::-1][: self.k_]
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[top_idx] = True
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.support_]

    def get_support(self, indices=False):
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
def load_data(input_csv):
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=[TARGET])
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET)]
    X = df[feature_cols]
    y = df[TARGET].values
    return df, X, y, feature_cols


# ----------------------------------------------------------------------
# PIPELINE BUILDER -- every model script calls this with its own
# sklearn estimator instance. Feature selection (k best by F-stat) and
# the Yeo-Johnson target transform are identical across models so
# results are comparable.
# ----------------------------------------------------------------------
def build_pipeline(model, k, n_features_available):
    k = min(k, n_features_available)
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", PermutationImportanceSelector(k=k, n_repeats=10, random_state=RANDOM_STATE)),
        ("model", model),
    ])
    return TransformedTargetRegressor(regressor=pipe, transformer=PowerTransformer(method="yeo-johnson"))


def run_loocv(estimator, X, y):
    """Leakage-free LOOCV: the whole pipeline (impute/scale/select/
    target-transform/fit) is refit on each of the n-1 training folds."""
    preds = cross_val_predict(estimator, X, y, cv=LeaveOneOut(), n_jobs=-1)
    return preds


def feature_selection_stability(X, y, k):
    """For each LOOCV fold, refit the selector on the training portion
    only, and count how often each feature survives. High % = robust
    signal, not a one-fold fluke."""
    from collections import Counter
    counts = Counter()
    n = len(y)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr = X.iloc[mask]
        X_tr_imp = SimpleImputer(strategy="median").fit_transform(X_tr)
        X_tr_scaled = StandardScaler().fit_transform(X_tr_imp)
        selector = PermutationImportanceSelector(k=k, n_repeats=8, random_state=RANDOM_STATE)
        selector.fit(X_tr_scaled, y[mask])
        for feat in np.array(X.columns)[selector.get_support()]:
            counts[feat] += 1
    return pd.Series(counts).sort_values(ascending=False) / n * 100


# ----------------------------------------------------------------------
# METRICS (22 total: association / error / bias / efficiency-agreement)
# ----------------------------------------------------------------------
def willmott_d(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    obs_mean = y_true.mean()
    num = np.sum((y_pred - y_true) ** 2)
    den = np.sum((np.abs(y_pred - obs_mean) + np.abs(y_true - obs_mean)) ** 2)
    return 1 - num / den if den > 0 else np.nan


def refined_ioa(y_true, y_pred):
    """Refined Index of Agreement (Willmott et al. 2012)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    obs_mean = y_true.mean()
    num = np.sum(np.abs(y_pred - y_true))
    den = np.sum(np.abs(y_true - obs_mean))
    if den == 0:
        return np.nan
    return 1 - num / (2 * den) if num <= 2 * den else (2 * den) / num - 1


def modified_kge(y_true, y_pred, eps=1e-9):
    """Modified Kling-Gupta Efficiency: correlation + variability ratio + bias ratio."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.std() == 0 or y_true.mean() == 0:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    cv_obs = y_true.std() / (y_true.mean() + eps)
    cv_sim = y_pred.std() / (y_pred.mean() + eps)
    cv_f = cv_sim / (cv_obs + eps)
    gamma = y_pred.mean() / (y_true.mean() + eps)
    return 1 - np.sqrt((r - 1) ** 2 + (cv_f - 1) ** 2 + (gamma - 1) ** 2)


def theils_tci(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    den = np.sqrt(np.sum(y_true ** 2))
    return np.sqrt(np.sum((y_pred - y_true) ** 2)) / den if den > 0 else np.nan


def pbias(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    den = np.sum(y_true)
    return 100 * np.sum(y_true - y_pred) / den if den != 0 else np.nan


def nmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    den = y_true.std() * y_pred.std()
    return np.mean((y_true - y_pred) ** 2) / den if den > 0 else np.nan


def modified_taylor_skill(y_true, y_pred, r=None):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if r is None:
        r = np.corrcoef(y_true, y_pred)[0, 1]
    sx, sy = y_true.std(), y_pred.std()
    if sx == 0 or sy == 0:
        return np.nan
    return (1 + r) ** 4 / (4 * (sx / sy + sy / sx) ** 2)


def aic_regression(y_true, y_pred, n_params):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    if rss <= 0 or n == 0:
        return np.nan
    return n * np.log(rss / n) + 2 * (n_params + 1)


def nse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def mape(y_true, y_pred, eps=1e-6):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    den = np.where(np.abs(y_true) < eps, np.nan, y_true)
    return np.nanmean(np.abs((y_true - y_pred) / den)) * 100


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    den = np.abs(y_true) + np.abs(y_pred)
    den = np.where(den == 0, np.nan, den)
    return np.nanmean(2 * np.abs(y_pred - y_true) / den) * 100


METRIC_ORDER = [
    "n", "pearson_r", "spearman_rho", "R2", , "Explained_Variance",
    "RMSE", "NRMSE_pct_of_range", "MAE", "MAPE_pct", "SMAPE_pct", "Max_Error",
    "TCI_Theil", "Bias_MBE", "PBIAS_pct", "Std_of_residuals",
    "NSE", "Willmott_d", "RIoA", "mKGE", "MTSS", "NMSE", "AIC",
]


def compute_metrics(y_true, y_pred, n_features_used):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    n = len(y_true)
    residuals = y_pred - y_true
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson_r = np.corrcoef(y_true, y_pred)[0, 1] if n > 1 else np.nan
    try:
        spearman_rho, _ = spearmanr(y_true, y_pred)
    except Exception:
        spearman_rho = np.nan
    if n - n_features_used - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features_used - 1)
    else:
        adj_r2 = np.nan
    y_range = y_true.max() - y_true.min()
    return {
        "n": n,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "R2": r2,
        "Adjusted_R2": adj_r2,
        "Explained_Variance": explained_variance_score(y_true, y_pred),
        "RMSE": rmse,
        "NRMSE_pct_of_range": rmse / y_range * 100 if y_range > 0 else np.nan,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE_pct": mape(y_true, y_pred),
        "SMAPE_pct": smape(y_true, y_pred),
        "Max_Error": max_error(y_true, y_pred),
        "TCI_Theil": theils_tci(y_true, y_pred),
        "Bias_MBE": residuals.mean(),
        "PBIAS_pct": pbias(y_true, y_pred),
        "Std_of_residuals": residuals.std(ddof=1) if n > 1 else np.nan,
        "NSE": nse(y_true, y_pred),
        "Willmott_d": willmott_d(y_true, y_pred),
        "RIoA": refined_ioa(y_true, y_pred),
        "mKGE": modified_kge(y_true, y_pred),
        "MTSS": modified_taylor_skill(y_true, y_pred, r=pearson_r),
        "NMSE": nmse(y_true, y_pred),
        "AIC": aic_regression(y_true, y_pred, n_features_used),
    }


# ----------------------------------------------------------------------
# REPORTING / PLOTS
# ----------------------------------------------------------------------
def print_summary_block(metrics_df, model_name, k):
    print("=" * 72)
    print(f"HEADLINE RESULT -- Leave-One-Out CV -- model = {model_name} (k={k} features)")
    print("=" * 72)
    loo = metrics_df["LOOCV"]
    for key in ["n", "R2", "NSE", "pearson_r", "spearman_rho", "RMSE", "MAE",
                "Bias_MBE", "PBIAS_pct", "mKGE", "RIoA"]:
        val = loo[key]
        print(f"  {key:>20s} : {val:.4f}" if pd.notna(val) else f"  {key:>20s} : NA")
    print("-" * 72)
    print(f"  Train R2={metrics_df['train_80pct']['R2']:.3f}  |  "
          f"Test R2={metrics_df['test_20pct']['R2']:.3f}  |  "
          f"LOOCV R2={loo['R2']:.3f}")
    print("=" * 72)
    print("\nFull metrics table (22 statistics x 3 evaluation splits):")
    print(metrics_df.loc[METRIC_ORDER].round(4))
    print()


def plot_predicted_vs_actual(splits, out_path, model_name):
    n = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (label, (y_true, y_pred)) in zip(axes, splits.items()):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        ax.scatter(y_true, y_pred, alpha=0.75, s=45, color="#C44E52", edgecolor="white")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        pad = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - pad, lims[1] + pad]
        ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ax.set_xlabel("Actual cost overrun (%)")
        ax.set_ylabel("Predicted cost overrun (%)")
        ax.set_title(f"{label}\nR2={r2:.3f}, RMSE={rmse:.1f}")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.legend(fontsize=8)
    fig.suptitle(model_name, y=1.03, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def run_full_evaluation(model, model_name, k, input_csv, out_dir):
    """Standard workflow every train_<model>.py script runs:
    LOOCV (leakage-free) + 80/20 split + metrics + plots + CSVs.
    Returns the metrics_df for use by compare_models.py."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    df, X, y, feature_cols = load_data(input_csv)
    n_feat_avail = X.shape[1]
    print(f"Loaded {len(df)} rows, {n_feat_avail} candidate features. Target: {TARGET}")
    print(f"Target skew={pd.Series(y).skew():.2f}, range=[{y.min():.1f}, {y.max():.1f}]\n")

    # ---- Leakage-free LOOCV ----
    loo_est = build_pipeline(model, k, n_feat_avail)
    loo_preds = run_loocv(loo_est, X, y)
    loo_metrics = compute_metrics(y, loo_preds, k)

    # ---- 80/20 split (fit only on train) ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    split_est = build_pipeline(model, k, n_feat_avail)
    split_est.fit(X_train, y_train)
    y_train_pred = split_est.predict(X_train)
    y_test_pred = split_est.predict(X_test)
    train_metrics = compute_metrics(y_train, y_train_pred, k)
    test_metrics = compute_metrics(y_test, y_test_pred, k)

    metrics_df = pd.DataFrame({"train_80pct": train_metrics, "test_20pct": test_metrics, "LOOCV": loo_metrics})
    print_summary_block(metrics_df, model_name, k)
    metrics_df.loc[METRIC_ORDER].to_csv(f"{out_dir}/model_performance_metrics.csv")

    pd.DataFrame({
        ID_COL: df[ID_COL].values if ID_COL in df.columns else np.arange(len(y)),
        "actual": y, "loocv_predicted": loo_preds, "residual": loo_preds - y,
    }).to_csv(f"{out_dir}/loocv_predictions.csv", index=False)

    stability = feature_selection_stability(X, y, k)
    stability.to_csv(f"{out_dir}/feature_selection_stability.csv", header=["pct_of_LOOCV_folds_selected"])
    print("Feature selection stability across LOOCV folds (top 10):")
    print(stability.head(10).round(1))
    print()

    plot_predicted_vs_actual(
        {"Train (80%)": (y_train, y_train_pred),
         "Test (20%)": (y_test, y_test_pred),
         "Leave-One-Out CV": (y, loo_preds)},
        f"{out_dir}/predicted_vs_actual.png", model_name,
    )

    print(f"Done. Outputs -> {out_dir}/")
    print("  - model_performance_metrics.csv   (22 metrics x train/test/LOOCV)")
    print("  - loocv_predictions.csv")
    print("  - feature_selection_stability.csv")
    print("  - predicted_vs_actual.png")

    return metrics_df, (df, X, y, feature_cols, n_feat_avail, split_est, y_train, y_test, y_train_pred, y_test_pred, loo_preds)
