import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, GridSearchCV, LeaveOneOut
)
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

# ── 0.  Config ──────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
PREPARED     = Path("DA/hep_prepared2.csv")
TARGET       = "pct_cost_overrun"
OUTLIER_CAP  = 99   # percentile cap on target (set None to disable)

np.random.seed(RANDOM_SEED)

# ── 1.  Load & preprocess data ──────────────────────────────────────────────
df = pd.read_csv(PREPARED)
assert df.isnull().sum().sum() == 0, "NaN found — run hep_data_prep.py first"

NECESSARY_COLUMNS = [
    'cost_per_mw', 'geo_prob_yes-fundstop+stresstransition',
    'transmission_km', 'initial_cost', 'geo_prob_yes-landslide',
    'installed_cap_mw', 'tunnel_length_m', 'glof_risk'
]
FEATURES = [c for c in df.columns if c in NECESSARY_COLUMNS]
X_raw = df[FEATURES].copy()
y_raw = df[TARGET].values

# -- Outlier capping on target -----------------------------------------------
if OUTLIER_CAP is not None:
    cap = np.percentile(y_raw, OUTLIER_CAP)
    mask = y_raw <= cap
    X_raw = X_raw[mask]
    y_raw = y_raw[mask]
    print(f"Outlier cap  : removed {(~mask).sum()} samples with target > {cap:.1f}%")

# -- Interaction features ----------------------------------------------------
X_raw = X_raw.copy()
X_raw['cost_x_km']  = X_raw['cost_per_mw'] * X_raw['transmission_km']
X_raw['cost_x_geo'] = X_raw['cost_per_mw'] * X_raw['geo_prob_yes-fundstop+stresstransition']
X_raw['cap_x_tun']  = X_raw['installed_cap_mw'] * X_raw['tunnel_length_m']
ALL_FEATURES = list(X_raw.columns)

X_full = X_raw.values
y_full = y_raw

# -- Log-transform target ----------------------------------------------------
y_shift = y_full.min() - 1.0
y_log   = np.log(y_full - y_shift)

print(f"Dataset shape : {X_full.shape}  (n={len(y_full)} samples, {len(ALL_FEATURES)} features)")
print(f"Target range  : [{y_full.min():.1f}%, {y_full.max():.1f}%]  mean={y_full.mean():.1f}%")
print(f"Log-target range: [{y_log.min():.3f}, {y_log.max():.3f}]")
print()

# ── 2.  Metric helpers (always in original space) ────────────────────────────
def nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / np.mean(np.abs(y_true))

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def inv_log(y_log_pred):
    return np.exp(y_log_pred) + y_shift

def evaluate(y_true, y_pred, label=""):
    m = mape(y_true, y_pred)
    n = nrmse(y_true, y_pred)
    r = r2_score(y_true, y_pred)
    print(f"  {label:<30}  MAPE={m:.3f}  NRMSE={n:.3f}  R2={r:.3f}")
    return {"label": label, "MAPE": m, "NRMSE": n, "R2": r}

# ── 3.  Hyperparameter grid ──────────────────────────────────────────────────
XGB_GRID = {
    "n_estimators"    : [50, 100, 200, 300],
    "max_depth"       : [2, 3, 4, 5],
    "learning_rate"   : [0.01, 0.05, 0.1, 0.2, 0.3],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "subsample"       : [0.6, 0.8, 1.0],
    "reg_alpha"       : [0, 0.1, 1.0],    # L1 regularisation
    "reg_lambda"      : [1.0, 2.0, 5.0],  # L2 regularisation
}

results = []

# ── 4.  Evaluation function for one train/test split ─────────────────────────
def run_split(X_train, X_test, y_train_log, y_test_orig, split_label):
    print(f"\n{'-'*60}")
    print(f"Split: {split_label}  (train={len(y_train_log)}, test={len(y_test_orig)})")
    print(f"{'-'*60}")

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    xgb_base = XGBRegressor(
        random_state=RANDOM_SEED, verbosity=0,
        objective="reg:squarederror", n_jobs=-1
    )
    xgb_cv = GridSearchCV(
        xgb_base, XGB_GRID, cv=3, scoring="neg_mean_absolute_error",
        n_jobs=-1, refit=True
    )
    xgb_cv.fit(X_tr_s, y_train_log)
    best_xgb = xgb_cv.best_estimator_
    print(f"[XGB] Best params: {xgb_cv.best_params_}")

    xgb_pred_log  = best_xgb.predict(X_te_s)
    xgb_pred_orig = inv_log(xgb_pred_log)
    r_xgb = evaluate(y_test_orig, xgb_pred_orig, label=f"XGBoost ({split_label})")
    results.append(r_xgb)

    return best_xgb, scaler, xgb_pred_orig

# ── 5.  75/25 split ──────────────────────────────────────────────────────────
X_tr75, X_te25, yl_tr75, yl_te25, y_tr75, y_te25 = train_test_split(
    X_full, y_log, y_full, test_size=0.25, random_state=RANDOM_SEED
)
xgb_75, sc_75, xgb_pred_25 = run_split(X_tr75, X_te25, yl_tr75, y_te25, "75/25")

# ── 6.  65/35 split ──────────────────────────────────────────────────────────
X_tr65, X_te35, yl_tr65, yl_te35, y_tr65, y_te35 = train_test_split(
    X_full, y_log, y_full, test_size=0.35, random_state=RANDOM_SEED
)
xgb_65, sc_65, xgb_pred_35 = run_split(X_tr65, X_te35, yl_tr65, y_te35, "65/35")

# ── 7.  Leave-One-Out CV ─────────────────────────────────────────────────────
print(f"\n{'-'*60}")
print("Leave-One-Out CV (LOO) — full dataset")
print(f"{'-'*60}")

loo = LeaveOneOut()
y_loo_pred = np.empty_like(y_full, dtype=float)

for train_idx, test_idx in loo.split(X_full):
    scaler_loo = RobustScaler()
    X_tr_s = scaler_loo.fit_transform(X_full[train_idx])
    X_te_s = scaler_loo.transform(X_full[test_idx])

    params = xgb_65.get_params()
    params['n_jobs'] = 1
    m_clone = XGBRegressor(**params)
    m_clone.fit(X_tr_s, y_log[train_idx])
    y_loo_pred[test_idx] = inv_log(m_clone.predict(X_te_s))

r = evaluate(y_full, y_loo_pred, label="XGBoost (LOO)")
results.append(r)

# ── 8.  Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUMMARY — all XGBoost splits")
print(f"{'='*65}")
df_res = pd.DataFrame(results).sort_values("MAPE")
print(df_res.to_string(index=False))
print()
print("Paper benchmark (RS dataset, 35% test):")
print("  AdaBoost     : MAPE=0.189, NRMSE=0.16")
print("  XGBoost paper benchmark range: 0.22–0.31")

# ── 9.  Feature importances ───────────────────────────────────────────────────
importances = pd.Series(xgb_65.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
print("\nTop 15 feature importances (XGBoost 65/35):")
print(importances.head(15).to_string())

# ── 10.  Plots ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("HEP Cost Overrun — XGBoost Predictions", fontsize=14, fontweight="bold")

def scatter(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5, s=60)
    lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual % Cost Overrun"); ax.set_ylabel("Predicted % Cost Overrun")
    ax.set_title(title); ax.legend(fontsize=8)

scatter(axes[0], y_te25, xgb_pred_25, "XGB — 75/25 split")
scatter(axes[1], y_te35, xgb_pred_35, "XGB — 65/35 split")
scatter(axes[2], y_full, y_loo_pred,   "XGB — LOO")
plt.tight_layout()
plt.savefig("hep_xgb_predictions.png", dpi=150, bbox_inches="tight")
print("\nPrediction scatter plots -> hep_xgb_predictions.png")

fig2, ax2 = plt.subplots(figsize=(10, 7))
importances.head(15).sort_values().plot.barh(ax=ax2, color="steelblue", edgecolor="k")
ax2.set_xlabel("Feature Importance (gain)")
ax2.set_title("Top 15 Feature Importances — XGBoost (65/35 split)")
plt.tight_layout()
plt.savefig("hep_xgb_feature_importance.png", dpi=150, bbox_inches="tight")
print("Feature importance chart  -> hep_xgb_feature_importance.png")

# ── 11.  Save best model ──────────────────────────────────────────────────────
best_row = df_res.iloc[0]
print(f"\nBest XGB configuration overall: {best_row['label']}  (MAPE={best_row['MAPE']:.3f})")
with open("hep_best_xgb.pkl", "wb") as f:
    pickle.dump({"model": xgb_65, "scaler": sc_65, "y_shift": y_shift,
                 "features": ALL_FEATURES}, f)
print("Saved: hep_best_xgb.pkl")