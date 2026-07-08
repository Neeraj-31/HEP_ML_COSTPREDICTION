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
    train_test_split, GridSearchCV, LeaveOneOut, cross_val_score
)
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

# -- 0. Config --------------------------------------------------------------
RANDOM_SEED = 42
PREPARED    = Path("DA\hep_prepared2.csv")
TARGET      = "pct_cost_overrun"

np.random.seed(RANDOM_SEED)

# -- 1. Load prepared data --------------------------------------------------
df = pd.read_csv(PREPARED)
assert df.isnull().sum().sum() == 0, "NaN found — run hep_data_prep.py first"
NECCESARY_COLUMNS=['cost_per_mw','geo_prob_yes-fundstop+stresstransition','transmission_km','initial_cost','geo_prob_yes-landslide','installed_cap_mw','tunnel_length_m','glof_risk']
FEATURES = [c for c in df.columns if c in NECCESARY_COLUMNS]
X = df[FEATURES].values
y = df[TARGET].values

print(f"Dataset shape : {X.shape}  (n={len(y)} samples, {len(FEATURES)} features)")
print(f"Target range  : [{y.min():.1f}%, {y.max():.1f}%]  mean={y.mean():.1f}%")
print()

# -- 2. Metric helpers -------------------------------------------------------
def nrmse(y_true, y_pred):
    """Normalised RMSE = RMSE / mean(y_true)"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / np.mean(np.abs(y_true))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (fractional, not %)."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def evaluate(y_true, y_pred, label=""):
    m = mape(y_true, y_pred)
    n = nrmse(y_true, y_pred)
    r = r2_score(y_true, y_pred)
    print(f"  {label:<30}  MAPE={m:.3f}  NRMSE={n:.3f}  R2={r:.3f}")
    return {"label": label, "MAPE": m, "NRMSE": n, "R2": r}

# -- 3. Hyperparameter grid (mirrors Table 4 of paper) ---------------------
XGB_GRID = {
    "n_estimators"      : [50, 100, 150, 200],
    "max_depth"         : [2, 3, 4, 5],
    "learning_rate"     : [0.05, 0.1, 0.3, 0.5],
    "colsample_bytree"  : [0.5, 0.8, 1.0],
    "subsample"         : [0.7, 0.85, 1.0],
}

results = []

# -- 4. Evaluation function for one train/test split ------------------------
def run_split(X_train, X_test, y_train, y_test, split_label):
    print("\n" + "-" * 60)
    print(f"Split: {split_label}  (train={len(y_train)}, test={len(y_test)})")
    print("-" * 60)

    xgb_base = XGBRegressor(
        random_state=RANDOM_SEED, verbosity=0,
        objective="reg:squarederror", n_jobs=-1
    )
    xgb_cv = GridSearchCV(
        xgb_base, XGB_GRID, cv=3, scoring="neg_mean_absolute_error",
        n_jobs=-1, refit=True
    )
    xgb_cv.fit(X_train, y_train)
    best_xgb = xgb_cv.best_estimator_
    print(f"[XGB] Best params: {xgb_cv.best_params_}")

    xgb_pred = best_xgb.predict(X_test)
    r_xgb = evaluate(y_test, xgb_pred, label=f"XGBoost ({split_label})")
    results.append(r_xgb)

    return best_xgb, xgb_pred

# -- 5. Run 75/25 split (paper's first split) ───────────────────────────────
X_tr75, X_te25, y_tr75, y_te25 = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_SEED
)
xgb_75, xgb_pred_25 = run_split(X_tr75, X_te25, y_tr75, y_te25, split_label="75/25")

# -- 6. Run 65/35 split (paper's second split) ──────────────────────────────
X_tr65, X_te35, y_tr65, y_te35 = train_test_split(
    X, y, test_size=0.35, random_state=RANDOM_SEED
)
xgb_65, xgb_pred_35 = run_split(X_tr65, X_te35, y_tr65, y_te35, split_label="65/35")

# -- 7. Leave-One-Out CV ----------------------------------------------------
print("\n" + "-" * 60)
print("Leave-One-Out CV (LOO) — full dataset")
print("-" * 60)

loo = LeaveOneOut()

y_loo_pred = np.empty_like(y, dtype=float)
for train_idx, test_idx in loo.split(X):
    # Force single threading inside the loop to avoid backend joblib crashing/warnings
    params = xgb_65.get_params()
    params['n_jobs'] = 1
    
    m_clone = XGBRegressor(**params)
    m_clone.fit(X[train_idx], y[train_idx])
    y_loo_pred[test_idx] = m_clone.predict(X[test_idx])
    
r = evaluate(y, y_loo_pred, label="XGBoost (LOO)")
results.append(r)

# -- 8. Summary table --------------------------------------------------------
print("\n" + "=" * 65)
print("SUMMARY — all XGBoost splits")
print("=" * 65)
df_res = pd.DataFrame(results)
df_res = df_res.sort_values("MAPE")
print(df_res.to_string(index=False))
print()
print("Paper benchmark (RS dataset, 35% test):")
print("  AdaBoost     : MAPE=0.189, NRMSE=0.16")
print("  XGBoost paper benchmark range: 0.22–0.31")
print()

# -- 9. Feature importances (best XGB on 65/35 split) -------------------------
importances = pd.Series(xgb_65.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)
print("Top 15 feature importances (XGBoost 65/35):")
print(importances.head(15).to_string())

# -- 10. Plots ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("HEP Cost Overrun — XGBoost Predictions", fontsize=14, fontweight="bold")

def scatter(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5, s=60)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual % Cost Overrun")
    ax.set_ylabel("Predicted % Cost Overrun")
    ax.set_title(title)
    ax.legend(fontsize=8)

scatter(axes[0], y_te25, xgb_pred_25, "XGB — 75/25 split")
scatter(axes[1], y_te35, xgb_pred_35, "XGB — 65/35 split")
scatter(axes[2], y, y_loo_pred,       "XGB — LOO")

plt.tight_layout()
plt.savefig("hep_xgb_predictions.png", dpi=150, bbox_inches="tight")
print("\nPrediction scatter plots -> hep_xgb_predictions.png")

# Feature importance bar chart
fig2, ax2 = plt.subplots(figsize=(10, 7))
importances.head(15).sort_values().plot.barh(ax=ax2, color="steelblue", edgecolor="k")
ax2.set_xlabel("Feature Importance (gain)")
ax2.set_title("Top 15 Feature Importances — XGBoost (65/35 split)")
plt.tight_layout()
plt.savefig("hep_xgb_feature_importance.png", dpi=150, bbox_inches="tight")
print("Feature importance chart  -> hep_xgb_feature_importance.png")

# -- 11. Save best model ------------------------------------------------------
best_row = df_res.iloc[0]
print(f"\nBest XGB configuration overall: {best_row['label']}  (MAPE={best_row['MAPE']:.3f})")

with open("hep_best_xgb.pkl", "wb") as f:
    pickle.dump(xgb_65, f)
print("Saved: hep_best_xgb.pkl")