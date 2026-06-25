import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, GridSearchCV, LeaveOneOut, cross_val_score
)
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# ── 0.  Config ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
PREPARED    = Path("DA\hep_prepared2.csv")
TARGET      = "pct_cost_overrun"

np.random.seed(RANDOM_SEED)

# ── 1.  Load prepared data ──────────────────────────────────────────────────
df = pd.read_csv(PREPARED)
assert df.isnull().sum().sum() == 0, "NaN found — run hep_data_prep.py first"
NECCESARY_COLUMNS=['cost_per_mw','geo_prob_yes-fundstop+stresstransition','transmission_km','initial_cost','geo_prob_yes-landslide','installed_cap_mw','tunnel_length_m',"glof_risk"]
FEATURES = [c for c in df.columns if c in NECCESARY_COLUMNS]
X = df[FEATURES].values
y = df[TARGET].values

print(f"Dataset shape : {X.shape}  (n={len(y)} samples, {len(FEATURES)} features)")
print(f"Target range  : [{y.min():.1f}%, {y.max():.1f}%]  mean={y.mean():.1f}%")
print()

# ── 2.  Metric helpers ───────────────────────────────────────────────────────
def nrmse(y_true, y_pred):
    """Normalised RMSE = RMSE / mean(y_true)  — same as paper's NRSME."""
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
    print(f"  {label:<30}  MAPE={m:.3f}  NRMSE={n:.3f}  R²={r:.3f}")
    return {"label": label, "MAPE": m, "NRMSE": n, "R2": r}

# ── 3.  Hyperparameter grid (mirrors Table 4 of paper) ─────────────────────
RF_GRID = {
    "n_estimators" : [50, 100, 150, 200, 250],
    "max_depth"    : [None, 2, 3, 4, 5],
    "min_samples_leaf" : [1, 2, 5],
}

results = []

# ── 4.  Evaluation function for one train/test split ────────────────────────
def run_split(X_train, X_test, y_train, y_test, split_label):
    print(f"\n{'-'*60}")
    print(f"Split: {split_label}  (train={len(y_train)}, test={len(y_test)})")
    print(f"{'-'*60}")

    rf_base = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
    rf_cv   = GridSearchCV(
        rf_base, RF_GRID, cv=3, scoring="neg_mean_absolute_error",
        n_jobs=1, refit=True
    )
    rf_cv.fit(X_train, y_train)
    best_rf = rf_cv.best_estimator_
    print(f"\n[RF] Best params: {rf_cv.best_params_}")

    rf_pred = best_rf.predict(X_test)
    r_rf = evaluate(y_test, rf_pred, label=f"Random Forest ({split_label})")
    results.append(r_rf)

    return best_rf, rf_pred

# ── 5.  Run 75/25 split (paper's first split) ───────────────────────────────
X_tr75, X_te25, y_tr75, y_te25 = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_SEED
)
rf_75, rf_pred_25 = run_split(X_tr75, X_te25, y_tr75, y_te25, split_label="75/25")

# ── 6.  Run 65/35 split (paper's second split) ──────────────────────────────
X_tr65, X_te35, y_tr65, y_te35 = train_test_split(
    X, y, test_size=0.35, random_state=RANDOM_SEED
)
rf_65, rf_pred_35 = run_split(X_tr65, X_te35, y_tr65, y_te35, split_label="65/35")

# ── 7.  Leave-One-Out CV ────────────────────────────────────────────────────
print(f"\n{'-'*60}")
print("Leave-One-Out CV (LOO)  full dataset")
print(f"{'-'*60}")

loo = LeaveOneOut()

# Reconstruct predictions manually for NRMSE and R²
y_loo_pred = np.empty_like(y, dtype=float)
for train_idx, test_idx in loo.split(X):
    m_clone = RandomForestRegressor(**rf_65.get_params())
    m_clone.fit(X[train_idx], y[train_idx])
    y_loo_pred[test_idx] = m_clone.predict(X[test_idx])
r = evaluate(y, y_loo_pred, label="Random Forest (LOO)")
results.append(r)

# ── 8.  Summary table ────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUMMARY - all Random Forest splits")
print(f"{'='*65}")
df_res = pd.DataFrame(results)
df_res = df_res.sort_values("MAPE")
print(df_res.to_string(index=False))
print()
print("Paper benchmark (RS dataset, 35% test):")
print("  Random Forest: MAPE=0.186, NRMSE=0.14")
print()

# ── 9.  Feature importances (best RF on 65/35 split) ─────────────────────────
importances = pd.Series(rf_65.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)
print("Top 15 feature importances (Random Forest 65/35):")
print(importances.head(15).to_string())

# ── 10.  Plots ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("HEP Cost Overrun - Random Forest Predictions", fontsize=14, fontweight="bold")

def scatter(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5, s=60)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual % Cost Overrun")
    ax.set_ylabel("Predicted % Cost Overrun")
    ax.set_title(title)
    ax.legend(fontsize=8)

scatter(axes[0], y_te25, rf_pred_25, "RF  — 75/25 split")
scatter(axes[1], y_te35, rf_pred_35, "RF  — 65/35 split")
scatter(axes[2], y, y_loo_pred,      "RF  — LOO")

plt.tight_layout()
plt.savefig("hep_rf_predictions.png", dpi=150, bbox_inches="tight")
print("\nPrediction scatter plots -> hep_rf_predictions.png")

# Feature importance bar chart
fig2, ax2 = plt.subplots(figsize=(10, 7))
importances.head(15).sort_values().plot.barh(ax=ax2, color="steelblue", edgecolor="k")
ax2.set_xlabel("Feature Importance (mean impurity decrease)")
ax2.set_title("Top 15 Feature Importances — Random Forest (65/35 split)")
plt.tight_layout()
plt.savefig("hep_rf_feature_importance.png", dpi=150, bbox_inches="tight")
print("Feature importance chart   -> hep_rf_feature_importance.png")

# ── 11.  Save best model ──────────────────────────────────────────────────────
best_row = df_res.iloc[0]
print(f"\nBest RF configuration overall: {best_row['label']}  (MAPE={best_row['MAPE']:.3f})")

with open("hep_best_rf.pkl", "wb") as f:
    pickle.dump(rf_65, f)
print("Saved: hep_best_rf.pkl")