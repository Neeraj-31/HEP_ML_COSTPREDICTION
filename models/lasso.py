import warnings
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, lasso_path
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

RANDOM_SEED = 42
DATA_PATH = Path("DA/hep_prepared2.csv")
TARGET = "pct_cost_overrun"
np.random.seed(RANDOM_SEED)

df = pd.read_csv(DATA_PATH)
FEATURES = [c for c in df.columns if c != TARGET]
X, y = df[FEATURES].values, df[TARGET].values


def nrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.mean(np.abs(y_true))


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def report(y_true, y_pred, label):
    m, n, r = mape(y_true, y_pred), nrmse(y_true, y_pred), r2_score(y_true, y_pred)
    print(f"  {label:<25}  MAPE={m:.3f}  NRMSE={n:.3f}  R²={r:.3f}")
    return {"label": label, "MAPE": m, "NRMSE": n, "R2": r}


results = []

# LOO CV Alpha Selection & Prediction
lasso_cv = LassoCV(cv=LeaveOneOut(), max_iter=10_000, random_state=RANDOM_SEED, n_alphas=200)
lasso_cv.fit(X, y)
best_alpha = lasso_cv.alpha_

lasso_fixed = Lasso(alpha=best_alpha, max_iter=10_000)
y_pred_loo = cross_val_predict(lasso_fixed, X, y, cv=LeaveOneOut())
results.append(report(y, y_pred_loo, "Lasso (LOO)"))

# Holdout Train/Test Splits
for test_size, label in [(0.25, "75/25"), (0.35, "65/35")]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)
    lcv = LassoCV(cv=min(5, len(y_tr)), max_iter=10_000, random_state=RANDOM_SEED, n_alphas=200)
    lcv.fit(X_tr, y_tr)
    lasso_split = Lasso(alpha=lcv.alpha_, max_iter=10_000)
    lasso_split.fit(X_tr, y_tr)
    results.append(report(y_te, lasso_split.predict(X_te), f"Lasso ({label})"))

# Feature Coefficients
lasso_fixed.fit(X, y)
coef_series = pd.Series(lasso_fixed.coef_, index=FEATURES)
nonzero = coef_series[coef_series != 0].sort_values(key=abs, ascending=False)

# Visualizations
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y, y_pred_loo, alpha=0.75, edgecolors="k", color="steelblue")
ax.plot([min(y) - 10, max(y) + 10], [min(y) - 10, max(y) + 10], "r--")
ax.set_title("Lasso Predicted vs Actual")
plt.savefig("hep_lasso_scatter.png", dpi=150, bbox_inches="tight")

if len(nonzero) > 0:
    fig2, ax2 = plt.subplots(figsize=(10, max(4, len(nonzero) * 0.45)))
    nonzero.sort_values().plot.barh(ax=ax2, color=["#d62728" if v < 0 else "#1f77b4" for v in nonzero.sort_values()])
    ax2.axvline(0, color="black", linewidth=0.8)
    plt.savefig("hep_lasso_coefficients.png", dpi=150, bbox_inches="tight")

alphas_path, coefs_path, _ = lasso_path(X, y, n_alphas=100)
fig3, ax3 = plt.subplots(figsize=(11, 5))
for i, fname in enumerate(FEATURES):
    if np.any(coefs_path[i] != 0):
        ax3.plot(np.log10(alphas_path), coefs_path[i], label=fname)
plt.savefig("hep_lasso_path.png", dpi=150, bbox_inches="tight")