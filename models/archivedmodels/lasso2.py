import warnings
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, lasso_path
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

RANDOM_SEED  = 42
DATA_PATH    = Path("DA/hep_prepared2.csv")
TARGET       = "pct_cost_overrun"
OUTLIER_CAP  = 99   # percentile cap (set None to disable)

np.random.seed(RANDOM_SEED)

# ── 1.  Load & preprocess ────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
FEATURES = [c for c in df.columns if c != TARGET]
X_raw = df[FEATURES].copy()
y_raw = df[TARGET].values

# -- Outlier capping ---------------------------------------------------------
if OUTLIER_CAP is not None:
    cap  = np.percentile(y_raw, OUTLIER_CAP)
    mask = y_raw <= cap
    X_raw = X_raw[mask]
    y_raw = y_raw[mask]
    print(f"Outlier cap  : removed {(~mask).sum()} samples with target > {cap:.1f}%")

# -- Interaction features ----------------------------------------------------
X_raw = X_raw.copy()
if 'cost_per_mw' in X_raw.columns and 'transmission_km' in X_raw.columns:
    X_raw['cost_x_km']  = X_raw['cost_per_mw'] * X_raw['transmission_km']
if 'cost_per_mw' in X_raw.columns and 'geo_prob_yes-fundstop+stresstransition' in X_raw.columns:
    X_raw['cost_x_geo'] = X_raw['cost_per_mw'] * X_raw['geo_prob_yes-fundstop+stresstransition']
if 'installed_cap_mw' in X_raw.columns and 'tunnel_length_m' in X_raw.columns:
    X_raw['cap_x_tun']  = X_raw['installed_cap_mw'] * X_raw['tunnel_length_m']
ALL_FEATURES = list(X_raw.columns)

X_full = X_raw.values
y_full = y_raw

# -- Log-transform target ----------------------------------------------------
y_shift = y_full.min() - 1.0
y_log   = np.log(y_full - y_shift)

# ── 2.  Metric helpers (always in original space) ────────────────────────────
def nrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.mean(np.abs(y_true))

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def inv_log(y_log_pred):
    return np.exp(y_log_pred) + y_shift

def report(y_true, y_pred, label):
    m, n, r = mape(y_true, y_pred), nrmse(y_true, y_pred), r2_score(y_true, y_pred)
    print(f"  {label:<25}  MAPE={m:.3f}  NRMSE={n:.3f}  R²={r:.3f}")
    return {"label": label, "MAPE": m, "NRMSE": n, "R2": r}

results = []

# ── 3.  LOO CV — fit scaler + model inside each fold ─────────────────────────
loo = LeaveOneOut()

# Find best alpha on the full scaled dataset first
scaler_full = RobustScaler()
X_scaled_full = scaler_full.fit_transform(X_full)

lasso_cv = LassoCV(
    cv=LeaveOneOut(), max_iter=10_000,
    random_state=RANDOM_SEED, n_alphas=200
)
lasso_cv.fit(X_scaled_full, y_log)
best_alpha = lasso_cv.alpha_
print(f"Best Lasso alpha (LOO): {best_alpha:.6f}")

# LOO predictions with per-fold scaling
y_loo_pred_log = np.empty_like(y_log, dtype=float)
for train_idx, test_idx in loo.split(X_full):
    sc = RobustScaler()
    X_tr_s = sc.fit_transform(X_full[train_idx])
    X_te_s = sc.transform(X_full[test_idx])
    m = Lasso(alpha=best_alpha, max_iter=10_000)
    m.fit(X_tr_s, y_log[train_idx])
    y_loo_pred_log[test_idx] = m.predict(X_te_s)

y_loo_pred = inv_log(y_loo_pred_log)
results.append(report(y_full, y_loo_pred, "Lasso (LOO)"))

# ── 4.  Holdout splits ───────────────────────────────────────────────────────
for test_size, label in [(0.25, "75/25"), (0.35, "65/35")]:
    X_tr, X_te, yl_tr, yl_te, y_tr, y_te = train_test_split(
        X_full, y_log, y_full, test_size=test_size, random_state=RANDOM_SEED
    )
    sc = RobustScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    lcv = LassoCV(
        cv=min(5, len(y_tr)), max_iter=10_000,
        random_state=RANDOM_SEED, n_alphas=200
    )
    lcv.fit(X_tr_s, yl_tr)
    lasso_split = Lasso(alpha=lcv.alpha_, max_iter=10_000)
    lasso_split.fit(X_tr_s, yl_tr)
    y_pred = inv_log(lasso_split.predict(X_te_s))
    results.append(report(y_te, y_pred, f"Lasso ({label})"))

# ── 5.  Feature coefficients ─────────────────────────────────────────────────
lasso_fixed = Lasso(alpha=best_alpha, max_iter=10_000)
lasso_fixed.fit(X_scaled_full, y_log)
coef_series = pd.Series(lasso_fixed.coef_, index=ALL_FEATURES)
nonzero = coef_series[coef_series != 0].sort_values(key=abs, ascending=False)

# ── 6.  Visualisations ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_full, y_loo_pred, alpha=0.75, edgecolors="k", color="steelblue")
ax.plot([min(y_full) - 10, max(y_full) + 10],
        [min(y_full) - 10, max(y_full) + 10], "r--")
ax.set_title("Lasso Predicted vs Actual (log-transform + RobustScaler)")
ax.set_xlabel("Actual % Cost Overrun"); ax.set_ylabel("Predicted % Cost Overrun")
plt.tight_layout()
plt.savefig("hep_lasso_scatter.png", dpi=150, bbox_inches="tight")

if len(nonzero) > 0:
    fig2, ax2 = plt.subplots(figsize=(10, max(4, len(nonzero) * 0.45)))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in nonzero.sort_values()]
    nonzero.sort_values().plot.barh(ax=ax2, color=colors)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title("Lasso Coefficients (log-space)")
    plt.tight_layout()
    plt.savefig("hep_lasso_coefficients.png", dpi=150, bbox_inches="tight")

alphas_path, coefs_path, _ = lasso_path(X_scaled_full, y_log, n_alphas=100)
fig3, ax3 = plt.subplots(figsize=(11, 5))
for i, fname in enumerate(ALL_FEATURES):
    if np.any(coefs_path[i] != 0):
        ax3.plot(np.log10(alphas_path), coefs_path[i], label=fname)
ax3.set_xlabel("log10(alpha)"); ax3.set_ylabel("Coefficient")
ax3.set_title("Lasso Path"); ax3.legend(fontsize=7)
plt.tight_layout()
plt.savefig("hep_lasso_path.png", dpi=150, bbox_inches="tight")

print("\nPlots saved: hep_lasso_scatter.png, hep_lasso_coefficients.png, hep_lasso_path.png")