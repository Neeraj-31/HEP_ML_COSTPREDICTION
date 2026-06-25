
from pathlib import Path
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, GridSearchCV, LeaveOneOut, cross_val_predict
)
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import joblib

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows (fixes the cp1252 encoding crash)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── paths ────────────────────────────────────────────────────────────────────
PREPARED = Path("DA/hep_prepared2.csv")   # adjust if needed
OUT_DIR  = Path(".")

# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(PREPARED)
print(f"Dataset shape : {df.shape}  (n={df.shape[0]} samples, {df.shape[1]-1} features)")

TARGET = "pct_cost_overrun"
X = df.drop(columns=[TARGET]).values
y = df[TARGET].values
print(f"Target range  : [{y.min():.1f}%, {y.max():.1f}%]  mean={y.mean():.1f}%\n")

# ── metrics ──────────────────────────────────────────────────────────────────
def nrmse(y_true, y_pred):
    rmse  = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = y_true.max() - y_true.min()
    return rmse / denom if denom != 0 else np.nan

def safe_mape(y_true, y_pred):
    """
    Clip near-zero actuals before computing MAPE.
    Standard MAPE blows up when |y_true| is near 0.
    We use a floor of 1% (absolute) so the metric stays interpretable.
    """
    eps   = 1.0          # 1 percentage-point floor (target is in % units)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

def report(y_true, y_pred, label):
    mape = safe_mape(y_true, y_pred)
    nrm  = nrmse(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {label:<40} MAPE={mape:.3f}  NRMSE={nrm:.3f}  R2={r2:.3f}")
    return {"label": label, "MAPE": mape, "NRMSE": nrm, "R2": r2}

# ── hyperparameter grids ──────────────────────────────────────────────────────

# RBF kernel: K(x,x') = exp(-gamma * ||x - x'||^2)
# Similarity falls off like a bell curve with distance.
# gamma controls how fast it falls: small gamma = wide influence, large = narrow.
RBF_GRID = {
    "svr__C":       [0.1, 1, 10, 100],
    "svr__gamma":   ["scale", "auto", 0.001, 0.01, 0.1],
    "svr__epsilon": [0.01, 0.1, 0.5, 1.0],
}

# Polynomial kernel: K(x,x') = (gamma * x.x' + coef0)^degree
# Captures interaction terms between features up to degree d.
POLY_GRID = {
    "svr__C":       [0.1, 1, 10, 100],
    "svr__degree":  [2, 3],
    "svr__gamma":   ["scale", "auto"],
    "svr__coef0":   [0.0, 1.0],
    "svr__epsilon": [0.01, 0.1, 0.5],
}

def build_pipeline(kernel, **kwargs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel=kernel, max_iter=100_000, **kwargs)),
    ])

def tune(X_tr, y_tr, kernel, grid, cv=5):
    pipe = build_pipeline(kernel)
    gs   = GridSearchCV(
        pipe, grid,
        cv=cv, scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1, refit=True
    )
    gs.fit(X_tr, y_tr)
    clean = {k.replace("svr__", ""): v for k, v in gs.best_params_.items()}
    print(f"  [SVR-{kernel}] Best params: {clean}")
    return gs.best_estimator_

results = []

# ════════════════════════════════════════════════════════════════════════════
# 75/25 split
# ════════════════════════════════════════════════════════════════════════════
print("-" * 60)
print("Split: 75/25  (train=33, test=12)")
print("-" * 60)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

rbf_75  = tune(X_tr, y_tr, "rbf",  RBF_GRID)
poly_75 = tune(X_tr, y_tr, "poly", POLY_GRID)

results.append(report(y_te, rbf_75.predict(X_te),  "SVR-RBF (75/25)"))
results.append(report(y_te, poly_75.predict(X_te), "SVR-Poly (75/25)"))

# ════════════════════════════════════════════════════════════════════════════
# 65/35 split
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("Split: 65/35  (train=29, test=16)")
print("-" * 60)

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.35, random_state=42)

rbf_65  = tune(X_tr2, y_tr2, "rbf",  RBF_GRID)
poly_65 = tune(X_tr2, y_tr2, "poly", POLY_GRID)

results.append(report(y_te2, rbf_65.predict(X_te2),  "SVR-RBF (65/35)"))
results.append(report(y_te2, poly_65.predict(X_te2), "SVR-Poly (65/35)"))

# ════════════════════════════════════════════════════════════════════════════
# Leave-One-Out CV
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("Leave-One-Out CV (LOO)  full dataset")
print("-" * 60)

# Reuse best params from 65/35 split (stable proxy for full-dataset params)
def loo_predict(best_model):
    pipe = build_pipeline(
        best_model.named_steps["svr"].kernel
    )
    pipe.set_params(**{
        k: v for k, v in best_model.get_params().items()
        if k.startswith("svr__")
    })
    return cross_val_predict(pipe, X, y, cv=LeaveOneOut(), n_jobs=-1)

preds_rbf_loo  = loo_predict(rbf_65)
preds_poly_loo = loo_predict(poly_65)

results.append(report(y, preds_rbf_loo,  "SVR-RBF (LOO)"))
results.append(report(y, preds_poly_loo, "SVR-Poly (LOO)"))

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY - SVR RBF and Polynomial kernels")
print("=" * 65)
df_res = pd.DataFrame(results).sort_values("MAPE")
print(df_res.to_string(index=False))

print("\nPaper benchmark (RS dataset, 35% test):")
print("  SVR RBF: MAPE approx 0.20-0.30, NRMSE approx 0.15-0.25")

# ════════════════════════════════════════════════════════════════════════════
# Scatter plots (6 subplots: RBF and Poly x 3 splits)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
combos = [
    ("RBF  75/25", y_te,  rbf_75.predict(X_te)),
    ("RBF  65/35", y_te2, rbf_65.predict(X_te2)),
    ("RBF  LOO",   y,     preds_rbf_loo),
    ("Poly 75/25", y_te,  poly_75.predict(X_te)),
    ("Poly 65/35", y_te2, poly_65.predict(X_te2)),
    ("Poly LOO",   y,     preds_poly_loo),
]
for ax, (label, yt, yp) in zip(axes.flat, combos):
    ax.scatter(yt, yp, alpha=0.7, edgecolors="k", linewidths=0.5)
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    r2 = r2_score(yt, yp)
    ax.set_title(f"SVR-{label}  R2={r2:.3f}")
    ax.set_xlabel("Actual % cost overrun")
    ax.set_ylabel("Predicted % cost overrun")
    ax.legend(fontsize=8)

plt.tight_layout()
scatter_path = OUT_DIR / "hep_svr_predictions.png"
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
print(f"\nPrediction scatter plots -> {scatter_path}")

# ════════════════════════════════════════════════════════════════════════════
# Save best model (pick whichever LOO result is better)
# ════════════════════════════════════════════════════════════════════════════
rbf_mape  = safe_mape(y, preds_rbf_loo)
poly_mape = safe_mape(y, preds_poly_loo)
best_pipe  = rbf_65 if rbf_mape <= poly_mape else poly_65
best_label = "RBF" if rbf_mape <= poly_mape else "Poly"

# Refit on full dataset
best_final = build_pipeline(best_pipe.named_steps["svr"].kernel)
best_final.set_params(**{
    k: v for k, v in best_pipe.get_params().items()
    if k.startswith("svr__")
})
best_final.fit(X, y)
model_path = OUT_DIR / "hep_best_svr.pkl"
joblib.dump(best_final, model_path)
print(f"Best SVR kernel: {best_label}  (LOO MAPE={min(rbf_mape, poly_mape):.3f})")
print(f"Saved -> {model_path}")