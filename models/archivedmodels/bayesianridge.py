
import sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, GridSearchCV,
    LeaveOneOut, cross_val_predict
)
from sklearn.metrics import r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── paths ─────────────────────────────────────────────────────────────────
PREPARED = Path("DA/hep_prepared2.csv")
OUT_DIR  = Path(".")

# ── load ───────────────────────────────────────────────────────────────────
df     = pd.read_csv(PREPARED)
TARGET = "pct_cost_overrun"
X_raw  = df.drop(columns=[TARGET]).values
y_raw  = df[TARGET].values
feat_names = df.drop(columns=[TARGET]).columns.tolist()

print(f"Dataset  : {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"Target   : min={y_raw.min():.1f}%  max={y_raw.max():.1f}%  mean={y_raw.mean():.1f}%\n")

# ── log transform on target ────────────────────────────────────────────────
# Shift so the minimum is above 0 before taking log.
# y_raw min is -59.1, so shift by 65 to give a floor of ~5.9 before log.
SHIFT = 65.0
y = np.log(y_raw + SHIFT)   # inverse: np.exp(y_pred) - SHIFT

def inverse(y_log):
    return np.exp(y_log) - SHIFT

# ── metrics (all computed in original % space) ─────────────────────────────
def nrmse(yt, yp):
    rmse  = np.sqrt(np.mean((yt - yp) ** 2))
    denom = yt.max() - yt.min()
    return rmse / denom if denom != 0 else np.nan

def safe_mape(yt, yp):
    """Floor denominator at 1 pp to stop near-zero blowup."""
    return np.mean(np.abs(yt - yp) / np.maximum(np.abs(yt), 1.0)) * 100

def report(yt, yp_log, label):
    yp = inverse(yp_log)
    mape = safe_mape(yt, yp)
    nrm  = nrmse(yt, yp)
    r2   = r2_score(yt, yp)
    print(f"  {label:<45} MAPE={mape:7.3f}  NRMSE={nrm:.3f}  R2={r2:.3f}")
    return {"label": label, "MAPE": mape, "NRMSE": nrm, "R2": r2,
            "y_true": yt, "y_pred": yp}

# ── pipeline ───────────────────────────────────────────────────────────────
# SelectKBest: keeps the k features most correlated with the LOG target.
# StandardScaler: Bayesian Ridge is sensitive to feature scale.
# BayesianRidge: alpha_1/2 and lambda_1/2 are gamma-distribution params
#   controlling how tightly weights are shrunk toward zero.
def build():
    return Pipeline([
        ("sel",    SelectKBest(f_regression)),
        ("scaler", StandardScaler()),
        ("br",     BayesianRidge(max_iter=500, tol=1e-6, compute_score=True)),
    ])

PARAM_GRID = {
    "sel__k":          [5, 8, 10, 12, 15, 20],
    "br__alpha_1":     [1e-6, 1e-4, 1e-2],
    "br__alpha_2":     [1e-6, 1e-4, 1e-2],
    "br__lambda_1":    [1e-6, 1e-4, 1e-2],
    "br__lambda_2":    [1e-6, 1e-4, 1e-2],
}

def tune(X_tr, y_tr, cv=5):
    gs = GridSearchCV(
        build(), PARAM_GRID,
        cv=cv, scoring="r2",
        n_jobs=-1, refit=True
    )
    gs.fit(X_tr, y_tr)
    k   = gs.best_params_["sel__k"]
    a1  = gs.best_params_["br__alpha_1"]
    l1  = gs.best_params_["br__lambda_1"]
    print(f"  [BR] Best: k={k}, alpha_1={a1}, lambda_1={l1}")
    return gs.best_estimator_

results = []
models  = {}

# ═══════════════════════════════════════════════════════════════════════════
# 75/25
# ═══════════════════════════════════════════════════════════════════════════
print("-" * 60)
print("Split: 75/25  (train=33, test=12)")
print("-" * 60)
X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.25, random_state=42)
m75 = tune(X_tr, y_tr)
res75 = report(inverse(y_te), m75.predict(X_te), "BayesianRidge (75/25)")
results.append(res75); models["75/25"] = (m75, inverse(y_te), inverse(m75.predict(X_te)))

# ═══════════════════════════════════════════════════════════════════════════
# 65/35
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("Split: 65/35  (train=29, test=16)")
print("-" * 60)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_raw, y, test_size=0.35, random_state=42)
m65 = tune(X_tr2, y_tr2)
res65 = report(inverse(y_te2), m65.predict(X_te2), "BayesianRidge (65/35)")
results.append(res65); models["65/35"] = (m65, inverse(y_te2), inverse(m65.predict(X_te2)))

# ═══════════════════════════════════════════════════════════════════════════
# LOO
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("Leave-One-Out CV  (full dataset)")
print("-" * 60)
# Reuse best params from 65/35 for LOO
loo_pipe = build()
loo_pipe.set_params(**{k: v for k, v in m65.get_params().items()
                       if k.startswith("sel__") or k.startswith("br__")})
preds_loo_log = cross_val_predict(loo_pipe, X_raw, y, cv=LeaveOneOut(), n_jobs=-1)
res_loo = report(y_raw, preds_loo_log, "BayesianRidge (LOO)")
results.append(res_loo)
models["LOO"] = (None, y_raw, inverse(preds_loo_log))


print("\n" + "=" * 65)
print("SUMMARY - Bayesian Ridge (log-transformed target)")
print("=" * 65)
df_res = (pd.DataFrame(results)[["label","MAPE","NRMSE","R2"]]
          .sort_values("R2", ascending=False))
print(df_res.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# Feature importances 
# ═══════════════════════════════════════════════════════════════════════════
# Refit on full data for final coefficients
best_params = {k: v for k, v in m65.get_params().items()
               if k.startswith("sel__") or k.startswith("br__")}
final_pipe = build()
final_pipe.set_params(**best_params)
final_pipe.fit(X_raw, y)

sel_mask   = final_pipe.named_steps["sel"].get_support()
sel_names  = [feat_names[i] for i, s in enumerate(sel_mask) if s]
coefs      = np.abs(final_pipe.named_steps["br"].coef_)
importance = pd.Series(coefs, index=sel_names).sort_values(ascending=False)

print("\nTop selected features (absolute coefficient):")
print(importance.to_string())

# ═══════════════════════════════════════════════════════════════════════════
# Scatter plots
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (split, (_, yt, yp)) in zip(axes, models.items()):
    ax.scatter(yt, yp, alpha=0.7, edgecolors="k", linewidths=0.5)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    ax.set_title(f"BayesianRidge ({split})  R2={r2_score(yt,yp):.3f}")
    ax.set_xlabel("Actual % cost overrun"); ax.set_ylabel("Predicted")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "hep_br_predictions.png", dpi=150, bbox_inches="tight")
print("\nScatter plots -> hep_br_predictions.png")

joblib.dump({"pipeline": final_pipe, "shift": SHIFT}, OUT_DIR / "hep_best_br.pkl")
print("Saved model -> hep_best_br.pkl")
print("\nTo predict on new data:")
print('  obj = joblib.load("hep_best_br.pkl")')
print('  y_pred_pct = np.exp(obj["pipeline"].predict(X_new)) - obj["shift"]')