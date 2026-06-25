import warnings
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
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

# LOO CV Hyperparameter Tuning & Prediction
enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
    cv=LeaveOneOut(),
    max_iter=10_000,
    random_state=RANDOM_SEED,
    n_alphas=100,
)
enet_cv.fit(X, y)

enet_fixed = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_, max_iter=10_000)
y_pred_loo = cross_val_predict(enet_fixed, X, y, cv=LeaveOneOut())
results.append(report(y, y_pred_loo, "ElasticNet (LOO)"))

# Holdout Train/Test Splits
for test_size, label in [(0.25, "75/25"), (0.35, "65/35")]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)
    ecv = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=min(5, len(y_tr)),
        max_iter=10_000,
        random_state=RANDOM_SEED,
        n_alphas=100,
    )
    ecv.fit(X_tr, y_tr)
    enet_split = ElasticNet(alpha=ecv.alpha_, l1_ratio=ecv.l1_ratio_, max_iter=10_000)
    enet_split.fit(X_tr, y_tr)
    results.append(report(y_te, enet_split.predict(X_te), f"ElasticNet ({label})"))

# Scatter Plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y, y_pred_loo, alpha=0.75, edgecolors="k", color="seagreen")
ax.plot([min(y) - 10, max(y) + 10], [min(y) - 10, max(y) + 10], "r--")
ax.set_title("ElasticNet Predicted vs Actual")
plt.savefig("hep_enet_scatter.png", dpi=150, bbox_inches="tight")