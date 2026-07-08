import pickle
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import dam_common as C

# ---------------- CONFIG ---------------- #
N_SIMS = 10000
MODEL_PATH = "dam_best_rf.pkl"
CSV_PATH = "dam_ml_ready_cleaned.csv"

UNCERTAIN_FEATURES = ["R30mm_total"]

# --------------- FUNCTIONS --------------- #

def fit_feature_distributions(df, uncertain_features, lower_pct=5, upper_pct=95):
    dists = {}

    for feat in uncertain_features:
        col = df[feat].astype(float).values

        mean_ = col.mean()
        std_ = col.std()
        cv = std_ / mean_ if mean_ != 0 else std_

        p_lo, p_hi = np.percentile(col, [lower_pct, upper_pct])
        sk = skew(col)

        if sk > 1 and col.min() > 0:
            dist = "lognormal"
        elif abs(sk) <= 0.5:
            dist = "normal"
        else:
            dist = "triangular"

        dists[feat] = {
            "type": dist,
            "cv": abs(cv),
            "dev_lo": p_lo - mean_,
            "dev_hi": p_hi - mean_,
            "non_negative": col.min() >= 0,
        }

    return dists


def sample_feature(baseline, spec, n, rng):

    if spec["type"] == "normal":
        scale = abs(baseline) * spec["cv"] if baseline != 0 else spec["cv"]
        x = rng.normal(baseline, max(scale, 1e-6), n)

    elif spec["type"] == "lognormal":
        cv = spec["cv"]
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(max(baseline, 1e-6)) - sigma**2 / 2
        x = rng.lognormal(mu, sigma, n)

    else:
        lo = baseline + spec["dev_lo"]
        hi = baseline + spec["dev_hi"]

        if spec["non_negative"]:
            lo = max(lo, 0)

        lo, hi = min(lo, hi), max(lo, hi)
        mode = min(max(baseline, lo), hi)

        if lo == hi:
            x = np.full(n, baseline)
        else:
            x = np.random.triangular(lo, mode, hi, n)

    if spec["non_negative"]:
        x = np.clip(x, 0, None)

    return x


# --------------- LOAD ---------------- #

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
y_shift = bundle["y_shift"]

df = pd.read_csv(CSV_PATH)

feature_dists = fit_feature_distributions(df, UNCERTAIN_FEATURES)

rng = np.random.default_rng(42)

results = []

# ------------ VALIDATION LOOP ------------ #

for _, row in df.iterrows():

    X_sim = np.zeros((N_SIMS, len(features)))

    for j, feat in enumerate(features):

        baseline = row[feat]

        if feat in UNCERTAIN_FEATURES:
            X_sim[:, j] = sample_feature(
                baseline,
                feature_dists[feat],
                N_SIMS,
                rng,
            )
        else:
            X_sim[:, j] = baseline

    X_sim = scaler.transform(X_sim)

    pred_log = model.predict(X_sim)
    pred = C.inv_log(pred_log, y_shift)

    actual = row["cost_overrun_pct"]

    results.append({
        "project": row.get("project_name", f"Project_{len(results)+1}"),
        "actual": actual,
        "pred_mean": pred.mean(),
        "pred_std": pred.std(),
        "P10": np.percentile(pred, 10),
        "P50": np.percentile(pred, 50),
        "P90": np.percentile(pred, 90),
        "P95": np.percentile(pred, 95),
    })

# ------------- METRICS ------------- #

results = pd.DataFrame(results)

mae = mean_absolute_error(results.actual, results.pred_mean)
rmse = np.sqrt(mean_squared_error(results.actual, results.pred_mean))
r2 = r2_score(results.actual, results.pred_mean)
bias = np.mean(results.pred_mean - results.actual)

coverage90 = (
    (results.actual >= results.P10) &
    (results.actual <= results.P90)
).mean() * 100

coverage95 = (
    (results.actual >= results.P10) &
    (results.actual <= results.P95)
).mean() * 100

interval_width = np.mean(results.P90 - results.P10)

print("\n========== MONTE CARLO VALIDATION ==========")
print(f"Projects                : {len(results)}")
print(f"MAE                     : {mae:.3f}")
print(f"RMSE                    : {rmse:.3f}")
print(f"R²                      : {r2:.3f}")
print(f"Bias                    : {bias:.3f}")
print(f"P10-P90 Coverage        : {coverage90:.2f}%")
print(f"P10-P95 Coverage        : {coverage95:.2f}%")
print(f"Average Interval Width  : {interval_width:.3f}")

results.to_csv("mc_validation_results.csv", index=False)

summary = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2",
        "Bias",
        "P10-P90 Coverage (%)",
        "P10-P95 Coverage (%)",
        "Average Interval Width"
    ],
    "Value": [
        mae,
        rmse,
        r2,
        bias,
        coverage90,
        coverage95,
        interval_width
    ]
})

summary.to_csv("mc_validation_summary.csv", index=False)

print("\nSaved:")
print("  mc_validation_results.csv")
print("  mc_validation_summary.csv")