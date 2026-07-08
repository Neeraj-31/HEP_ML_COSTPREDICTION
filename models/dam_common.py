"""
dam_common.py
──────────────
Shared config, data loading, metrics, and evaluation harness used by
dam_lasso.py, dam_rf.py, dam_xgb.py, and dam_catboost.py.

All four model scripts follow the same evaluation protocol so results are
directly comparable:
  - 75/25 train/test split
  - 65/35 train/test split
  - Leave-One-Out cross-validation (full dataset)
Each split is scaled with RobustScaler (fit on train only) and the target
is trained in log-space (log-transform reduces the influence of the
right-skewed tail that's typical of cost-overrun %).

Feature set: dam_lasso.py runs first and selects a subset of features via
LassoCV, saving them to dam_selected_features.json. dam_rf.py, dam_xgb.py,
and dam_catboost.py all load that same file, so every model is trained on
the identical feature set for a fair comparison.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler

RANDOM_SEED = 42
CLEANED_CSV = Path("dam_ml_ready_cleaned.csv")
SELECTED_FEATURES_JSON = Path("models\dam_selected_features.json")
TARGET = "cost_overrun_pct"
DROP_COLS = ["project_name", TARGET]   # never used as features

np.random.seed(RANDOM_SEED)


# ── Data loading ──────────────────────────────────────────────────────────
def load_cleaned():
    """Load the cleaned dataset. Run dam_prepare_data.py first if missing."""
    if not CLEANED_CSV.exists():
        raise FileNotFoundError(
            f"{CLEANED_CSV} not found. Run `python dam_prepare_data.py` first."
        )
    df = pd.read_csv(CLEANED_CSV)
    assert df.isnull().sum().sum() == 0, "NaN found in cleaned data"
    return df


def load_selected_features():
    """Load the feature list chosen by dam_lasso.py."""
    if not SELECTED_FEATURES_JSON.exists():
        raise FileNotFoundError(
            f"{SELECTED_FEATURES_JSON} not found. Run `python dam_lasso.py` "
            "first — it performs feature selection and saves the file the "
            "other model scripts depend on."
        )
    with open(SELECTED_FEATURES_JSON) as f:
        return json.load(f)["selected_features"]


def get_all_candidate_features(df):
    """All engineered-feature columns available for Lasso to choose from."""
    return [c for c in df.columns if c not in DROP_COLS]


def get_X_y(df, features):
    X = df[features].values
    y = df[TARGET].values
    return X, y


# ── Log-target transform ─────────────────────────────────────────────────
def make_log_transform(y_full):
    y_shift = y_full.min() - 1.0        # guarantees y - y_shift > 0
    y_log = np.log(y_full - y_shift)
    return y_log, y_shift


def inv_log(y_log_pred, y_shift):
    return np.exp(y_log_pred) + y_shift


# ── Metrics ───────────────────────────────────────────────────────────────
def nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / np.mean(np.abs(y_true))


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def evaluate(y_true, y_pred, label=""):
    m = mape(y_true, y_pred)
    n = nrmse(y_true, y_pred)
    r = r2_score(y_true, y_pred)
    print(f"  {label:<28}  MAPE={m:.3f}  NRMSE={n:.3f}  R2={r:.3f}")
    return {"label": label, "MAPE": m, "NRMSE": n, "R2": r}


# ── Generic split evaluation (used by every model script) ─────────────────
def run_split(model_search_fn, X_train, X_test, y_train_log, y_test_orig,
              y_shift, split_label, model_tag):
    """
    model_search_fn: callable(X_tr_scaled, y_train_log) -> fitted estimator
                      with a .predict method (typically a fitted
                      GridSearchCV.best_estimator_)
    Returns (best_estimator, scaler, predictions_in_original_space, metrics_dict)
    """
    print(f"\n{'-'*60}")
    print(f"Split: {split_label}  (train={len(y_train_log)}, test={len(y_test_orig)})")
    print(f"{'-'*60}")

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    best_model = model_search_fn(X_tr_s, y_train_log)

    pred_log = best_model.predict(X_te_s)
    pred_orig = inv_log(pred_log, y_shift)
    metrics = evaluate(y_test_orig, pred_orig, label=f"{model_tag} ({split_label})")

    return best_model, scaler, pred_orig, metrics


def run_loo(clone_fn, X_full, y_log, y_full, y_shift, model_tag):
    """
    clone_fn: callable() -> a fresh, unfitted estimator instance configured
              with the best hyperparameters (found on a train/test split).
    Runs Leave-One-Out CV over the full dataset and returns predictions
    plus the metrics dict.
    """
    from sklearn.model_selection import LeaveOneOut
    print(f"\n{'-'*60}")
    print(f"Leave-One-Out CV (LOO) — full dataset — {model_tag}")
    print(f"{'-'*60}")

    loo = LeaveOneOut()
    y_loo_pred = np.empty_like(y_full, dtype=float)

    for train_idx, test_idx in loo.split(X_full):
        scaler_loo = RobustScaler()
        X_tr_s = scaler_loo.fit_transform(X_full[train_idx])
        X_te_s = scaler_loo.transform(X_full[test_idx])

        m_clone = clone_fn()
        m_clone.fit(X_tr_s, y_log[train_idx])
        y_loo_pred[test_idx] = inv_log(m_clone.predict(X_te_s), y_shift)

    metrics = evaluate(y_full, y_loo_pred, label=f"{model_tag} (LOO)")
    return y_loo_pred, metrics


# ── Plotting ──────────────────────────────────────────────────────────────
def plot_predictions(y_te25, pred_25, y_te35, pred_35, y_full, loo_pred,
                      model_tag, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Dam Cost Overrun — {model_tag} Predictions",
                 fontsize=14, fontweight="bold")

    def scatter(ax, y_true, y_pred, title):
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5, s=60)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
        ax.set_xlabel("Actual % Cost Overrun")
        ax.set_ylabel("Predicted % Cost Overrun")
        ax.set_title(title)
        ax.legend(fontsize=8)

    scatter(axes[0], y_te25, pred_25, f"{model_tag} — 75/25 split")
    scatter(axes[1], y_te35, pred_35, f"{model_tag} — 65/35 split")
    scatter(axes[2], y_full, loo_pred, f"{model_tag} — LOO")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPrediction scatter plots -> {out_path}")


def plot_importances(importances, model_tag, out_path, ylabel="Importance"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(importances))))
    importances.sort_values().plot.barh(ax=ax, color="steelblue", edgecolor="k")
    ax.set_xlabel(ylabel)
    ax.set_title(f"Feature Importance — {model_tag}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance chart  -> {out_path}")


# ── Save model ────────────────────────────────────────────────────────────
def save_model(payload, out_path):
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved: {out_path}")


def summarize(results, label):
    print(f"\n{'='*65}")
    print(f"SUMMARY — {label}")
    print(f"{'='*65}")
    df_res = pd.DataFrame(results).sort_values("MAPE")
    print(df_res.to_string(index=False))
    return df_res
