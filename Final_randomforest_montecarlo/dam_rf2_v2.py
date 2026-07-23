#!/usr/bin/env python3
"""
dam_risk_predictor.py
======================
Single-project cost-overrun risk predictor for the dam RF model.

Loads the bundle produced by dam_train.py (dam_best_rf.pkl) and, given the
feature values for ONE project, reports:

  - P50 / P70 / P95 predicted cost overrun %
  - A Low / Medium / High risk category
  - A calibration check: across the 45 historical projects, what fraction
    of true outcomes actually fell inside the model's P10-P90 and P5-P95
    bands (honest, leave-one-out - tells you whether to trust the bands).

HOW THE PERCENTILES ARE BUILT
------------------------------
For a brand-new project we don't have a "true" uncertainty range to feed
in (unlike the Monte-Carlo sensitivity script), so the spread has to come
from the model itself. Two honest sources of uncertainty are combined:

  1. Tree-level spread: each of the 500 trees in the forest gives its own
     prediction for this input. Their spread reflects how much the trees
     disagree about *this particular point* - usually small near the
     training data, larger if the input is unusual/extrapolated.

  2. LOOCV residual distribution: the real, honest, out-of-fold errors
     the model made on all 45 historical projects (recomputed here from
     the original training data if available). This captures the actual
     real-world error the model tends to make - which, given n=45 and a
     shallow RF, is considerably larger than tree-to-tree disagreement
     alone would suggest.

  We use (1) only as a per-point SCALE FACTOR on top of (2): if this
  project's trees disagree more than the training-set average
  (avg_tree_log_std, stored in the bundle), the historical residual
  spread is widened proportionally; otherwise the historical spread is
  used as-is. This avoids double-counting uncertainty while still making
  the bands point-specific.

  combined_sample = point_prediction + (LOOCV_residual * scale_factor)

  P50/P70/P95/etc. are then just the percentiles of that combined,
  resampled distribution.

If dam_common.py / the original training data are not importable, the
script falls back to treating the residual distribution as Normal with
std = the LOOCV RMSE stored in the bundle, and skips the calibration
check (clearly flagged in the output).

USAGE
-----
Interactive:
    python dam_risk_predictor.py

From a JSON file:
    python dam_risk_predictor.py --json myproject.json

From the command line:
    python dam_risk_predictor.py --set cost_per_mw=120 --set initial_cost=450 ...

Skip the (slower) historical calibration recheck:
    python dam_risk_predictor.py --json myproject.json --skip-calibration
"""

import argparse
import json
import pickle
import sys

import numpy as np

MODEL_PATH = "dam_best_rf.pkl"

FEATURE_HINTS = {
    "cost_per_mw": "Cost per MW of installed capacity",
    "initial_cost": "Initial planned/sanctioned project cost",
    "actual_dur": "Actual project duration",
    "R30mm_total": "Total count of days with >=30mm rainfall",
    "schedule_overrun_pct": "Schedule overrun, % of planned duration",
    "dist_to_nearest_pilgrim_km": "Distance to nearest pilgrimage site (km)",
    "R100mm_total": "Total count of days with >=100mm rainfall",
    "glof_composite_score": "Glacial lake outburst flood (GLOF) composite risk score",
    "steep_frac_25": "Fraction of catchment/site with slope > 25 degrees",
    "CWD_mean": "Mean consecutive wet days (construction-halting weather)",
    "landowner_displaced": "Number of landowners/households displaced",
    "dist_to_nearest_resort_km": "Distance to nearest resort/tourism site (km)",
    "PRCPTOT_max": "Max annual total precipitation observed",
}


# --------------------------------------------------------------------------
# Bundle / module loading
# --------------------------------------------------------------------------

def load_bundle(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def try_load_common():
    """Best-effort import of dam_common.py (used by dam_train.py). Returns
    None if not importable - the script degrades gracefully in that case."""
    try:
        import dam_common as C
        return C
    except Exception:
        return None


def _log_transform_fallback(y, shift):
    return np.log(np.asarray(y, dtype=float) + shift)


def _inv_log_fallback(y_log, shift):
    return np.exp(np.asarray(y_log, dtype=float)) - shift


def get_transforms(C):
    """Prefer the project's own log_transform/inv_log for exactness; fall
    back to a standard log/exp-with-shift if dam_common isn't available."""
    if C is not None and hasattr(C, "log_transform") and hasattr(C, "inv_log"):
        return C.log_transform, C.inv_log
    return _log_transform_fallback, _inv_log_fallback


# --------------------------------------------------------------------------
# Recompute honest LOOCV out-of-fold predictions/residuals (for the
# calibration check, and to source the residual distribution for new
# predictions). Requires dam_common.py + the original training data.
# --------------------------------------------------------------------------

def recompute_loocv(bundle, C):
    if C is None or not hasattr(C, "load_cleaned"):
        print("  (dam_common.py not found - skipping recomputation of honest "
              "LOOCV residuals; falling back to a Normal-approximation using "
              "the stored LOOCV RMSE.)", file=sys.stderr)
        return None
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import LeaveOneOut

        df = C.load_cleaned()
        features = bundle["features"]
        y = df["cost_overrun_pct"].values.astype(float)
        X = df[features].values.astype(float)
        y_shift = bundle["y_shift"]
        log_transform, inv_log = get_transforms(C)
        y_log = log_transform(y, y_shift)

        rf_params = bundle["model"].get_params()

        loo = LeaveOneOut()
        preds_log = np.zeros(len(y_log))
        for tr_idx, te_idx in loo.split(X):
            scaler = StandardScaler().fit(X[tr_idx])
            Xtr, Xte = scaler.transform(X[tr_idx]), scaler.transform(X[te_idx])
            m = RandomForestRegressor(**rf_params)
            m.fit(Xtr, y_log[tr_idx])
            preds_log[te_idx] = m.predict(Xte)
        oof_pred = inv_log(preds_log, y_shift)

        bc = bundle["bias_correction"]
        oof_pred_cal = oof_pred + (bc["intercept"] + bc["coef"] * oof_pred)
        residuals = y - oof_pred_cal

        return dict(X=X, y=y, oof_pred_cal=oof_pred_cal, residuals=residuals)
    except Exception as e:
        print(f"  (warning: could not recompute LOOCV from source data ({e}); "
              "falling back to a Normal-approximation using the stored LOOCV "
              "RMSE.)", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Single-point prediction
# --------------------------------------------------------------------------

def predict_point(bundle, x_dict, loocv_data=None, n_samples=10000, rng=None):
    rng = rng or np.random.default_rng(42)
    features = bundle["features"]
    x = np.array([[x_dict[f] for f in features]], dtype=float)
    x_scaled = bundle["scaler"].transform(x)

    model = bundle["model"]
    tree_logs = np.array([t.predict(x_scaled)[0] for t in model.estimators_])
    tree_log_std = float(tree_logs.std())

    y_shift = bundle["y_shift"]
    C = try_load_common()
    log_transform, inv_log = get_transforms(C)

    tree_preds = inv_log(tree_logs, y_shift)
    bc = bundle["bias_correction"]
    tree_preds_cal = tree_preds + (bc["intercept"] + bc["coef"] * tree_preds)
    point_pred = float(np.mean(tree_preds_cal))

    avg_tree_log_std = bundle["avg_tree_log_std"]
    scale_factor = max(1.0, tree_log_std / avg_tree_log_std) if avg_tree_log_std > 0 else 1.0

    if loocv_data is not None:
        residuals = loocv_data["residuals"]
        sampled_resid = rng.choice(residuals, size=n_samples, replace=True)
        mode = "empirical LOOCV residuals"
    else:
        rmse = bundle["loocv_metrics_calibrated"]["rmse"]
        sampled_resid = rng.normal(loc=0.0, scale=rmse, size=n_samples)
        mode = "Normal approximation (source data unavailable)"

    combined = point_pred + sampled_resid * scale_factor

    percentiles = {p: float(np.percentile(combined, p)) for p in [5, 10, 50, 70, 90, 95]}
    return dict(
        point_pred=point_pred,
        tree_log_std=tree_log_std,
        scale_factor=scale_factor,
        percentiles=percentiles,
        mode=mode,
    )


# --------------------------------------------------------------------------
# Risk categorisation
# --------------------------------------------------------------------------

def categorize_risk(p50, p95, hist_y=None):
    if hist_y is not None and len(hist_y) > 5:
        low_cut = float(np.percentile(hist_y, 33))
        high_cut = float(np.percentile(hist_y, 67))
        tail_cut = float(np.percentile(hist_y, 90))
    else:
        # Reasonable defaults if historical data isn't available to derive
        # data-driven cut points from.
        low_cut, high_cut, tail_cut = 15.0, 40.0, 80.0

    if p50 <= low_cut:
        category = "Low"
    elif p50 <= high_cut:
        category = "Medium"
    else:
        category = "High"

    # Bump the category up if the tail (P95) is unusually extreme, even
    # when the median looks moderate.
    if p95 > tail_cut:
        order = ["Low", "Medium", "High"]
        idx = min(order.index(category) + 1, len(order) - 1)
        category = order[idx]

    return category, dict(low_cut=low_cut, high_cut=high_cut, tail_cut=tail_cut)


# --------------------------------------------------------------------------
# Calibration check
# --------------------------------------------------------------------------

def calibration_check(bundle, loocv_data):
    if loocv_data is None:
        print("\nCalibration check skipped (dam_common.py / training data not found).")
        return

    X, y = loocv_data["X"], loocv_data["y"]
    oof_pred_cal = loocv_data["oof_pred_cal"]
    residuals = loocv_data["residuals"]
    n = len(y)

    model = bundle["model"]
    scaler = bundle["scaler"]
    avg_tree_log_std = bundle["avg_tree_log_std"]

    in_80 = 0  # inside nominal P10-P90
    in_90 = 0  # inside nominal P5-P95

    for i in range(n):
        # Leave-one-out: build project i's band from the OTHER 44 residuals,
        # so the check isn't trivially self-fulfilling.
        pool = np.delete(residuals, i)
        x_scaled = scaler.transform(X[i:i + 1])
        tree_logs = np.array([t.predict(x_scaled)[0] for t in model.estimators_])
        tree_log_std = tree_logs.std()
        scale_factor = max(1.0, tree_log_std / avg_tree_log_std) if avg_tree_log_std > 0 else 1.0

        band_samples = oof_pred_cal[i] + pool * scale_factor
        p10, p90 = np.percentile(band_samples, [10, 90])
        p5, p95 = np.percentile(band_samples, [5, 95])

        if p10 <= y[i] <= p90:
            in_80 += 1
        if p5 <= y[i] <= p95:
            in_90 += 1

    cov80 = 100 * in_80 / n
    cov90 = 100 * in_90 / n

    if cov80 < 70 or cov90 < 80:
        verdict = "bands look too narrow - treat P95 as optimistic"
    elif cov80 > 95 and cov90 > 98:
        verdict = "bands are conservative (safe, a bit wide)"
    else:
        verdict = "bands look reasonably trustworthy"

    print(f"P10-P90 actual coverage: {cov80:.0f}% (nominal 80%)  |  "
          f"P5-P95 actual coverage: {cov90:.0f}% (nominal 90%)  ->  {verdict}")


# --------------------------------------------------------------------------
# Input gathering
# --------------------------------------------------------------------------

def gather_features_interactive(features):
    print("Enter values for each feature:")
    vals = {}
    for f in features:
        hint = FEATURE_HINTS.get(f, "")
        while True:
            raw = input(f"  {f} [{hint}]: ").strip()
            try:
                vals[f] = float(raw)
                break
            except ValueError:
                print("    please enter a numeric value.")
    return vals


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict P50/P70/P95 cost-overrun % and risk category "
                    "for a single dam project, using the trained RF bundle."
    )
    p.add_argument("--model", default=MODEL_PATH, help="Path to the trained bundle pickle")
    p.add_argument("--json", help="Path to a JSON file with feature values for the project")
    p.add_argument("--set", action="append", default=[], metavar="FEATURE=VALUE",
                    help="Set a single feature value; repeatable, e.g. --set cost_per_mw=120")
    p.add_argument("--samples", type=int, default=10000,
                    help="Number of Monte-Carlo resamples used to build the percentile distribution")
    p.add_argument("--skip-calibration", action="store_true",
                    help="Skip the (slower) historical LOOCV calibration recheck")
    return p.parse_args()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    bundle = load_bundle(args.model)
    features = bundle["features"]

    x_dict = {}
    if args.json:
        with open(args.json) as f:
            x_dict = json.load(f)
    for kv in args.set:
        if "=" not in kv:
            print(f"Ignoring malformed --set value (expected FEATURE=VALUE): {kv}", file=sys.stderr)
            continue
        k, v = kv.split("=", 1)
        x_dict[k] = float(v)

    missing = [f for f in features if f not in x_dict]
    if missing:
        if x_dict:
            print(f"Missing values for: {missing}\n")
        x_dict.update(gather_features_interactive(missing))

    C = try_load_common()
    loocv_data = None if args.skip_calibration else recompute_loocv(bundle, C)

    result = predict_point(bundle, x_dict, loocv_data=loocv_data, n_samples=args.samples)
    hist_y = loocv_data["y"] if loocv_data is not None else None
    category, cuts = categorize_risk(result["percentiles"][50], result["percentiles"][95], hist_y)

    p = result["percentiles"]

    print(f"\nP50 = {p[50]:+.1f}%")
    print(f"P70 = {p[70]:+.1f}%")
    print(f"P90 = {p[90]:+.1f}%")
    print(f"P95 = {p[95]:+.1f}%")
    print(f"Risk category: {category}")

    print(f"\nMedian expected overrun: {p[50]:.0f}%. "
          f"90% chance it stays below {p[90]:.0f}%. "
          f"Risk category: {category}.")

    calibration_check(bundle, loocv_data)


if __name__ == "__main__":
    main()