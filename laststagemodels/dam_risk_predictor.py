import argparse
import json
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "dam_best_rf_v2.pkl"

def log_transform(y, shift):
    return np.log(np.asarray(y, dtype=float) + shift)


def inv_log(y_log, shift):
    return np.exp(np.asarray(y_log, dtype=float)) - shift


FEATURE_HINTS = {
    "cost_per_mw": "Cost per MW of installed capacity",
    "initial_cost": "Initial planned/sanctioned project cost",
    "R30mm_total": "Total count of days with >=30mm rainfall",
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
    try:
        import dam_common as C
        return C
    except Exception:
        return None


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

    mean_pred = float(np.mean(combined))
    median_pred = float(np.median(combined))
    percentiles = {p: float(np.percentile(combined, p)) for p in [5, 10, 50, 70, 90, 95]}
    
    return dict(
        point_pred=point_pred,
        mean_pred=mean_pred,
        median_pred=median_pred,
        tree_log_std=tree_log_std,
        scale_factor=scale_factor,
        percentiles=percentiles,
        mode=mode,
        combined_samples=combined
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
        description="Predict mean, median, and percentile cost-overrun % and risk category "
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
    category, cuts = categorize_risk(result["median_pred"], result["percentiles"][95], hist_y)
    generate_and_save_plots(
        bundle=bundle, 
        loocv_data=loocv_data, 
        result=result, 
        category=category, 
        cuts=cuts, 
        combined_samples=result["combined_samples"]
    )
    p = result["percentiles"]

    print(f"\nMean Overrun:   {result['mean_pred']:+.1f}%")
    print(f"Median (P50):   {result['median_pred']:+.1f}%")
    print(f"P70 Overrun:    {p[70]:+.1f}%")
    print(f"P90 Overrun:    {p[90]:+.1f}%")
    print(f"P95 Overrun:    {p[95]:+.1f}%")
    print(f"Risk category:  {category}")

    print(f"\nSummary Analysis:")
    print(f"  - The expected average cost overrun is {result['mean_pred']:.1f}%.")
    print(f"  - The median expected overrun is {result['median_pred']:.0f}% (50% of scenarios fall below this).")
    print(f"  - There is a 90% chance that the overrun stays below {p[90]:.0f}%.")
    print(f"  - Risk Assessment Category: {category}.\n")

    calibration_check(bundle, loocv_data)

def generate_and_save_plots(bundle, loocv_data, result, category, cuts, combined_samples):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot the simulated distribution from the combined samples
    sns.kdeplot(combined_samples, fill=True, color="#2ca02c", alpha=0.3, linewidth=2.5, label='Monte Carlo Risk Distribution')
    
    # Add vertical indicators for key percentiles
    p = result['percentiles']
    plt.axvline(result['median_pred'], color='blue', linestyle='-', linewidth=2, label=f"Median (P50): {result['median_pred']:+.1f}%")
    plt.axvline(result['mean_pred'], color='purple', linestyle=':', linewidth=2, label=f"Mean Expected: {result['mean_pred']:+.1f}%")
    plt.axvline(p[70], color='orange', linestyle='--', linewidth=1.5, label=f"P70 Bound: {p[70]:+.1f}%")
    plt.axvline(p[90], color='darkorange', linestyle='--', linewidth=2, label=f"P90 Bound: {p[90]:+.1f}%")
    plt.axvline(p[95], color='red', linestyle='-.', linewidth=2, label=f"P95 Tail Risk: {p[95]:+.1f}%")
    
    # Shade background color risk categories based on historical cuts
    ax = plt.gca()
    xlim = ax.get_xlim()
    plt.axvspan(xlim[0], cuts['low_cut'], color='green', alpha=0.04, label=f"Low Risk Tier (<={cuts['low_cut']:.1f}%)")
    plt.axvspan(cuts['low_cut'], cuts['high_cut'], color='yellow', alpha=0.04, label='Medium Risk Tier')
    plt.axvspan(cuts['high_cut'], xlim[1], color='red', alpha=0.04, label=f"High Risk Tier (>{cuts['high_cut']:.1f}%)")
    plt.xlim(xlim)
    
    plt.xlabel('Predicted Cost Overrun (%)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title(f'Project-Specific Cost Overrun Risk Profile (Assigned: {category} Risk)\n'
              f'[Forest Variance Spread Scaling: {result["scale_factor"]:.2f}x]', fontsize=13, fontweight='bold')
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    plt.tight_layout()
    
    plt.savefig('cost_overrun_prob_dist.png', dpi=300)
    plt.close()
    print("-> Saved: cost_overrun_prob_dist.png")

    if loocv_data is None:
        print("-> Skipped predicted_vs_actual_scatter.png (historical LOOCV data unavailable).")
        return

    X, y = loocv_data["X"], loocv_data["y"]
    oof_pred_cal = loocv_data["oof_pred_cal"]
    residuals = loocv_data["residuals"]
    n = len(y)

    model = bundle["model"]
    scaler = bundle["scaler"]
    avg_tree_log_std = bundle["avg_tree_log_std"]

    p10_bars = np.zeros(n)
    p90_bars = np.zeros(n)
    
    # Reconstruct point-specific validation intervals for historical points
    for i in range(n):
        pool = np.delete(residuals, i)
        x_scaled = scaler.transform(X[i:i + 1])
        tree_logs = np.array([t.predict(x_scaled)[0] for t in model.estimators_])
        scale_factor = max(1.0, tree_logs.std() / avg_tree_log_std) if avg_tree_log_std > 0 else 1.0
        band_samples = oof_pred_cal[i] + pool * scale_factor
        p10_bars[i], p90_bars[i] = np.percentile(band_samples, [10, 90])

    plt.figure(figsize=(8, 7))
    yerr = [oof_pred_cal - p10_bars, p90_bars - oof_pred_cal]
    
    plt.errorbar(oof_pred_cal, y, yerr=yerr, fmt='o', color='#1f77b4', ecolor='gray', 
                 alpha=0.6, elinewidth=1, capsize=2, label='LOOCV Out-of-Fold Points (with P10-P90 Bands)')

    # Ideal match line
    min_val = min(y.min(), oof_pred_cal.min()) - 10
    max_val = max(y.max(), oof_pred_cal.max()) + 10
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5, label='Perfect Calibration (y = x)')

    plt.xlabel('Bias-Corrected Predicted Cost Overrun (%)', fontsize=12)
    plt.ylabel('Actual Cost Overrun (%)', fontsize=12)
    plt.title('Model Validation: Leave-One-Out Cross-Validation (LOOCV)\nPredicted vs. Actual Cost Overruns', fontsize=13, fontweight='bold')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    plt.savefig('predicted_vs_actual_scatter.png', dpi=300)
    plt.close()
    print("-> Saved: predicted_vs_actual_scatter.png")


if __name__ == "__main__":
    main()