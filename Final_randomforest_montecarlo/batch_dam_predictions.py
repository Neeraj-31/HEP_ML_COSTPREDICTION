import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for batch plot generation
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = r"laststagemodels\dam_best_rf_v2.pkl"
CSV_PATH = r"C:\Users\User\.vscode\HEP_ML\dam_ml_ready_cleaned.csv"
OUTPUT_DIR = "cost_overrun_distributions"



def log_transform(y, shift):
    return np.log(np.asarray(y, dtype=float) + shift)


def inv_log(y_log, shift):
    return np.exp(np.asarray(y_log, dtype=float)) - shift


def load_bundle(path):
    import pickle
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
        combined_samples=combined,
    )


def categorize_risk(p50, p95, hist_y=None):
    if hist_y is not None and len(hist_y) > 5:
        low_cut = float(np.percentile(hist_y, 33))
        high_cut = float(np.percentile(hist_y, 67))
        tail_cut = float(np.percentile(hist_y, 90))
    else:
        low_cut, high_cut, tail_cut = 15.0, 40.0, 80.0

    if p50 <= low_cut:
        category = "Low"
    elif p50 <= high_cut:
        category = "Medium"
    else:
        category = "High"

    if p95 > tail_cut:
        order = ["Low", "Medium", "High"]
        idx = min(order.index(category) + 1, len(order) - 1)
        category = order[idx]

    return category, dict(low_cut=low_cut, high_cut=high_cut, tail_cut=tail_cut)


# --------------------------------------------------------------------------
# Batch-specific plotting: one KDE distribution PNG per project
# --------------------------------------------------------------------------

def sanitize_filename(name, fallback):
    name = str(name).strip() if name and str(name).strip() else fallback
    name = re.sub(r"[^\w\-. ]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:120]  # keep filenames sane on Windows


def plot_project_distribution(project_label, result, category, cuts, outpath):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    combined_samples = result["combined_samples"]
    p = result["percentiles"]

    sns.kdeplot(combined_samples, fill=True, color="#2ca02c", alpha=0.3,
                linewidth=2.5, label="Monte Carlo Risk Distribution")

    plt.axvline(result["median_pred"], color="blue", linestyle="-", linewidth=2,
                label=f"Median (P50): {result['median_pred']:+.1f}%")
    plt.axvline(result["mean_pred"], color="purple", linestyle=":", linewidth=2,
                label=f"Mean Expected: {result['mean_pred']:+.1f}%")
    plt.axvline(p[70], color="orange", linestyle="--", linewidth=1.5,
                label=f"P70 Bound: {p[70]:+.1f}%")
    plt.axvline(p[90], color="darkorange", linestyle="--", linewidth=2,
                label=f"P90 Bound: {p[90]:+.1f}%")
    plt.axvline(p[95], color="red", linestyle="-.", linewidth=2,
                label=f"P95 Tail Risk: {p[95]:+.1f}%")

    ax = plt.gca()
    xlim = ax.get_xlim()
    plt.axvspan(xlim[0], cuts["low_cut"], color="green", alpha=0.04,
                label=f"Low Risk Tier (<={cuts['low_cut']:.1f}%)")
    plt.axvspan(cuts["low_cut"], cuts["high_cut"], color="yellow", alpha=0.04,
                label="Medium Risk Tier")
    plt.axvspan(cuts["high_cut"], xlim[1], color="red", alpha=0.04,
                label=f"High Risk Tier (>{cuts['high_cut']:.1f}%)")
    plt.xlim(xlim)

    plt.xlabel("Predicted Cost Overrun (%)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title(f"{project_label}\nCost Overrun Risk Profile (Assigned: {category} Risk)\n"
              f"[Forest Variance Spread Scaling: {result['scale_factor']:.2f}x]",
              fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()

    plt.savefig(outpath, dpi=300)
    plt.close()


# --------------------------------------------------------------------------
# Main batch driver
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a unique cost-overrun risk distribution PNG "
                    "for every project listed in a CSV file."
    )
    p.add_argument("--csv", default=CSV_PATH, help="Path to the projects CSV")
    p.add_argument("--model", default=MODEL_PATH, help="Path to the trained bundle pickle")
    p.add_argument("--outdir", default=OUTPUT_DIR, help="Folder to save the distribution PNGs into")
    p.add_argument("--samples", type=int, default=10000,
                    help="Number of Monte-Carlo resamples per project")
    p.add_argument("--skip-calibration", action="store_true",
                    help="Skip recomputing honest LOOCV residuals (uses Normal approximation instead)")
    p.add_argument("--name-col", default="project_name",
                    help="Column to use for naming each output PNG (default: project_name)")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading model bundle: {args.model}")
    bundle = load_bundle(args.model)
    features = bundle["features"]

    print(f"Loading projects CSV: {args.csv}")
    df = pd.read_csv(args.csv)

    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        print(f"ERROR: CSV is missing required feature columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    C = try_load_common()
    loocv_data = None if args.skip_calibration else recompute_loocv(bundle, C)
    hist_y = loocv_data["y"] if loocv_data is not None else None

    rng = np.random.default_rng(42)
    summary_rows = []
    used_filenames = set()

    for i, row in df.iterrows():
        raw_label = row[args.name_col] if args.name_col in df.columns else None
        label = str(raw_label).strip() if pd.notna(raw_label) and str(raw_label).strip() else f"project_{i+1}"

        feature_row = row[features]
        if feature_row.isna().any():
            bad = list(feature_row[feature_row.isna()].index)
            print(f"  Skipping '{label}' (row {i}): missing values for {bad}")
            continue

        x_dict = {f: float(row[f]) for f in features}

        result = predict_point(bundle, x_dict, loocv_data=loocv_data,
                                n_samples=args.samples, rng=rng)
        category, cuts = categorize_risk(result["median_pred"], result["percentiles"][95], hist_y)

        fname = sanitize_filename(label, f"project_{i+1}") + ".png"
        # guard against duplicate project names overwriting each other
        base_fname = fname
        n_dupe = 1
        while fname in used_filenames:
            n_dupe += 1
            fname = base_fname.replace(".png", f"_{n_dupe}.png")
        used_filenames.add(fname)

        outpath = os.path.join(args.outdir, fname)
        plot_project_distribution(label, result, category, cuts, outpath)

        p = result["percentiles"]
        print(f"  [{i+1}/{len(df)}] {label}: median={result['median_pred']:+.1f}%  "
              f"P90={p[90]:+.1f}%  P95={p[95]:+.1f}%  risk={category}  -> {fname}")

        summary_rows.append({
            args.name_col: label,
            "mean_pred": result["mean_pred"],
            "median_pred": result["median_pred"],
            "p5": p[5], "p10": p[10], "p70": p[70], "p90": p[90], "p95": p[95],
            "risk_category": category,
            "scale_factor": result["scale_factor"],
            "mode": result["mode"],
            "plot_file": fname,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, "_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nDone. {len(summary_rows)} distribution plots saved to: {args.outdir}")
    print(f"Summary table saved to: {summary_path}")


if __name__ == "__main__":
    main()
