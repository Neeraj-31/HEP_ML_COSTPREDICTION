import pickle
import numpy as np
import pandas as pd
from scipy.stats import skew

import dam_common as C

N_SIMS = 10_000
MODEL_PATH = "dam_best_rf.pkl"
OUT_CSV = "dam_monte_carlo_samples.csv"
OUT_PLOT = "dam_monte_carlo_distribution.png"

# -- Default / fallback values, used to pre-fill the interactive prompts ----
# (kept only as sensible starting points; the user is asked to confirm or
# override every value at runtime - see get_new_project() below)
DEFAULTS = {
    "cost_per_mw":                 8.5,     # Rs Cr per MW, planning estimate
    "initial_cost":                10000.0, # Rs Cr, estimated/sanctioned cost
    "actual_dur":                  6.0,     # years, planned/expected duration
    "R30mm_total":                 45.0,    # count of >30mm rainfall days/yr
    "schedule_overrun_pct":        15.0,    # % planned schedule slip assumed
    "dist_to_nearest_pilgrim_km":  12.0,    # km, site-fixed
    "R100mm_total":                8.0,     # count of >100mm rainfall days/yr
    "glof_composite_score":        0.35,    # site-fixed GLOF risk score
    "steep_frac_25":               0.40,    # site-fixed, fraction slope >25 deg
}

# Human-readable prompts + units shown to the user for each feature.
FEATURE_PROMPTS = {
    "cost_per_mw":                "Cost per MW (Rs Cr/MW)",
    "initial_cost":               "Initial / sanctioned cost (Rs Cr)",
    "actual_dur":                 "Planned/expected duration (years)",
    "R30mm_total":                "Count of >30mm rainfall days per year",
    "schedule_overrun_pct":       "Assumed schedule overrun (%)",
    "dist_to_nearest_pilgrim_km": "Distance to nearest pilgrim site (km)",
    "R100mm_total":               "Count of >100mm rainfall days per year",
    "glof_composite_score":       "GLOF composite risk score (0-1)",
    "steep_frac_25":              "Fraction of catchment with slope >25 deg (0-1)",
}

# Which of the above are genuinely uncertain pre-construction (sampled every
# simulation) vs fixed site/planning facts (held constant every simulation).
# Distribution family for each uncertain feature is auto-detected from the
# historical data's skewness (see fit_feature_distributions below) -
# this only controls WHICH features get randomized.
UNCERTAIN_FEATURES = [
    "R30mm_total",            # rainfall -> naturally uncertain, right-skewed
    "R100mm_total",           # rainfall -> naturally uncertain, right-skewed
    "actual_dur",             # construction delay is not known in advance
    "schedule_overrun_pct",   # ditto -- schedule slip is a risk, not a fact
]
# Everything else (cost_per_mw, initial_cost, the two site-geography
# features, glof score) is treated as a known/fixed input.


# -- 0. Interactive input for the new project -------------------------------
def get_float(prompt_text, default_value):
    """Ask the user for a float, falling back to default_value on blank
    input, and re-asking on invalid (non-numeric) input."""
    while True:
        raw = input(f"  {prompt_text} [default={default_value}]: ").strip()
        if raw == "":
            return default_value
        try:
            return float(raw)
        except ValueError:
            print("    Please enter a numeric value (or press Enter for the default).")


def get_new_project(feature_order):
    """Interactively collect baseline values for every feature the model
    expects. feature_order is the exact ordered list of feature names the
    trained model was built with."""
    print("\nEnter the new project's planning-stage values.")
    print("Press Enter on any line to accept the default shown in brackets.\n")

    project = {}
    for feat in feature_order:
        label = FEATURE_PROMPTS.get(feat, feat)
        default_value = DEFAULTS.get(feat, 0.0)
        project[feat] = get_float(label, default_value)
    return project


# -- 1. Auto-fit a distribution shape for each uncertain feature ------------
# NOTE: with only ~45 historical projects, absolute min/max are dominated by
# single outliers. We use robust 5th/95th percentiles instead of min/max,
# and shift them ADDITIVELY onto your project's baseline (baseline + typical
# deviation) rather than scaling MULTIPLICATIVELY (baseline * historical
# ratio) - multiplicative scaling blows up whenever your baseline is far
# from the historical mean, producing absurd tails (e.g. an upper bound 15x
# the baseline from one extreme-rainfall project).
def fit_feature_distributions(df, uncertain_features, lower_pct=5, upper_pct=95):
    dists = {}
    for feat in uncertain_features:
        col = df[feat].values.astype(float)
        mean_, std_ = col.mean(), col.std()
        p_lo, p_hi = np.percentile(col, [lower_pct, upper_pct])
        sk = skew(col)
        cv = std_ / mean_ if mean_ != 0 else std_
        non_negative = col.min() >= 0  # does this feature have a natural floor at 0?

        if sk > 1.0 and col.min() > 0:
            dist_type = "lognormal"
        elif abs(sk) <= 0.5:
            dist_type = "normal"
        else:
            dist_type = "triangular"

        dists[feat] = {
            "type": dist_type,
            "cv": abs(cv),
            "dev_lo": p_lo - mean_,   # additive deviation, robust to outliers
            "dev_hi": p_hi - mean_,
            "non_negative": non_negative,
            "skew": sk,
        }
        print(f"  {feat:<26} skew={sk:+.2f}  CV={cv:.2f}  "
              f"p{lower_pct}/p{upper_pct} dev=({p_lo - mean_:+.1f}, {p_hi - mean_:+.1f})  -> {dist_type}")
    return dists


def sample_feature(baseline, spec, n_sims, rng):
    """Draw n_sims samples for one feature, centered on `baseline`."""
    dist_type = spec["type"]

    if dist_type == "normal":
        scale = abs(baseline) * spec["cv"] if baseline != 0 else spec["cv"]
        samples = rng.normal(loc=baseline, scale=max(scale, 1e-6), size=n_sims)

    elif dist_type == "lognormal":
        cv = spec["cv"]
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(max(baseline, 1e-6)) - sigma**2 / 2
        samples = rng.lognormal(mean=mu, sigma=sigma, size=n_sims)

    else:  # triangular -- additive, percentile-based bounds (see note above)
        lo = baseline + spec["dev_lo"]
        hi = baseline + spec["dev_hi"]
        if spec["non_negative"]:
            lo = max(lo, 0.0)
        lo, hi = min(lo, hi), max(lo, hi)
        mode = min(max(baseline, lo), hi)  # clip mode into [lo, hi]
        if lo == hi:
            samples = np.full(n_sims, baseline)
        else:
            samples = rng.triangular(left=lo, mode=mode, right=hi, size=n_sims)

    if spec["non_negative"]:
        samples = np.clip(samples, 0.0, None)
    return samples


# -- 2. Load trained model + historical data -----------------------------
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
model, scaler, y_shift, features = (
    bundle["model"], bundle["scaler"], bundle["y_shift"], bundle["features"]
)

df_hist = C.load_cleaned()

print(f"Loaded model expecting {len(features)} features: {features}")

# -- 3. Collect the new project's baseline values from the user ---------
NEW_PROJECT = get_new_project(features)

missing = [f for f in features if f not in NEW_PROJECT]
if missing:
    raise ValueError(f"NEW_PROJECT is missing values for: {missing}")

print(f"\nFitting uncertainty distributions from historical data "
      f"({len(df_hist)} projects):")
feature_dists = fit_feature_distributions(df_hist, UNCERTAIN_FEATURES)

# -- 4. Deterministic point prediction (single "computer predicts" value) --
x_point = np.array([[NEW_PROJECT[f] for f in features]])
x_point_s = scaler.transform(x_point)
point_pred_log = model.predict(x_point_s)[0]
point_pred_pct = C.inv_log(np.array([point_pred_log]), y_shift)[0]
print(f"\nPoint ML prediction (no uncertainty): "
      f"Expected Cost Overrun = {point_pred_pct:.1f}%")

# -- 5. Monte Carlo sampling --------------------------------------------
rng = np.random.default_rng(C.RANDOM_SEED)
X_sim = np.zeros((N_SIMS, len(features)))

for j, feat in enumerate(features):
    baseline = NEW_PROJECT[feat]
    if feat in UNCERTAIN_FEATURES:
        X_sim[:, j] = sample_feature(baseline, feature_dists[feat], N_SIMS, rng)
    else:
        X_sim[:, j] = baseline  # fixed input, same every simulation

X_sim_s = scaler.transform(X_sim)
pred_log_sim = model.predict(X_sim_s)
overrun_pct_sim = C.inv_log(pred_log_sim, y_shift)

# -- 6. Convert to absolute cost using the project's estimated cost --------
estimated_cost_cr = NEW_PROJECT["initial_cost"]
predicted_cost_cr = estimated_cost_cr * (1 + overrun_pct_sim / 100.0)

# -- 7. Risk-adjusted summary --------------------------------------------
levels = [10, 50, 80, 90, 95]
print(f"\n{'='*60}")
print(f"Monte Carlo Risk Summary  (n={N_SIMS:,} simulations)")
print(f"{'='*60}")
print(f"Estimated Cost (baseline)     : Rs {estimated_cost_cr:,.0f} Cr")
print(f"Mean predicted Cost Overrun   : {overrun_pct_sim.mean():.1f}%  "
      f"(std={overrun_pct_sim.std():.1f}%)")
print(f"\n{'Probability Level':<20}{'Cost Overrun %':<18}{'Predicted Cost (Rs Cr)'}")
summary_rows = []
for p in levels:
    ov = np.percentile(overrun_pct_sim, p)
    cost = np.percentile(predicted_cost_cr, p)
    print(f"P{p:<19}{ov:<18.1f}{cost:,.0f}")
    summary_rows.append({"Probability_Level": f"P{p}",
                          "Cost_Overrun_pct": ov,
                          "Predicted_Cost_Cr": cost})
print(f"{'='*60}\n")

pd.DataFrame(summary_rows).to_csv("dam_monte_carlo_summary.csv", index=False)
print("Saved: dam_monte_carlo_summary.csv")

pd.DataFrame({
    "cost_overrun_pct": overrun_pct_sim,
    "predicted_cost_cr": predicted_cost_cr,
}).to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")

# -- 8. Plot -------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Monte Carlo Risk Simulation - Cost Overrun", fontsize=14, fontweight="bold")

axes[0].hist(overrun_pct_sim, bins=60, color="steelblue", edgecolor="k", alpha=0.75)
for p in [50, 80, 90]:
    v = np.percentile(overrun_pct_sim, p)
    axes[0].axvline(v, color="red", ls="--", lw=1.2)
    axes[0].text(v, axes[0].get_ylim()[1] * 0.92, f"P{p}", rotation=90,
                 va="top", ha="right", fontsize=8, color="red")
axes[0].set_xlabel("Predicted Cost Overrun (%)")
axes[0].set_ylabel("Simulation count")
axes[0].set_title("Cost Overrun % distribution")

axes[1].hist(predicted_cost_cr, bins=60, color="darkorange", edgecolor="k", alpha=0.75)
for p in [50, 80, 90]:
    v = np.percentile(predicted_cost_cr, p)
    axes[1].axvline(v, color="red", ls="--", lw=1.2)
    axes[1].text(v, axes[1].get_ylim()[1] * 0.92, f"P{p}", rotation=90,
                 va="top", ha="right", fontsize=8, color="red")
axes[1].axvline(estimated_cost_cr, color="green", ls="-", lw=1.5, label="Estimated cost")
axes[1].set_xlabel("Predicted Actual Cost (Rs Cr)")
axes[1].set_ylabel("Simulation count")
axes[1].set_title("Predicted Cost distribution")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PLOT}")