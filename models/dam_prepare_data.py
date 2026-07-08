"""
dam_prepare_data.py
────────────────────
One-time cleaning step for the dam cost-overrun dataset.

What it does:
  1. Loads the raw 50-project dataset (all 47 engineered features).
  2. Drops the same 5 extreme-outlier projects that were already excluded
     from hep_prepared2.csv (Ranjit Sagar Dam, Tehri, Ramganga, Vyasi,
     Maneri Bhali II) — these have cost overruns of 750%-3400%, far beyond
     the rest of the distribution, and were removed for that dataset too.
  3. Median-imputes the small number of missing values (14 of 45 rows have
     at least one NaN, concentrated in 7 columns) using the whole-dataset
     median. NOTE: this is a practical simplification for a very small
     dataset (imputing before any train/test split) — it uses information
     from the full set, so treat cross-validated metrics downstream as
     mildly optimistic, not as a fully leakage-free estimate.
  4. Adds one domain-informed engineered feature: cost_per_mw
     (initial_cost / installed_cap), mirroring the feature that was most
     predictive in the HEP-specific model.
  5. Saves the result to dam_ml_ready_cleaned.csv — this is the file all
     four model scripts (dam_lasso.py, dam_rf.py, dam_xgb.py,
     dam_catboost.py) read from.

Run this once before running any of the model scripts:
    python dam_prepare_data.py
"""
import pandas as pd
from pathlib import Path

RAW_CSV     = Path("dam_ml_ready (1).csv")
RAW_CSV_ALT = Path("dam_ml_ready_raw.csv")   # fallback name
CLEANED_CSV = Path("dam_ml_ready_cleaned.csv")

# The 5 projects excluded from hep_prepared2.csv (matched by exact
# cost_overrun_pct value, since project names differ slightly in spelling
# between the two source files).
OUTLIER_COST_VALUES = [
    3365.947631,   # Ranjit Sagar Dam (Thein)
    1658.029106,   # tehri
    1002.269308,   # ramganga
    757.882491,    # vyasi
    2502.739726,   # Maneri Bhali Ii
]


def main():
    src = RAW_CSV if RAW_CSV.exists() else RAW_CSV_ALT
    if not src.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_CSV} or {RAW_CSV_ALT}. "
            "Place the raw dam_ml_ready csv in this directory."
        )
    df = pd.read_csv(src)
    print(f"Loaded {src}  shape={df.shape}")

    # -- 1. Drop the 5 outlier projects (match on target value) -------------
    mask = ~df["cost_overrun_pct"].round(6).isin(
        [round(v, 6) for v in OUTLIER_COST_VALUES]
    )
    removed = df.loc[~mask, ["project_name", "cost_overrun_pct"]]
    df = df[mask].reset_index(drop=True)
    print(f"Removed {len(removed)} outlier projects:")
    print(removed.to_string(index=False))

    # -- 2. Median-impute remaining NaNs -------------------------------------
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        print(f"\nMedian-imputing NaNs in: {nan_cols}")
        df[nan_cols] = df[nan_cols].fillna(df[nan_cols].median())
    assert df.isnull().sum().sum() == 0, "NaNs remain after imputation"

    # -- 3. Engineer cost_per_mw ----------------------------------------------
    df["cost_per_mw"] = df["initial_cost"] / df["installed_cap"]

    # -- 4. Save --------------------------------------------------------------
    df.to_csv(CLEANED_CSV, index=False)
    print(f"\nSaved cleaned dataset -> {CLEANED_CSV}  shape={df.shape}")
    print(f"Target range: [{df['cost_overrun_pct'].min():.1f}%, "
          f"{df['cost_overrun_pct'].max():.1f}%]  "
          f"mean={df['cost_overrun_pct'].mean():.1f}%  "
          f"std={df['cost_overrun_pct'].std():.1f}%")


if __name__ == "__main__":
    main()
