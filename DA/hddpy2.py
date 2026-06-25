import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── 0.  Paths ──────────────────────────────────────────────────────────────
RAW_PATH   = Path(r"C:\Users\User\Downloads\HEP_DATA.csv")
OUT_CSV    = Path("hep_prepared.csv")
OUT_SCALER = Path("hep_scaler.pkl")
OUT_REPORT = Path("hep_prep_report.txt")

report_lines = []
def log(msg=""):
    try:
        print(msg)
    except UnicodeEncodeError:
        # If Windows CP1252 chokes, swap the arrow out just for the terminal
        safe_msg = str(msg).replace("→", "->").replace("\u2192", "->")
        print(safe_msg)
    report_lines.append(str(msg))

df = pd.read_csv(RAW_PATH)
log(f"Raw shape: {df.shape}")
log(f"Columns  : {list(df.columns)}")

drop_always = ["Unnamed: 22", "Unnamed: 23", "Unnamed: 24", "source", "project_name"]
df.drop(columns=[c for c in drop_always if c in df.columns], inplace=True)
log(f"\nAfter dropping metadata cols: {df.shape}")


def to_float(val):
    if pd.isna(val):
        return np.nan
    try:
        return float(str(val).replace("*", "").replace(" ", "").strip())
    except ValueError:
        return np.nan

df["initial_cost"] = pd.to_numeric(df["initial_cost"], errors="coerce")
df["final_cost"]   = df["final_cost"].apply(to_float)

df["pct_cost_overrun"] = (
    (df["final_cost"] - df["initial_cost"]) / df["initial_cost"]
) * 100

n_before = len(df)
df.dropna(subset=["pct_cost_overrun"], inplace=True)
log(f"\nRows with valid target: {len(df)}  (dropped {n_before - len(df)} with missing cost data)")
log(f"Target stats (raw):\n{df['pct_cost_overrun'].describe().to_string()}")


Q1, Q3 = df["pct_cost_overrun"].quantile([0.25, 0.75])
IQR     = Q3 - Q1
extreme_lo = Q1 - 3 * IQR
extreme_hi = Q3 + 3 * IQR
mask_extreme = (df["pct_cost_overrun"] < extreme_lo) | (df["pct_cost_overrun"] > extreme_hi)
log(f"\nIQR outlier bounds (extreme +-3xIQR): [{extreme_lo:.1f}, {extreme_hi:.1f}]")
log(f"Extreme outlier projects removed    : {mask_extreme.sum()}")
log(f"  {df.loc[mask_extreme, 'project_name' if 'project_name' in df.columns else df.columns[0]].tolist() if 'project_name' in df.columns else df[mask_extreme].index.tolist()}")
df = df[~mask_extreme].copy()
log(f"Dataset size after outlier removal  : {len(df)}")
log(f"Target stats (post-clean):\n{df['pct_cost_overrun'].describe().to_string()}")


numeric_dirty = [
    "dam_height", "tunnel length", "elevation", "damlength",
    "transmission line length cktkm", "forest_area_div", "dist_road",
    "landowner_displaced",
]
for col in numeric_dirty:
    df[col] = df[col].apply(to_float)

rename_map = {
    "installed_cap"                  : "installed_cap_mw",
    "dam_height"                     : "dam_height_m",
    "planned_dur"                    : "planned_dur_yr",
    "actual_dur"                     : "actual_dur_yr",
    "tunnel length"                  : "tunnel_length_m",
    "elevation"                      : "elevation_m",
    "damlength"                      : "dam_length_m",
    "transmission line length cktkm" : "transmission_km",
    "forest_area_div"                : "forest_area_ha",
    "dist_road"                      : "dist_road_km",
    "landowner_displaced"            : "landowner_displaced",
    "contract type"                  : "contract_type",
    "contract retendered"            : "contract_retendered",
    "geological/social_prob"         : "geo_social_prob",
}
df.rename(columns=rename_map, inplace=True)

# ──  Feature engineering ──────────────────────────────────────────────────
#   Duration ratio — captures schedule performance (strong overrun proxy)
df["duration_ratio"] = df["actual_dur_yr"] / df["planned_dur_yr"].replace(0, np.nan)

#   Cost per MW — normalises scale across small/mega projects
df["cost_per_mw"] = df["initial_cost"] / df["installed_cap_mw"].replace(0, np.nan)

#   Tunnel length per MW — proxy for complexity / underground work
df["tunnel_per_mw"] = df["tunnel_length_m"] / df["installed_cap_mw"].replace(0, np.nan)

log("\n[Feature engineering] Added: duration_ratio, cost_per_mw, tunnel_per_mw")

# ──   Categorical encoding ──────────────────────────────────────────────────


df["state"] = df["state"].str.lower().str.strip()
df = pd.get_dummies(df, columns=["state"], prefix="state", drop_first=True)
log("[Encoding] state  OHE (himachal dropped as reference)")


seismic_map = {"IV": 1, "IV/V": 2, "V": 3}
df["seismic_zone"] = df["seismic_zone"].str.strip().map(seismic_map)
log("[Encoding] seismic_zone -> ordinal 1-3")

df.drop(columns=["category"], inplace=True)

df["geo_social_prob"] = df["geo_social_prob"].str.strip().str.lower()

geo_dummies = pd.get_dummies(df["geo_social_prob"], prefix="geo_prob", dtype=int)

df = pd.concat([df, geo_dummies], axis=1)
df.drop(columns=["geo_social_prob"], inplace=True)

log("[Encoding] geo_social_prob - one-hot encoded columns")

def parse_retendered(val):
    if pd.isna(val): return np.nan
    v = str(val).upper().replace("*", "").strip()
    if v.startswith("YES"):   return 1
    if v.startswith("NO"):    return 0
    return np.nan  # truly unknown

df["contract_retendered"] = df["contract_retendered"].apply(parse_retendered)
log("[Encoding] contract_retendered  binary 0/1")


def group_contract(val):
    if pd.isna(val): return "OTHER"
    v = str(val).upper().strip()
    if "EPC" in v:           return "EPC"
    if v in ("IR",):         return "IR"
    if "DEPARTMENTAL" in v:  return "DEPARTMENTAL"
    if "BOOT" in v:          return "BOOT"
    return "OTHER"

df["contract_type"] = df["contract_type"].apply(group_contract)
df = pd.get_dummies(df, columns=["contract_type"], prefix="ctype", drop_first=True)
log("[Encoding] contract_type -> grouped (EPC/IR/DEPARTMENTAL/BOOT/OTHER) then OHE")

# 7g.  FUNDER — binary institutional flags (more informative than 28-level OHE)
def funder_flags(series):
    s = series.fillna("").str.upper()
    flags = pd.DataFrame(index=series.index)
    flags["funder_CG"]       = s.str.contains("CG").astype(int)      # Central Govt
    flags["funder_SG"]       = s.str.contains("SG").astype(int)      # State Govt
    flags["funder_WB"]       = s.str.contains("WB").astype(int)      # World Bank
    flags["funder_ADB"]      = s.str.contains("ADB").astype(int)     # Asian Dev Bank
    flags["funder_IFC"]      = s.str.contains("IFC").astype(int)     # IFC
    flags["funder_private"]  = s.str.contains("PSC|GMR|BHILWARA|LANCO|ICB|GE").astype(int)
    flags["funder_PSU"]      = s.str.contains("NTPC|SJVN|NHPC|PSU|UJVN").astype(int)
    return flags

funder_df = funder_flags(df["funder"])
df = pd.concat([df, funder_df], axis=1)
df.drop(columns=["funder"], inplace=True)
log("[Encoding] funder → 7 binary institutional flags")

# ── 8.  Select & order final features ────────────────────────────────────────
# ── 8. Select & order final features ────────────────────────────────────────
TARGET = "pct_cost_overrun"

# Columns we keep as features
FEATURES = [
    # EVM / schedule
    "planned_dur_yr", "actual_dur_yr", "duration_ratio",
    # Scale / technical
    "installed_cap_mw", "initial_cost", "cost_per_mw",
    "dam_height_m", "dam_length_m", "tunnel_length_m", "tunnel_per_mw",
    "elevation_m", "transmission_km",
    # Environmental / risk
    "rainfall", "seismic_zone", "forest_area_ha", "dist_road_km",
    "landowner_displaced", # Removed the broken "has_geo_problem" here
    # Project type
    "category",
    # Contract
    "contract_retendered",
] + [c for c in df.columns if c.startswith("state_")] \
  + [c for c in df.columns if c.startswith("ctype_")] \
  + [c for c in df.columns if c.startswith("funder_")] \
  + [c for c in df.columns if c.startswith("geo_prob_")] # <-- Added this line!

# Keep only columns that exist after all transforms
FEATURES = [f for f in FEATURES if f in df.columns]
log(f"\nFinal feature list ({len(FEATURES)} features):")
for f in FEATURES:
    log(f"  {f}")

# ── 9.  Median imputation for remaining NaNs (MUST happen before scaling) ─────
df_model = df[FEATURES + [TARGET]].copy()
log("\n[Imputation] Missing values before:")
missing = df_model[FEATURES].isnull().sum()
log(missing[missing > 0].to_string())

# Impute using raw (unscaled) medians
for col in FEATURES:
    if df_model[col].isnull().any():
        med = df_model[col].median()
        df_model[col] = df_model[col].fillna(med)

log("\n[Imputation] All NaNs filled with column median")
assert not df_model[FEATURES].isnull().any().any(), "NaN still present after imputation!"
log("[Imputation] No NaN remaining")

# ── 10.  MinMax normalisation ─────────────────────────────────────────────────
#  Scale features only (NOT the target — we want interpretable predictions)
scaler = MinMaxScaler()
df_model[FEATURES] = scaler.fit_transform(df_model[FEATURES])

with open(OUT_SCALER, "wb") as f:
    pickle.dump({"scaler": scaler, "feature_names": FEATURES}, f)
log(f"\n[Scaling] MinMaxScaler fitted and saved → {OUT_SCALER}")

# ── 11.  Save ─────────────────────────────────────────────────────────────────
df_model.to_csv(OUT_CSV, index=False)
log(f"\n[Output] Saved prepared dataset → {OUT_CSV}")
log(f"         Shape: {df_model.shape}")
log(f"\nFinal target stats:")
log(df_model[TARGET].describe().to_string())

# ── 12.  Write report ──────────────────────────────────────────────────────────
with open(OUT_REPORT, "w",encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\nPrep report saved - {OUT_REPORT}")
print("\n Data preparation complete.")