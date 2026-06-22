
import pandas as pd
import numpy as np
import re

# ── 0. LOAD ────────────────────────────────────────────────────────────────────
df = pd.read_csv("C:\\Users\\User\\Downloads\\HEP_DATA.csv", encoding="utf-8", low_memory=False,index_col=0)
print(f"Loaded {len(df)} rows × {len(df.columns)} columns")

df = df.drop(columns=[c for c in df.columns if 'Unnamed' in str(c)])

df = df.map(lambda x: x.strip() if isinstance(x, str) else x)


NUMERIC_COLS = [
    "installed_cap", "dam_height", "initial_cost", "final_cost",
    "planned_dur", "actual_dur", "rainfall", "forest_area_div",
    "dist_road", "tunnel length", "elevation", "damlength",
    "transmission line length cktkm", "landowner_displaced"
]

def clean_numeric(val):
    """Strip asterisks/stars, coerce to float, return NaN for non-parseable."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace("*", "").replace(",", "").strip()
    # Handle entries like "0-LA" → take the numeric part
    s = re.sub(r"[^0-9.\-].*$", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan

for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# ── 3. DERIVED TARGET / FEATURE: COST OVERRUN & DURATION OVERRUN ───────────--
df["cost_overrun"]     = ((df["final_cost"] - df["initial_cost"]) / df["initial_cost"])*100   
df["duration_overrun"] = ((df["actual_dur"]  - df["planned_dur"])/ df["planned_dur"])*100   

# ── 4. GEOLOGICAL / SOCIAL PROBLEM COLUMN ──────────────────────────────────--

if "geological/social_prob" in df.columns:
    raw = df["geological/social_prob"].fillna("no").str.lower()

  
    PROB_CODES = {
        "landslide":            1,
        "sz":                   2,      # shear zone
        "ff":                   4,      # forest fire
        "wi":                   8,      # water ingress
        "wateringress":         8,      # same as wi, merge
        "rockburst":            16,
        "penstockburst":        32,
        "labour":               64,
        "cloudburst":           128,
        "flashflood":           256,
        "silterosion":          512,
        "fundstop":             1024,
        "forcemajeur":          2048,
        "forcemajure":          2048,   # typo variant, same code
        "shivalik":             4096,
        "suspended":            8192,
        "foundationslump":      16384,
        "delays":               32768,
        "reservoirfloor":       65536,
        "rollingboulder":       131072,
        "poorlyconsolidatedrock": 262144,
        "stresstransition":     524288,
    }

    def encode_prob(val):
        if val == "no":
            return 0
        code = 0
        for keyword, bitmask in PROB_CODES.items():
            if keyword in val:
                code |= bitmask  
        return code

    df["geo_prob_encoded"] = raw.apply(encode_prob)

    # Drop all the individual prob_ columns
    prob_cols = [c for c in df.columns if c.startswith("prob_") or c == "has_geo_prob"]
    df.drop(columns=prob_cols, inplace=True)

    print(f"[geo_prob] encoded into single column 'geo_prob_encoded' (bitmask)")
    print(df[["geological/social_prob", "geo_prob_encoded"]].drop_duplicates().sort_values("geo_prob_encoded").to_string())

# ── 5. CONTRACT TYPE 

# ── CONTRACT TYPE → NUMERIC CODE ────────────────────────────────────────────
# Single types: EPC=1, IR=2, SPLIT=3, BILATERAL=4, BOOT=5, MP=6,
#               DEPARTMENTAL=7, TURNKEY=8
# Combinations: digits concatenated e.g. IR+EPC → 21, IR+SPLIT → 23
# Find the contract type column
contract_col = None
for c in df.columns:
    if "contract" in c.lower() and "retend" not in c.lower():
        contract_col = c
        break
CONTRACT_CODES = {
    "EPC":          1,
    "IR":           2,
    "SPLIT":        3,
    "BILATERAL":    4,
    "BOOT":         5,
    "MP":           6,
    "DEPARTMENTAL": 7,
    "TURNKEY":      8,
}

if contract_col:
    def encode_contract(val):
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        # split on + or / or space
        parts = re.split(r"[+/\s]+", str(val).strip().upper())
        codes = []
        for part in parts:
            part = part.strip()
            if part in CONTRACT_CODES:
                codes.append(str(CONTRACT_CODES[part]))
        if not codes:
            return np.nan
        # sort so IR+EPC and EPC+IR give the same code
        return int("".join(sorted(codes)))

    df["contract_encoded"] = df[contract_col].apply(encode_contract)

    # drop the OHE columns and hybrid flag if they were already created
    ohe_cols = [c for c in df.columns if c.startswith("contract_") 
                and c != "contract_encoded"]
    df.drop(columns=ohe_cols, inplace=True, errors="ignore")

    print(f"[contract_type] encoded into single column 'contract_encoded'")
    print(df[[contract_col, "contract_encoded"]].drop_duplicates()
                                                .sort_values("contract_encoded")
                                                .to_string())
# ── 6. CONTRACT RETENDERED → ORDINAL ────────────────────────────────────────--

retend_col = None
for c in df.columns:
    if "retend" in c.lower():
        retend_col = c
        break

if retend_col:
    RETEND_MAP = {
        "NO": 0, "NO-LD": 1, "YES": 2,
        "NA": np.nan, "": np.nan
    }
    df["retendered_encoded"] = (
        df[retend_col].fillna("NA")
                      .str.upper()
                      .str.strip()
                      .map(RETEND_MAP)
    )
    print(f"[retendered] ordinal encoded: NO=0, NO-LD=1, YES=2")

# ── 7. SEISMIC ZONE → ORDINAL ───────────────────────────────────────────────--
SEISMIC_MAP = {
    "II":   1.0,
    "III":  2.0,
    "IV":   3.0,
    "IV/V": 3.5,   # boundary zone
    "V":    4.0,
}
if "seismic_zone" in df.columns:
    df["seismic_encoded"] = (
        df["seismic_zone"].fillna("").str.upper().str.strip().map(SEISMIC_MAP)
    )
    print(f"[seismic_zone] ordinal encoded: II=1 … V=4, IV/V=3.5")

# ── 8. CATEGORY → ORDINAL ───────────────────────────────────────────────────--
CATEGORY_MAP = {"small": 1, "large": 2, "mega": 3}
if "category" in df.columns:
    df["category_encoded"] = (
        df["category"].fillna("").str.lower().str.strip().map(CATEGORY_MAP)
    )
    print(f"[category] ordinal encoded: small=1, large=2, mega=3")

# ── 9. STATE → BINARY ───────────────────────────────────────────────────────--
if "state" in df.columns:
    df["state_encoded"] = (
        df["state"].str.lower().str.strip() == "uttarakhand"
    ).astype(int)
    # 0 = Himachal Pradesh, 1 = Uttarakhand
    print(f"[state] binary: 0=Himachal Pradesh, 1=Uttarakhand")

# ── 10. FUNDER → FUNDER TYPE FLAGS ──────────────────────────────────────────--
#
## ── FUNDER → LABEL ENCODING (single column) ─────────────────────────────────
if "funder" in df.columns:
    funder_clean = df["funder"].fillna("UNKNOWN").str.upper().str.strip()

    # Define label order (loosely by funding type: multilateral → govt → psu → private → unknown)
    funder_order = [
        # Multilateral / international
        "WB", "ADB", "IFC", "JICA",
        # Central government
        "CG", "CG+NHPC", "CG+SG", "CG+SG+USSR", "CG+DEBT",
        # State government
        "SG", "SG+PFC",
        # PSU / public financial institutions
        "NTPC", "NHPC", "SJVN", "PFC", "PFC+SJVN", "THDC", "PSU",
        "ADB/NTPC", "IFC+ADB+ICB", "IFC+ADHP",
        # Consortium / mixed
        "CONSORTIUM", "RENEWJAL", "UJVN", "EPPL+ICB", "LANCO+BANKS",
        # Private
        "PSC", "GMR", "BHILWARA",
        # Unknown / other
        "UNKNOWN",
    ]

    # Assign integer labels (1-indexed; 0 reserved for truly unseen values)
    funder_label_map = {name: idx + 1 for idx, name in enumerate(funder_order)}

    df["funder_encoded"] = funder_clean.map(funder_label_map).fillna(0).astype(int)

    print(f"[funder] label encoded into single column (0=unseen, 1–{len(funder_order)}=known)")
    print(df[["funder", "funder_encoded"]].drop_duplicates().sort_values("funder_encoded").to_string())

# ── 11. DROP RAW CATEGORICAL COLUMNS (keep originals for reference) ──────────
# We keep the original columns so you can inspect them; drop in modeling step.
COLS_TO_DROP_FOR_ML = [
    "geological/social_prob", contract_col, retend_col,
    "seismic_zone", "category", "state", "funder",
    "contract_type_clean",
    "project_name",   # identifier, not a feature
    "source",         # metadata
]
COLS_TO_DROP_FOR_ML = [c for c in COLS_TO_DROP_FOR_ML if c and c in df.columns]

df_ml = df.drop(columns=COLS_TO_DROP_FOR_ML)

# ── 12. HANDLE REMAINING NA VALUES ──────────────────────────────────────────--
# For numeric columns: fill with column median (safe for skewed distributions)
numeric_df_cols = df_ml.select_dtypes(include=[np.number]).columns
na_before = df_ml[numeric_df_cols].isna().sum().sum()

df_ml[numeric_df_cols] = df_ml[numeric_df_cols].fillna(
    df_ml[numeric_df_cols].median()
)
na_after = df_ml.isna().sum().sum()
print(f"\n[NA handling] filled {na_before} numeric NAs with column medians")
print(f"             remaining NAs: {na_after}")

# Drop any remaining all-NaN or non-numeric cols
df_ml = df_ml.select_dtypes(include=[np.number])

# ── 13. SAVE OUTPUT ─────────────────────────────────────────────────────────--
out_path = "DA\hydropower_ml_ready.csv" 
df_ml.to_csv(out_path, index=False)

print(f"   Shape: {df_ml.shape[0]} rows x {df_ml.shape[1]} columns")

# ── 14. FEATURE SUMMARY ─────────────────────────────────────────────────────--
print("\n Final feature list")
for col in df_ml.columns:
    n_miss = df_ml[col].isna().sum()
    print(f"  {col:<45}  dtype={df_ml[col].dtype}  NaN={n_miss}")