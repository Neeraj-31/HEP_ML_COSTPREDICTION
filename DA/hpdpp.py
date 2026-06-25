import pandas as pd
import numpy as np
import re

# ── 0. LOAD ────────────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\User\Downloads\HEP_DATA.csv", encoding="utf-8", low_memory=False, index_col=0)
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
    s = re.sub(r"[^0-9.\-].*$", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan

for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# ── 3. DERIVED TARGET / FEATURE: COST OVERRUN & DURATION OVERRUN ───────────
df["cost_overrun"]     = ((df["final_cost"] - df["initial_cost"]) / df["initial_cost"]) * 100   
df["duration_overrun"] = ((df["actual_dur"]  - df["planned_dur"]) / df["planned_dur"]) * 100   

# ── 4. GEOLOGICAL / SOCIAL PROBLEM (ONE-HOT ENCODED + DILUTION PREVENTION) ─
if "geological/social_prob" in df.columns:
    raw_probs = df["geological/social_prob"].fillna("no").str.lower()
    raw_probs = raw_probs.str.replace('yes-', '').str.replace('yea-', '')
    
    # Create One-Hot Encoded columns
    prob_dummies = raw_probs.str.get_dummies(sep='+')
    prob_dummies.columns = ['geo_prob_' + c.strip() for c in prob_dummies.columns]
    
    if 'geo_prob_no' in prob_dummies.columns:
        prob_dummies = prob_dummies.drop(columns=['geo_prob_no'])
        
    # Prevent Feature Dilution: Group rare problems into an "other" category
    min_freq_prob = 2 
    rare_prob_cols = [c for c in prob_dummies.columns if prob_dummies[c].sum() < min_freq_prob]
    
    if rare_prob_cols:
        prob_dummies['geo_prob_other'] = prob_dummies[rare_prob_cols].max(axis=1)
        prob_dummies = prob_dummies.drop(columns=rare_prob_cols)

    df = pd.concat([df, prob_dummies], axis=1)
    print(f"[geo_prob] Multi-label encoded. Grouped {len(rare_prob_cols)} rare problems into 'geo_prob_other'.")

# ── 5. CONTRACT TYPE (ONE-HOT ENCODED + DILUTION PREVENTION) ───────────────
contract_col = None
for c in df.columns:
    if "contract" in c.lower() and "retend" not in c.lower():
        contract_col = c
        break

if contract_col:
    raw_contracts = df[contract_col].fillna("UNKNOWN").str.upper()
    raw_contracts = raw_contracts.str.replace(r'[/ ]+', '+', regex=True)
    
    contract_dummies = raw_contracts.str.get_dummies(sep='+')
    contract_dummies.columns = ['contract_' + c for c in contract_dummies.columns]
    
    # Prevent Feature Dilution: Group rare contracts
    min_freq_contract = 3
    rare_contract_cols = [c for c in contract_dummies.columns if contract_dummies[c].sum() < min_freq_contract]
    
    if rare_contract_cols:
        contract_dummies['contract_OTHER'] = contract_dummies[rare_contract_cols].max(axis=1)
        contract_dummies = contract_dummies.drop(columns=rare_contract_cols)
        
    df = pd.concat([df, contract_dummies], axis=1)
    print(f"[contract_type] Multi-label encoded. Grouped {len(rare_contract_cols)} rare contracts into 'contract_OTHER'.")

# ── 6. CONTRACT RETENDERED → ORDINAL ────────────────────────────────────────
retend_col = None
for c in df.columns:
    if "retend" in c.lower():
        retend_col = c
        break

if retend_col:
    RETEND_MAP = {"NO": 0, "NO-LD": 1, "YES": 2, "NA": np.nan, "": np.nan}
    df["retendered_encoded"] = df[retend_col].fillna("NA").str.upper().str.strip().map(RETEND_MAP)
    print(f"[retendered] ordinal encoded: NO=0, NO-LD=1, YES=2")

# ── 7. SEISMIC ZONE → ORDINAL ───────────────────────────────────────────────
SEISMIC_MAP = {"II": 2.0, "III": 3.0, "IV": 4.0, "IV/V": 4.5, "V": 5.0}
if "seismic_zone" in df.columns:
    df["seismic_encoded"] = df["seismic_zone"].fillna("").str.upper().str.strip().map(SEISMIC_MAP)
    print(f"[seismic_zone] ordinal encoded: II=2 … V=5")

# ── 8. CATEGORY → ORDINAL ───────────────────────────────────────────────────
CATEGORY_MAP = {"small": 1, "large": 2, "mega": 3}
if "category" in df.columns:
    df["category_encoded"] = df["category"].fillna("").str.lower().str.strip().map(CATEGORY_MAP)
    print(f"[category] ordinal encoded: small=1, large=2, mega=3")

# ── 9. STATE → BINARY ───────────────────────────────────────────────────────
if "state" in df.columns:
    df["state_encoded"] = (df["state"].str.lower().str.strip() == "uttarakhand").astype(int)
    print(f"[state] binary: 0=Himachal Pradesh, 1=Uttarakhand")

# ── 10. FUNDER → LABEL ENCODING ─────────────────────────────────────────────
if "funder" in df.columns:
    funder_clean = df["funder"].fillna("UNKNOWN").str.upper().str.strip()
    funder_order = [
        "WB", "ADB", "IFC", "JICA", "CG", "CG+NHPC", "CG+SG", "CG+SG+USSR", "CG+DEBT",
        "SG", "SG+PFC", "NTPC", "NHPC", "SJVN", "PFC", "PFC+SJVN", "THDC", "PSU",
        "ADB/NTPC", "IFC+ADB+ICB", "IFC+ADHP", "CONSORTIUM", "RENEWJAL", "UJVN", 
        "EPPL+ICB", "LANCO+BANKS", "PSC", "GMR", "BHILWARA", "UNKNOWN"
    ]
    funder_label_map = {name: idx + 1 for idx, name in enumerate(funder_order)}
    df["funder_encoded"] = funder_clean.map(funder_label_map).fillna(0).astype(int)
    print(f"[funder] label encoded into single column")

# ── 11. DROP RAW CATEGORICAL COLUMNS ────────────────────────────────────────
COLS_TO_DROP_FOR_ML = [
    "geological/social_prob", contract_col, retend_col,
    "seismic_zone", "category", "state", "funder",
    "contract_type_clean", "project_name", "source"
]
COLS_TO_DROP_FOR_ML = [c for c in COLS_TO_DROP_FOR_ML if c and c in df.columns]
df_ml = df.drop(columns=COLS_TO_DROP_FOR_ML)

# ── 12. HANDLE REMAINING NA VALUES ──────────────────────────────────────────
numeric_df_cols = df_ml.select_dtypes(include=[np.number]).columns
na_before = df_ml[numeric_df_cols].isna().sum().sum()

df_ml[numeric_df_cols] = df_ml[numeric_df_cols].fillna(df_ml[numeric_df_cols].median())
na_after = df_ml.isna().sum().sum()
print(f"\n[NA handling] filled {na_before} numeric NAs with column medians. Remaining NAs: {na_after}")

df_ml = df_ml.select_dtypes(include=[np.number])

# ── 13. SAVE OUTPUT ─────────────────────────────────────────────────────────
out_path = r"DA\hydropower_ml_readyv2.csv" 
df_ml.to_csv(out_path, index=False)
print(f"   Shape: {df_ml.shape[0]} rows x {df_ml.shape[1]} columns")