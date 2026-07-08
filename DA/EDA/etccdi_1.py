"""
INDICES COMPUTED (per dam, aggregated over construction window):
  PRCPTOT  Annual total precip on wet days (≥1mm) [Mean & Max]
  RX1day   Max 1-day precip [Mean & Max (Absolute Peak Cloudburst)]
  RX5day   Max 5-day running-sum precip [Mean & Max (Peak Deluge)]
  SDII     Simple Daily Intensity Index = Total Precip / Wet-day count
  R10mm    Annual count of days ≥10mm [Mean & Max]
  R20mm    Annual count of days ≥20mm [Mean & Max]
  R25mm    Annual count of days ≥25mm [Mean & Max]
  CDD      Max annual consecutive dry days (<1mm) [Mean & Max (Worst Drought)]
  CWD      Max annual consecutive wet days (≥1mm) [Mean & Max (Worst Persistent Rain)]
  R95p     Annual sum of precip on days > 95th pct of base-period wet days [Mean & Max]
  R99p     Annual sum of precip on days > 99th pct of base-period wet days [Mean & Max]

NON-STANDARD ADDITIONS (Flash-flood/Cloudburst proxies):
  R30mm    Annual count of days >30mm [Mean & Total Cumulative Shocks]
  R100mm   Annual count of days >100mm [Mean & Total Cumulative Shocks]
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data 
df_master = pd.read_csv(r'DA\dam_dataset_catchment_verified (2).csv', encoding='cp1252')
constr = pd.read_csv(r'DA\raw_imd_construction_window.csv', parse_dates=['date'])
base   = pd.read_csv(r'DA\raw_imd_base_period.csv', parse_dates=['date'])

print(f"Construction window rows: {len(constr):,} | Base period rows: {len(base):,}")


# 2. Base-period wet-day percentile thresholds (per dam)
def base_thresholds(group):
    wet = group.loc[group['precip_mm'] >= 1.0, 'precip_mm']
    if wet.empty:
        return pd.Series({'p95_thresh': np.nan, 'p99_thresh': np.nan})
    return pd.Series({
        'p95_thresh': wet.quantile(0.95),
        'p99_thresh': wet.quantile(0.99)
    })

print("Computing base-period percentile thresholds...")
thresholds = (base.groupby('project_name')
                   .apply(base_thresholds)
                   .reset_index())


# 3. Helper function for continuous spells
def max_consecutive(bool_series):
    """Longest run of True values in a boolean series."""
    runs, cur = 0, 0
    for v in bool_series:
        if v:
            cur += 1
            runs = max(runs, cur)
        else:
            cur = 0
    return runs


# 4. Core ETCCDI computation per dam (Dual Aggregation)
def etccdi_features(group, p95, p99):
    g = group.dropna(subset=['precip_mm']).copy()
    g['year'] = g['date'].dt.year
    
    # Define a master list of columns for empty/fallback groups
    all_cols = [
        'PRCPTOT_mean', 'PRCPTOT_max',
        'RX1day_mean',  'RX1day_max',
        'RX5day_mean',  'RX5day_max',
        'R10mm_mean',   'R10mm_max',
        'R20mm_mean',   'R20mm_max',
        'R25mm_mean',   'R25mm_max',
        'R30mm_mean',   'R30mm_total',
        'R100mm_mean',  'R100mm_total',
        'CDD_mean',     'CDD_max',
        'CWD_mean',     'CWD_max',
        'R95p_mean',    'R95p_max',
        'R99p_mean',    'R99p_max',
        'SDII'
    ]
    
    if g.empty:
        return pd.Series({k: np.nan for k in all_cols})

    # Group measurements by year locally
    yearly = {}
    for yr, yg in g.groupby('year'):
        wet = yg.loc[yg['precip_mm'] >= 1.0, 'precip_mm']
        
        yearly.setdefault('PRCPTOT', []).append(wet.sum())
        yearly.setdefault('RX1day',  []).append(yg['precip_mm'].max())
        yearly.setdefault('RX5day',  []).append(
            yg['precip_mm'].rolling(5, min_periods=5).sum().max()
        )
        yearly.setdefault('R10mm',   []).append((yg['precip_mm'] >= 10).sum())
        yearly.setdefault('R20mm',   []).append((yg['precip_mm'] >= 20).sum())
        yearly.setdefault('R25mm',   []).append((yg['precip_mm'] >= 25).sum())
        yearly.setdefault('R30mm',   []).append((yg['precip_mm'] > 30).sum())
        yearly.setdefault('R100mm',  []).append((yg['precip_mm'] > 100).sum())
        yearly.setdefault('CDD',     []).append(max_consecutive(yg['precip_mm'] < 1.0))
        yearly.setdefault('CWD',     []).append(max_consecutive(yg['precip_mm'] >= 1.0))
        yearly.setdefault('R95p',    []).append(
            yg.loc[yg['precip_mm'] > p95, 'precip_mm'].sum() if pd.notna(p95) else np.nan
        )
        yearly.setdefault('R99p',    []).append(
            yg.loc[yg['precip_mm'] > p99, 'precip_mm'].sum() if pd.notna(p99) else np.nan
        )

    features = {}
    
    # Precipitation Totals (Baseline vs Worst Year Volume)
    features['PRCPTOT_mean'] = np.mean(yearly['PRCPTOT'])
    features['PRCPTOT_max']  = np.max(yearly['PRCPTOT'])
    
    # Peak Storm Intensities (Mean Max vs Absolute Flash-Flood Event)
    features['RX1day_mean']  = np.mean(yearly['RX1day'])
    features['RX1day_max']   = np.max(yearly['RX1day'])   
    features['RX5day_mean']  = np.mean(yearly['RX5day'])
    features['RX5day_max']   = np.max(yearly['RX5day'])   
    
    # Frequency Counts (Mean Yearly Occurrences vs Max Annual Threshold Breakers)
    features['R10mm_mean']   = np.mean(yearly['R10mm'])
    features['R10mm_max']    = np.max(yearly['R10mm'])
    features['R20mm_mean']   = np.mean(yearly['R20mm'])
    features['R20mm_max']    = np.max(yearly['R20mm'])
    features['R25mm_mean']   = np.mean(yearly['R25mm'])
    features['R25mm_max']    = np.max(yearly['R25mm'])
    
    # Heavy Rain / Cloudburst Shock Proxies (Mean Frequency vs Total Cumulative Shocks Across Window)
    features['R30mm_mean']   = np.mean(yearly['R30mm'])
    features['R30mm_total']  = np.sum(yearly['R30mm'])   
    features['R100mm_mean']  = np.mean(yearly['R100mm'])
    features['R100mm_total'] = np.sum(yearly['R100mm'])  
    
    # Climate Spells (Mean Length vs Worst-Case Single Season Drought/Rain Wave)
    features['CDD_mean']     = np.mean(yearly['CDD'])
    features['CDD_max']      = np.max(yearly['CDD'])      
    features['CWD_mean']     = np.mean(yearly['CWD'])
    features['CWD_max']      = np.max(yearly['CWD'])      
    
    # Historic Anomalies (Mean Anomalous Total vs Highest Single Year Anomaly)
    features['R95p_mean']    = np.mean(yearly['R95p'])
    features['R95p_max']     = np.max(yearly['R95p'])
    features['R99p_mean']    = np.mean(yearly['R99p'])
    features['R99p_max']     = np.max(yearly['R99p'])

    # Core overall intensity across entire series
    wet_all = g.loc[g['precip_mm'] >= 1.0, 'precip_mm']
    features['SDII'] = wet_all.sum() / len(wet_all) if len(wet_all) > 0 else np.nan

    return pd.Series(features)


# 5. Process Loops
print("Extracting dual-aggregated features across all construction profiles...")
rows = []
for name, group in constr.groupby('project_name'):
    th = thresholds.loc[thresholds['project_name'] == name]
    p95 = th['p95_thresh'].values[0] if len(th) else np.nan
    p99 = th['p99_thresh'].values[0] if len(th) else np.nan
    
    feats = etccdi_features(group, p95, p99)
    feats['project_name'] = name
    rows.append(feats)

# Convert results array to DataFrame
etccdi_df = pd.DataFrame(rows)

# Dynamic column sorting: keeps project_name first, followed by all explicit indices
col_order = ['project_name'] + [c for c in etccdi_df.columns if c != 'project_name']
etccdi_df = etccdi_df[col_order]


# 6. Merge & Output Export
final = df_master.merge(etccdi_df, on='project_name', how='left')
final.to_csv('DA\dam_etccdi_features.csv', index=False)

print(f"\n[SUCCESS] Saved dam_etccdi_features.csv")
print(f"Dataset Size: {final.shape[0]} rows | {final.shape[1]} total columns.")
print(f"Added {len(col_order) - 1} hazard columns (Both climatological means and physical shock maximums).")

print("\n--- Missing Data Vector Matrix (%) ---")
print((etccdi_df[col_order[1:]].isnull().mean() * 100).round(1).to_string())

print("\n--- Head Preview of Result Set (Selected Shock Vectors) ---")
preview_cols = ['project_name', 'PRCPTOT_mean', 'RX1day_max', 'R100mm_total', 'CWD_max']
print(etccdi_df[preview_cols].head(5).to_string(index=False))