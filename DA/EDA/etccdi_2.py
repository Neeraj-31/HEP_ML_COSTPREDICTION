"""
================================================================================
ETCCDI Feature Engineering — with RX7day + Yearly Output + ML Aggregations
================================================================================
OUTPUTS TWO FILES:
  1. dam_etccdi_yearly.csv    — long format, one row per dam per year
                                 (all raw yearly index values, keep for reference)
  2. dam_etccdi_features.csv  — one row per dam, ML-ready aggregations
                                 derived from the yearly table

INDICES:
  Standard ETCCDI: PRCPTOT, RX1day, RX5day, RX7day*, SDII, R10mm, R20mm,
                   R25mm, CDD, CWD, R95p, R99p
  Non-standard:    R30mm, R100mm (cloudburst proxies)

  *RX7day: max 7-day running accumulation per year — added to capture
   week-long sustained deluge events that are distinct from RX5day and
   are more relevant to riverbank saturation and slope failure timescales
   in Himalayan terrain than single-day peak events.

AGGREGATIONS (from yearly → one row per dam):
  _mean    Mean across construction years      (baseline chronic exposure)
  _max     Worst single year                   (physical shock peak)
  _total   Sum across all years                (cumulative burden, R30mm/R100mm only)
  _trend   Sen's slope across years            (is hazard worsening over construction?)
  _p90     90th percentile year                (near-worst without outlier sensitivity)
  _cv      Coefficient of variation            (year-to-year volatility — unpredictability)
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load data ───────────────────────────────────────────────────────────────
df_master = pd.read_csv(r'DA\dam_dataset_catchment_verified (2).csv', encoding='cp1252')
constr    = pd.read_csv(r'DA\raw_imd_construction_window.csv', parse_dates=['date'])
base      = pd.read_csv(r'DA\raw_imd_base_period.csv',        parse_dates=['date'])

print(f"Construction window: {len(constr):,} rows | Base period: {len(base):,} rows")

# ── 2. Base-period percentile thresholds ───────────────────────────────────────
def base_thresholds(group):
    wet = group.loc[group['precip_mm'] >= 1.0, 'precip_mm']
    if wet.empty:
        return pd.Series({'p95_thresh': np.nan, 'p99_thresh': np.nan})
    return pd.Series({'p95_thresh': wet.quantile(0.95),
                      'p99_thresh': wet.quantile(0.99)})

print("Computing base-period percentile thresholds...")
thresholds = base.groupby('project_name').apply(base_thresholds).reset_index()

# ── 3. Helpers ─────────────────────────────────────────────────────────────────
def max_consecutive(bool_series):
    runs, cur = 0, 0
    for v in bool_series:
        if v:
            cur += 1; runs = max(runs, cur)
        else:
            cur = 0
    return runs

def sens_slope(values):
    """
    Sen's slope — median of all pairwise slopes.
    More robust than linear regression for short, noisy climate series.
    Returns slope per year (positive = index increasing over construction window).
    """
    vals = [v for v in values if not np.isnan(v)]
    n = len(vals)
    if n < 3:
        return np.nan
    slopes = []
    for i in range(n):
        for j in range(i+1, n):
            slopes.append((vals[j] - vals[i]) / (j - i))
    return np.median(slopes)

# ── 4. Yearly ETCCDI computation per dam ──────────────────────────────────────
def compute_yearly(group, p95, p99):
    """
    Returns a list of dicts — one per year — with all index values for that year.
    """
    g = group.dropna(subset=['precip_mm']).copy()
    g['year'] = g['date'].dt.year

    yearly_rows = []

    for yr, yg in g.groupby('year'):
        wet = yg.loc[yg['precip_mm'] >= 1.0, 'precip_mm']

        row = {'year': yr}
        row['PRCPTOT'] = wet.sum()
        row['RX1day']  = yg['precip_mm'].max()
        row['RX5day']  = yg['precip_mm'].rolling(5, min_periods=5).sum().max()
        row['RX7day']  = yg['precip_mm'].rolling(7, min_periods=7).sum().max()  # ← NEW
        row['R10mm']   = int((yg['precip_mm'] >= 10).sum())
        row['R20mm']   = int((yg['precip_mm'] >= 20).sum())
        row['R25mm']   = int((yg['precip_mm'] >= 25).sum())
        row['R30mm']   = int((yg['precip_mm'] >  30).sum())
        row['R100mm']  = int((yg['precip_mm'] > 100).sum())
        row['CDD']     = max_consecutive(yg['precip_mm'] < 1.0)
        row['CWD']     = max_consecutive(yg['precip_mm'] >= 1.0)
        row['R95p']    = yg.loc[yg['precip_mm'] > p95, 'precip_mm'].sum() if pd.notna(p95) else np.nan
        row['R99p']    = yg.loc[yg['precip_mm'] > p99, 'precip_mm'].sum() if pd.notna(p99) else np.nan

        # SDII: computed per year here (differs from your original which was global)
        row['SDII'] = wet.sum() / len(wet) if len(wet) > 0 else np.nan

        yearly_rows.append(row)

    return yearly_rows

# ── 5. Main loop — build yearly long-format table ─────────────────────────────
print("Computing yearly ETCCDI values for all dams...")
all_yearly_rows = []

for name, group in constr.groupby('project_name'):
    th  = thresholds.loc[thresholds['project_name'] == name]
    p95 = th['p95_thresh'].values[0] if len(th) else np.nan
    p99 = th['p99_thresh'].values[0] if len(th) else np.nan

    yearly = compute_yearly(group, p95, p99)
    for r in yearly:
        r['project_name'] = name
    all_yearly_rows.extend(yearly)

yearly_df = pd.DataFrame(all_yearly_rows)

# Column order for yearly output
INDEX_COLS = ['PRCPTOT','RX1day','RX5day','RX7day','SDII',
              'R10mm','R20mm','R25mm','R30mm','R100mm',
              'CDD','CWD','R95p','R99p']
yearly_df = yearly_df[['project_name','year'] + INDEX_COLS]
yearly_df = yearly_df.sort_values(['project_name','year']).reset_index(drop=True)

yearly_df.to_csv('dam_etccdi_yearly.csv', index=False)
print(f" dam_etccdi_yearly.csv: {len(yearly_df):,} rows "
      f"({yearly_df['project_name'].nunique()} dams)")

# ── 6. Aggregate yearly → one row per dam for ML ──────────────────────────────
print("Aggregating yearly values to ML feature rows...")

# Indices that get mean + max + trend + p90 + cv
STANDARD_INDICES = ['PRCPTOT','RX1day','RX5day','RX7day','SDII',
                    'R10mm','R20mm','R25mm','CDD','CWD','R95p','R99p']
# Cloudburst proxies: mean + total + trend (total is more meaningful than max here)
SHOCK_INDICES    = ['R30mm','R100mm']

feat_rows = []

for name, g in yearly_df.groupby('project_name'):
    g = g.sort_values('year')
    row = {'project_name': name, 'n_construction_years': len(g)}

    for idx in STANDARD_INDICES:
        vals = g[idx].dropna().tolist()
        if not vals:
            row[f'{idx}_mean'] = np.nan
            row[f'{idx}_max']  = np.nan
            row[f'{idx}_p90']  = np.nan
            row[f'{idx}_cv']   = np.nan
            row[f'{idx}_trend'] = np.nan
            continue
        row[f'{idx}_mean']  = np.mean(vals)
        row[f'{idx}_max']   = np.max(vals)
        row[f'{idx}_p90']   = np.percentile(vals, 90)
        row[f'{idx}_cv']    = (np.std(vals) / np.mean(vals)) if np.mean(vals) != 0 else np.nan
        row[f'{idx}_trend'] = sens_slope(vals)

    for idx in SHOCK_INDICES:
        vals = g[idx].dropna().tolist()
        if not vals:
            row[f'{idx}_mean']  = np.nan
            row[f'{idx}_total'] = np.nan
            row[f'{idx}_trend'] = np.nan
            continue
        row[f'{idx}_mean']  = np.mean(vals)
        row[f'{idx}_total'] = np.sum(vals)
        row[f'{idx}_trend'] = sens_slope(vals)

    feat_rows.append(row)

etccdi_feats = pd.DataFrame(feat_rows)

# ── 7. Merge with master and export ───────────────────────────────────────────
final = df_master.merge(etccdi_feats, on='project_name', how='left')
final.to_csv('dam_etccdi_features.csv', index=False)

feat_cols = [c for c in etccdi_feats.columns if c != 'project_name']
print(f"\n[SUCCESS] dam_etccdi_features.csv: {final.shape[0]} rows | {final.shape[1]} columns")
print(f"          {len(feat_cols)} ETCCDI feature columns added")

print("\n--- Missing Data (%) ---")
print((etccdi_feats[feat_cols].isnull().mean() * 100).round(1).to_string())

print("\n--- Feature Column List ---")
for c in feat_cols:
    print(f"  {c}")

print("\n--- Sample: RX7day values across dams ---")
rx7_cols = ['project_name','RX7day_mean','RX7day_max','RX7day_p90','RX7day_trend']
print(etccdi_feats[[c for c in rx7_cols if c in etccdi_feats.columns]].head(10).to_string(index=False))