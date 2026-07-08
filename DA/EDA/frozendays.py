"""
================================================================================
PILLAR 4 — Frozen Working Window (ERA5-Land via GEE)
================================================================================


BANDS EXTRACTED (daily, per dam, over construction window):
  temperature_2m_min    → frozen-day count, frost season length
  temperature_2m_max    → heat-stress days, freeze-thaw cycling proxy
  temperature_2m        → mean temp, growing degree days, workable season
  snow_depth            → direct work-halt signal (separate from Tmin)
  soil_temperature_level_1  → ground frost depth (foundation/slope work)
  total_precipitation   → ERA5 precip as cross-check on IMD data


FEATURES ENGINEERED (per dam, aggregated over construction window):
  frozen_days_mean        Mean annual days with Tmin < -5°C
  frozen_days_max         Worst single year frozen day count
  frozen_days_total       Total frozen days across whole window
  frost_free_season_days  Mean annual days with Tmin > 0°C (workable window)
  heat_stress_days_mean   Mean annual days with Tmax > 35°C
  freeze_thaw_days_mean   Mean annual days where Tmin<0°C AND Tmax>0°C
                          (daily freeze-thaw cycle — fractures rock/concrete)
  diurnal_temp_range_mean Mean of (Tmax - Tmin) — thermal stress proxy
  snow_depth_mean_cm      Mean daily snow depth over construction window (cm)
  snow_days_mean          Mean annual days with snow_depth > 0.01m
  high_snow_days_mean     Mean annual days with snow_depth > 0.5m (work halt)
  soil_frozen_days_mean   Mean annual days with soil_temp_L1 < 0°C
  gdd_mean                Mean annual Growing Degree Days (base 5°C)
                          — total thermal energy available for construction work
====================================================
"""

import ee
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

ee.Authenticate()
ee.Initialize(project='hepml-501008')   #  replace

CSV_PATH = r'DA\dam_dataset_catchment_verified (2).csv'
ERA5_START = 1950
ERA5_END   = 2024

MIN_RADIUS_KM = 8
MAX_RADIUS_KM = 150

df = pd.read_csv(CSV_PATH,encoding='latin1')
df['starting_year']      = df['starting_year'].astype(int)
df['commissioning_year'] = df['commissioning_year'].astype(int)

def clamp(y, lo, hi): return max(lo, min(hi, y))
def catchment_radius_m(area_sq_km):
    if pd.isna(area_sq_km) or area_sq_km <= 0:
        return MIN_RADIUS_KM * 1000
    r = np.sqrt(area_sq_km / np.pi)
    return float(np.clip(r, MIN_RADIUS_KM, MAX_RADIUS_KM)) * 1000

df['buffer_m'] = df['catchment_area_sq_km'].apply(catchment_radius_m)

BANDS = [
    'temperature_2m_min',
    'temperature_2m_max',
    'temperature_2m',
    'snow_depth',
    'soil_temperature_level_1',
    'total_precipitation',
]

print(f"Loaded {len(df)} dams.")

all_rows = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Dams"):
    name   = row['project_name']
    s_yr   = clamp(row['starting_year'],      ERA5_START, ERA5_END)
    e_yr   = clamp(row['commissioning_year'], ERA5_START, ERA5_END)
    geom   = ee.Geometry.Point([row['Longitude'], row['Latitude']]).buffer(
                 row['buffer_m'])

    if s_yr > e_yr:
        print(f"  Skipping {name}: window outside ERA5 record")
        continue

    for year in range(s_yr, e_yr + 1):
        start = f'{year}-01-01'
        end   = f'{year}-12-31'

        era5 = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                  .filterDate(start, end)
                  .select(BANDS))

        def to_feat(img):
            vals = img.reduceRegion(
                reducer  = ee.Reducer.mean(),
                geometry = geom,
                scale    = 11132,
                maxPixels= 1e9
            )
            return ee.Feature(None, {
                'project_name'          : name,
                'date'                  : img.date().format('YYYY-MM-dd'),
                'tmin_k'                : vals.get('temperature_2m_min'),
                'tmax_k'                : vals.get('temperature_2m_max'),
                'tmean_k'               : vals.get('temperature_2m'),
                'snow_depth_m'          : vals.get('snow_depth'),
                'soil_temp_l1_k'        : vals.get('soil_temperature_level_1'),
                'precip_m'              : vals.get('total_precipitation'),
            })

        try:
            data = era5.map(to_feat).getInfo()
            for f in data['features']:
                p = f['properties']
                all_rows.append({
                    'project_name'  : p['project_name'],
                    'date'          : p['date'],
                    'tmin_c'        : (p['tmin_k']       - 273.15) if p['tmin_k']       else np.nan,
                    'tmax_c'        : (p['tmax_k']       - 273.15) if p['tmax_k']       else np.nan,
                    'tmean_c'       : (p['tmean_k']      - 273.15) if p['tmean_k']      else np.nan,
                    'snow_depth_m'  : p['snow_depth_m'],
                    'soil_temp_c'   : (p['soil_temp_l1_k'] - 273.15) if p['soil_temp_l1_k'] else np.nan,
                    'precip_mm'     : (p['precip_m'] * 1000)        if p['precip_m']    else np.nan,
                })
        except Exception as ex:
            print(f"  ERROR {name} / {year}: {ex}")

        time.sleep(0.2)

raw_df = pd.DataFrame(all_rows)
raw_df.to_csv('raw_era5_construction_window.csv', index=False)
print(f"\n→ raw_era5_construction_window.csv ({len(raw_df):,} rows)")

# ── Feature Engineering ────────────────────────────────────────────────────────
print("\nEngineering Pillar 4 features …")

raw_df['date'] = pd.to_datetime(raw_df['date'])
raw_df['year'] = raw_df['date'].dt.year

feat_rows = []

for name, g in raw_df.groupby('project_name'):
    g = g.dropna(subset=['tmin_c'])
    if g.empty:
        feat_rows.append({'project_name': name})
        continue

    yearly = {}
    for yr, yg in g.groupby('year'):
        yearly.setdefault('frozen_days',       []).append((yg['tmin_c'] < -5).sum())
        yearly.setdefault('frost_free_days',   []).append((yg['tmin_c'] >  0).sum())
        yearly.setdefault('heat_stress_days',  []).append((yg['tmax_c'] > 35).sum()
                                                           if yg['tmax_c'].notna().any() else np.nan)
        yearly.setdefault('freeze_thaw_days',  []).append(
            ((yg['tmin_c'] < 0) & (yg['tmax_c'] > 0)).sum()
            if yg['tmax_c'].notna().any() else np.nan)
        yearly.setdefault('dtr',               []).append(
            (yg['tmax_c'] - yg['tmin_c']).mean()
            if yg['tmax_c'].notna().any() else np.nan)
        yearly.setdefault('snow_days',         []).append(
            (yg['snow_depth_m'] > 0.01).sum()
            if yg['snow_depth_m'].notna().any() else np.nan)
        yearly.setdefault('high_snow_days',    []).append(
            (yg['snow_depth_m'] > 0.5).sum()
            if yg['snow_depth_m'].notna().any() else np.nan)
        yearly.setdefault('soil_frozen_days',  []).append(
            (yg['soil_temp_c'] < 0).sum()
            if yg['soil_temp_c'].notna().any() else np.nan)
        gdd = (yg['tmean_c'] - 5).clip(lower=0).sum() if yg['tmean_c'].notna().any() else np.nan
        yearly.setdefault('gdd',               []).append(gdd)

    feat_rows.append({
        'project_name'          : name,
        'frozen_days_mean'      : np.nanmean(yearly['frozen_days']),
        'frozen_days_max'       : np.nanmax(yearly['frozen_days']),
        'frozen_days_total'     : np.nansum(yearly['frozen_days']),
        'frost_free_season_days': np.nanmean(yearly['frost_free_days']),
        'heat_stress_days_mean' : np.nanmean(yearly['heat_stress_days']),
        'freeze_thaw_days_mean' : np.nanmean(yearly['freeze_thaw_days']),
        'diurnal_temp_range_mean': np.nanmean(yearly['dtr']),
        'snow_depth_mean_cm'    : g['snow_depth_m'].mean() * 100
                                   if g['snow_depth_m'].notna().any() else np.nan,
        'snow_days_mean'        : np.nanmean(yearly['snow_days']),
        'high_snow_days_mean'   : np.nanmean(yearly['high_snow_days']),
        'soil_frozen_days_mean' : np.nanmean(yearly['soil_frozen_days']),
        'gdd_mean'              : np.nanmean(yearly['gdd']),
    })

feat_df = pd.DataFrame(feat_rows)
feat_df.to_csv('dam_frozen_window_features.csv', index=False)

print(f"→ dam_frozen_window_features.csv ({feat_df.shape[0]} rows, "
      f"{feat_df.shape[1]-1} features)")
print("\nMissing %:")
print((feat_df.drop(columns='project_name').isnull().mean()*100).round(1).to_string())
print("\nSample:")
print(feat_df.head(5).to_string(index=False))

