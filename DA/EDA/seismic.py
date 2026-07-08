"""


FEATURES ENGINEERED:
  pga_g                 PGA at dam point (g) — primary seismic hazard value
  pga_mean_buffer       Mean PGA within catchment buffer (regional context)
  pga_max_buffer        Max PGA within buffer (worst-case local hazard)
  seismic_zone_encoded  Ordinal encoding of your existing zone column
                        (IV=1, IV/V=2, V=3) — kept as cross-check
  pga_zone_consistency  Flag: 1 if PGA and zone agree directionally, 0 if not
                        (useful for catching data quality issues)
==================================================
"""

import ee
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

ee.Authenticate()
ee.Initialize(project='hepml-501008')   

CSV_PATH = r'DA\dam_dataset_catchment_verified (2).csv'
MIN_RADIUS_KM = 8
MAX_RADIUS_KM = 150

df = pd.read_csv(CSV_PATH,encoding='latin1')

def catchment_radius_m(area_sq_km):
    if pd.isna(area_sq_km) or area_sq_km <= 0:
        return MIN_RADIUS_KM * 1000
    r = np.sqrt(area_sq_km / np.pi)
    return float(np.clip(r, MIN_RADIUS_KM, MAX_RADIUS_KM)) * 1000

df['buffer_m'] = df['catchment_area_sq_km'].apply(catchment_radius_m)

zone_map = {'IV': 1, 'IV/V': 2, 'V': 3}
df['seismic_zone_encoded'] = df['seismic_zone'].map(zone_map).fillna(1)

# ── Load seismic hazard image ──────────────────────────────────────────────────
# If USGS/GLOBAL_SEISMIC_HAZARD_V2 is not available in your GEE region,
# the script falls back to a community-uploaded version — see note below.
try:
    pga_img = ee.Image('USGS/GLOBAL_SEISMIC_HAZARD_V2').select('PGA')
    pga_img.getInfo()   # quick check to confirm the asset loads
    print(" USGS/GLOBAL_SEISMIC_HAZARD_V2 loaded successfully.")
except Exception:
    # Fallback: OpenQuake Global Hazard mosaic (community GEE asset)
    # This is the GSHAP-derived dataset often used when the USGS image
    # isn't accessible. Units are identical (g, 10% in 50yr).
    print("  Note: USGS V2 not found, trying OpenQuake community asset …")
    try:
        pga_img = ee.Image('users/openquake/GHM_PGA_475yr').select('b1')
        print("OpenQuake GHM PGA loaded.")
    except Exception as ex2:
        print(f"  ERROR: neither seismic asset loaded: {ex2}")
        print("  Manual fallback: download GSHAP PGA raster from")
        print("  https://www.gfz-potsdam.de/en/section/seismic-hazard-and-risk-dynamics/")
        print("  upload as a GEE asset and replace the image ID above.")
        raise

print(f"\nExtracting seismic PGA for {len(df)} dams …")

rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    name  = row['project_name']
    point = ee.Geometry.Point([row['Longitude'], row['Latitude']])
    buff  = point.buffer(row['buffer_m'])

    try:
        # Point value (exact dam site)
        point_val = pga_img.reduceRegion(
            reducer  = ee.Reducer.first(),
            geometry = point,
            scale    = 1000,
            maxPixels= 1e6
        ).getInfo()

        # Buffer stats (catchment-wide hazard context)
        buff_stats = pga_img.reduceRegion(
            reducer  = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
            geometry = buff,
            scale    = 1000,
            maxPixels= 1e9
        ).getInfo()

        pga_point = list(point_val.values())[0] if point_val else np.nan
        pga_mean  = list(buff_stats.values())[0] if buff_stats else np.nan
        pga_max   = list(buff_stats.values())[1] if len(buff_stats) > 1 else np.nan

        # Consistency check: zone V should have higher PGA than zone IV
        # Flag = 1 if zone and PGA agree, 0 if they contradict
        zone_enc = row['seismic_zone_encoded']
        if pd.notna(pga_point) and pd.notna(zone_enc):
            # Rough expected PGA ranges for BIS zones in g:
            # Zone IV: 0.10–0.24g, Zone V: >0.24g
            pga_implies_v = pga_point > 0.24
            zone_is_v     = zone_enc >= 2   # IV/V or V
            consistency   = int(pga_implies_v == zone_is_v)
        else:
            consistency = np.nan

        rows.append({
            'project_name'          : name,
            'pga_g'                 : pga_point,
            'pga_mean_buffer'       : pga_mean,
            'pga_max_buffer'        : pga_max,
            'seismic_zone_encoded'  : zone_enc,
            'pga_zone_consistency'  : consistency,
        })

    except Exception as ex:
        print(f"  ERROR {name}: {ex}")
        rows.append({'project_name': name, 'seismic_zone_encoded': row['seismic_zone_encoded']})

    time.sleep(0.15)

feat_df = pd.DataFrame(rows)
feat_df.to_csv('dam_seismic_features.csv', index=False)

print(f"\n→ dam_seismic_features.csv ({feat_df.shape[0]} rows)")
print("\nMissing %:")
print((feat_df.drop(columns='project_name').isnull().mean()*100).round(1).to_string())
print("\nSample (sorted by PGA descending):")
print(feat_df.sort_values('pga_g', ascending=False).head(10).to_string(index=False))

# Quick sanity check — high-seismic dams should cluster in known zones
print("\nZone V dams and their extracted PGA:")
zone_v = feat_df.merge(
    df[['project_name','seismic_zone']], on='project_name'
).query("seismic_zone == 'V'")[['project_name','seismic_zone','pga_g']]
print(zone_v.to_string(index=False))

