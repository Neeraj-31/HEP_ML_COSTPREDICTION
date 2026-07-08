"""

  slope_mean_deg     Mean slope (degrees)
  slope_max_deg      Max slope pixel
  slope_std_deg      Slope heterogeneity (std dev)
  steep_frac_25      Fraction of buffer area with slope > 25°
                      (25° is the GSI/BIS threshold commonly used as the
                      onset of high landslide susceptibility in Himalayan
                      terrain — used here as a proxy since true GSI zonation
                      polygons aren't in this script; see note at bottom)
  elev_mean_m        Mean elevation in buffer
  elev_range_m       Elevation range (max - min) — ruggedness proxy
  aspect_circ_std    Circular std dev of aspect (slope-facing direction
                      diversity — higher = more topographically complex)

"""

import ee
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

ee.Authenticate()
ee.Initialize(project='hepml-501008')   

CSV_PATH = r'DA\dam_dataset_catchment_verified (2).csv'
df = pd.read_csv(CSV_PATH,encoding='latin1')

MIN_RADIUS_KM = 8
MAX_RADIUS_KM = 150

def catchment_radius_km(area_sq_km):
    if pd.isna(area_sq_km) or area_sq_km <= 0:
        return MIN_RADIUS_KM
    r = np.sqrt(area_sq_km / np.pi)
    return float(np.clip(r, MIN_RADIUS_KM, MAX_RADIUS_KM))

df['catchment_radius_km'] = df['catchment_area_sq_km'].apply(catchment_radius_km)
df['catchment_radius_m']  = df['catchment_radius_km'] * 1000

print(f"Loaded {len(df)} dams.")
print(df[['project_name','catchment_area_sq_km','catchment_radius_km']]
      .head(10).to_string(index=False))

# ── Terrain rasters (static, computed once) ─────────────────────────────────────
srtm   = ee.Image('USGS/SRTMGL1_003')
slope  = ee.Terrain.slope(srtm)     # degrees
aspect = ee.Terrain.aspect(srtm)    # degrees, 0-360

aspect_rad = aspect.multiply(np.pi / 180)
aspect_sin = aspect_rad.sin().rename('aspect_sin')
aspect_cos = aspect_rad.cos().rename('aspect_cos')

terrain_stack = ee.Image.cat([srtm.rename('elevation'), slope.rename('slope'),
                               aspect_sin, aspect_cos])

# ── Per-dam extraction ───────────────────────────────────────────────────────────
rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    name   = row['project_name']
    geom   = ee.Geometry.Point([row['Longitude'], row['Latitude']]).buffer(
        row['catchment_radius_m']
    )

    try:
        stats = terrain_stack.reduceRegion(
            reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)
                                       .combine(ee.Reducer.min(), sharedInputs=True)
                                       .combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry = geom,
            scale    = 30,
            maxPixels= 1e10
        ).getInfo()

        # steep_frac_25: fraction of pixels with slope > 25°
        steep_mask = slope.gt(25)
        steep_stats = steep_mask.reduceRegion(
            reducer  = ee.Reducer.mean(),   # mean of a 0/1 mask = fraction
            geometry = geom,
            scale    = 30,
            maxPixels= 1e10
        ).getInfo()

        mean_sin = stats.get('aspect_sin_mean')
        mean_cos = stats.get('aspect_cos_mean')
        # circular std dev: 0 = all slopes face same direction, 1 = fully random
        if mean_sin is not None and mean_cos is not None:
            R = np.sqrt(mean_sin**2 + mean_cos**2)
            aspect_circ_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.nan
        else:
            aspect_circ_std = np.nan

        rows.append({
            'project_name'    : name,
            'catchment_radius_km': row['catchment_radius_km'],
            'slope_mean_deg'  : stats.get('slope_mean'),
            'slope_max_deg'   : stats.get('slope_max'),
            'slope_std_deg'   : stats.get('slope_stdDev'),
            'steep_frac_25'   : steep_stats.get('slope'),
            'elev_mean_m'     : stats.get('elevation_mean'),
            'elev_min_m'      : stats.get('elevation_min'),
            'elev_max_m'      : stats.get('elevation_max'),
            'elev_range_m'    : (stats.get('elevation_max') - stats.get('elevation_min'))
                                 if stats.get('elevation_max') is not None
                                 and stats.get('elevation_min') is not None else np.nan,
            'aspect_circ_std' : aspect_circ_std,
        })

    except Exception as ex:
        print(f"  ERROR {name}: {ex}")
        rows.append({'project_name': name})

    time.sleep(0.2)

out_df = pd.DataFrame(rows)
out_df.to_csv('DA/dam_slope_landslide_features.csv', index=False)

print(f"\nSaved dam_slope_landslide_features.csv ({out_df.shape[0]} rows, "
      f"{out_df.shape[1]} columns)")
print("\nMissing %:")
print((out_df.drop(columns=['project_name']).isnull().mean() * 100).round(1).to_string())
print("\nSample:")
print(out_df.head(5).to_string(index=False))

