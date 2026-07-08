"""
  glof_lake_count_50km     Count of glacial lakes within 50km of dam point
  glof_lake_area_km2_50km  Total area of those lakes (km²)
  glof_nearest_lake_km     Distance to single nearest glacial lake (km)
  glof_max_hazard_class    Highest ICIMOD hazard rank among nearby lakes
                            (only populated if ICIMOD file has a hazard field
                            — varies by which inventory file you grab; script
                            checks for common field names and tells you if
                            it can't find one)
  glof_risk_encoded        Ordinal encoding of your existing glof_risk column
                            (Low=1, Medium/High=2, High=3) — kept as a
                            cross-check against the computed metrics above
  data_source_used         'ICIMOD' 
"""

import ee
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

ee.Authenticate()
ee.Initialize(project='hepml-501008')   # ← replace with your GEE project

CSV_PATH = r'DA\dam_dataset_catchment_verified (2).csv'
ICIMOD_DIR = Path(r"C:\Users\User\.vscode\HEP_ML\DA\glacial_lake_hkh_2005\data")
GLOF_SEARCH_RADIUS_M = 50_000   # 50km — GLOF travel distances are basin-scale,
                                 # larger than the local catchment buffer

# ── TEMPORAL MASKING SETTINGS ────────────────────────────────────────────────────
# Replace 'year' with your actual column name for dam construction/commissioning
YEAR_COL = 'year'  
GLOF_START_YEAR = 1985
# ─────────────────────────────────────────────────────────────────────────────────

# Added encoding='latin1' to fix Windows/Python UnicodeDecodeError
df = pd.read_csv(CSV_PATH, encoding='latin1')

glof_risk_map = {
    'Low/Unknown'                      : 1,
    'Medium/High (Elevation > 1500m)'  : 2,
    'High (History/Prone)'             : 3,
}
# Safely apply map, defaulting to 1 if missing
if 'glof_risk' in df.columns:
    df['glof_risk_encoded'] = df['glof_risk'].map(glof_risk_map).fillna(1)
else:
    df['glof_risk_encoded'] = 1

# ── Try to find an ICIMOD file ───────────────────────────────────────────────────
def find_icimod_file():
    if not ICIMOD_DIR.exists():
        return None
    for ext in ('*.geojson', '*.json', '*.shp'):
        matches = list(ICIMOD_DIR.glob(ext))
        if matches:
            return matches[0]
    return None

icimod_file = find_icimod_file()
USE_ICIMOD = icimod_file is not None

if USE_ICIMOD:
    print(f" Found ICIMOD file: {icimod_file}")
    print("  Will compute REAL glacial lake distances/areas from this inventory.")
else:
    print(" No ICIMOD file found in ./icimod_glacial_lakes/")
    print("  Falling back to JRC Global Surface Water + elevation>3500m proxy.")
    print("  (See the docstring at the top of this script for how to get the")
    print("   ICIMOD file and upgrade accuracy.)")

print()

# ==============================================================================
# PATH A — ICIMOD real inventory (runs if file found)
# ==============================================================================
if USE_ICIMOD:
    import geopandas as gpd
    from shapely.geometry import Point

    lakes = gpd.read_file(icimod_file)
    print(f"  Loaded {len(lakes)} glacial lake features from ICIMOD file.")
    print(f"  Columns available: {list(lakes.columns)}")

    # Try to auto-detect an area field and a hazard-class field by common names
    AREA_FIELD_CANDIDATES = ['Area_km2', 'AREA_KM2', 'area_km2', 'Area', 'AREA',
                              'Area_sqkm', 'LAKE_AREA']
    HAZARD_FIELD_CANDIDATES = ['Hazard', 'HAZARD', 'Danger_Ran', 'PDGL', 'Risk',
                                'Hazard_Cla', 'hazard_class']

    area_field = next((f for f in AREA_FIELD_CANDIDATES if f in lakes.columns), None)
    hazard_field = next((f for f in HAZARD_FIELD_CANDIDATES if f in lakes.columns), None)

    if area_field is None:
        print("  NOTE: no recognized area column found — computing area from")
        print("        geometry directly (re-projecting to UTM for accuracy).")
    if hazard_field is None:
        print("  NOTE: no recognized hazard-class column found —")
        print("        glof_max_hazard_class will be NaN for all rows.")

    # Ensure CRS is geographic (WGS84) for distance math via geodesic approx,
    # then reproject to a local UTM-like equal-area CRS for accurate area/distance
    if lakes.crs is None:
        lakes = lakes.set_crs(epsg=4326)
    lakes_wgs84 = lakes.to_crs(epsg=4326)

    # India spans UTM zones 43N-45N roughly; use EPSG:32644 (UTM 44N) as a
    # reasonable single projected CRS for the whole HP/Uttarakhand extent —
    # introduces minor distortion at the zone edges but fine for 50km buffers
    lakes_proj = lakes_wgs84.to_crs(epsg=32644)

    if area_field is None:
        lakes_proj['computed_area_km2'] = lakes_proj.geometry.area / 1e6
        area_field = 'computed_area_km2'

    rows = []
    masked_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ICIMOD spatial query"):
        name = row['project_name']
        
        # Determine year safely
        dam_year = 2000 # default if missing
        if YEAR_COL in row:
            try:
                dam_year = float(row[YEAR_COL])
            except ValueError:
                pass
        
        # If built before 1985, assign zero risk baseline to prevent ML data leakage
        if dam_year < GLOF_START_YEAR:
            lake_count = 0
            total_area = 0.0
            nearest_km = 50.0  # max search radius bounds
            max_hazard = np.nan
            masked_count += 1
        else:
            # Perform actual spatial intersection for modern dams
            dam_pt_wgs84 = gpd.GeoSeries([Point(row['Longitude'], row['Latitude'])],
                                           crs='EPSG:4326')
            dam_pt_proj = dam_pt_wgs84.to_crs(epsg=32644).iloc[0]

            distances_m = lakes_proj.geometry.distance(dam_pt_proj)
            within_radius = distances_m <= GLOF_SEARCH_RADIUS_M
            nearby = lakes_proj[within_radius]

            lake_count = len(nearby)
            total_area = nearby[area_field].sum() if lake_count > 0 else 0.0
            nearest_km = distances_m.min() / 1000 if len(distances_m) > 0 else 50.0

            max_hazard = np.nan
            if hazard_field is not None and lake_count > 0:
                try:
                    max_hazard = pd.to_numeric(nearby[hazard_field], errors='coerce').max()
                except Exception:
                    max_hazard = nearby[hazard_field].astype(str).max()  # fallback: lexical

        rows.append({
            'project_name'            : name,
            'glof_lake_count_50km'    : lake_count,
            'glof_lake_area_km2_50km' : total_area,
            'glof_nearest_lake_km'    : nearest_km,
            'glof_max_hazard_class'   : max_hazard,
            'glof_risk_encoded'       : row['glof_risk_encoded'],
            'data_source_used'        : 'ICIMOD'
        })
    print(f"\nApplied 1985 temporal mask to {masked_count} older projects.")
    out_df = pd.DataFrame(rows)

# ==============================================================================
# PATH B — JRC Global Surface Water proxy (fallback, runs if no ICIMOD file)
# ==============================================================================
else:
    srtm = ee.Image('USGS/SRTMGL1_003')
    gsw  = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
    glacial_water = gsw.gt(70).And(srtm.gt(3500))   # permanent water, high elev

    rows = []
    masked_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="JRC proxy query"):
        name = row['project_name']
        
        # Determine year safely
        dam_year = 2000
        if YEAR_COL in row:
            try:
                dam_year = float(row[YEAR_COL])
            except ValueError:
                pass
                
        if dam_year < GLOF_START_YEAR:
            rows.append({
                'project_name'            : name,
                'glof_lake_count_50km'    : np.nan,
                'glof_lake_area_km2_50km' : 0.0,
                'glof_nearest_lake_km'    : np.nan,
                'glof_max_hazard_class'   : np.nan,
                'glof_risk_encoded'       : row['glof_risk_encoded'],
                'data_source_used'        : 'JRC_PROXY_PRE_1985'
            })
            masked_count += 1
            continue

        geom = ee.Geometry.Point([row['Longitude'], row['Latitude']]).buffer(
            GLOF_SEARCH_RADIUS_M
        )

        try:
            area_stats = glacial_water.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=geom, scale=30, maxPixels=1e10
            ).getInfo()

            rows.append({
                'project_name'            : name,
                'glof_lake_count_50km'    : np.nan,   # proxy can't count discrete lakes
                'glof_lake_area_km2_50km' : (area_stats.get('occurrence', 0) or 0) / 1e6,
                'glof_nearest_lake_km'    : np.nan,   # proxy can't do nearest-feature distance
                'glof_max_hazard_class'   : np.nan,
                'glof_risk_encoded'       : row['glof_risk_encoded'],
                'data_source_used'        : 'JRC_PROXY'
            })
        except Exception as ex:
            print(f"  ERROR {name}: {ex}")
        time.sleep(0.3)

    print(f"\nApplied 1985 temporal mask to {masked_count} older projects.")
    out_df = pd.DataFrame(rows)

# ── Composite score (works with either path; NaN-tolerant) ─────────────────────
def minmax(s):
    s = s.astype(float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(0.0, index=s.index)

out_df['glof_composite_score'] = (
  0.40 * minmax(out_df['glof_lake_area_km2_50km'].fillna(0))
  + 0.30 * minmax(out_df['glof_risk_encoded'])
  + 0.30 * (1 - minmax(out_df['glof_nearest_lake_km'].fillna(50)))
)

out_df.to_csv('dam_glof_features.csv', index=False)

print(f"\nSaved dam_glof_features.csv ({out_df.shape[0]} rows)")
print(f"Method used: {out_df['data_source_used'].iloc[0]}")
print("\nSample:")
print(out_df.head(8).to_string(index=False))

