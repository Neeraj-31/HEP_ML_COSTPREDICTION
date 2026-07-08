import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG — edit these two paths ──────────────────────────────────────────────
NC_DIR   = Path(r'DA\Rainfall\nc_yearwise')               # folder containing rainfall_YYYY.nc
CSV_PATH = r'C:\Users\User\.vscode\HEP_ML\DA\dam_dataset_catchment_verified (2).csv'

IMD_AVAIL    = (1951, 2024)
BASE_PERIOD  = (1981, 2010)        # ETCCDI standard base period
MIN_RADIUS_KM = 8                  # floor, so even tiny catchments get ≥1 grid cell
MAX_RADIUS_KM = 150                # ceiling, so mega-basins (Bhakra, Nathpa Jhakri)
                                    # don't swallow neighbouring dams' signal entirely

# ── Load dam metadata ───────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH,encoding='latin1')
df['starting_year']      = df['starting_year'].astype(int)
df['commissioning_year'] = df['commissioning_year'].astype(int)

def clamp(y, lo, hi): return max(lo, min(hi, y))

def catchment_radius_km(area_sq_km):
    if pd.isna(area_sq_km) or area_sq_km <= 0:
        return MIN_RADIUS_KM
    r = np.sqrt(area_sq_km / np.pi)
    return float(np.clip(r, MIN_RADIUS_KM, MAX_RADIUS_KM))

df['catchment_radius_km'] = df['catchment_area_sq_km'].apply(catchment_radius_km)

print(f"Loaded {len(df)} dams.")
print(df[['project_name','catchment_area_sq_km','catchment_radius_km']]
      .head(10).to_string(index=False))

dam_years_constr = {}
dam_years_base    = {}

for _, row in df.iterrows():
    name = row['project_name']
    s_yr = clamp(row['starting_year'], *IMD_AVAIL)
    e_yr = clamp(row['commissioning_year'], *IMD_AVAIL)
    dam_years_constr[name] = list(range(s_yr, e_yr + 1)) if s_yr <= e_yr else []
    dam_years_base[name]   = list(range(BASE_PERIOD[0], BASE_PERIOD[1] + 1))

all_years_needed = sorted(set(
    y for ys in list(dam_years_constr.values()) + list(dam_years_base.values())
    for y in ys
))
print(f"\nIMD years needed across all dams: {all_years_needed[0]}–{all_years_needed[-1]} "
      f"({len(all_years_needed)} files)")

# ── Helper: extract daily catchment-mean series for one dam from one year's file ─
def extract_year_for_dam(ds, lat, lon, radius_km, varname):
  
    lat_name = 'lat' if 'lat' in ds.coords else 'LATITUDE'
    lon_name = 'lon' if 'lon' in ds.coords else 'LONGITUDE'

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # Rough km-per-degree conversion (good enough at these latitudes ~29-33°N)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat))

    lat_mask = np.abs(lats - lat) * km_per_deg_lat <= radius_km
    lon_mask = np.abs(lons - lon) * km_per_deg_lon <= radius_km

    if not lat_mask.any() or not lon_mask.any():
        # fall back to nearest single cell if buffer misses the grid entirely
        sel = ds[varname].sel({lat_name: lat, lon_name: lon}, method='nearest')
        return sel.to_series()

    sub = ds[varname].isel({lat_name: np.where(lat_mask)[0],
                             lon_name: np.where(lon_mask)[0]})
    # Catchment mean across the spatial subset, per day
    daily_mean = sub.mean(dim=[lat_name, lon_name], skipna=True)
    return daily_mean.to_series()



print("\nExtracting catchment-mean daily rainfall from each .nc file …")

constr_records = []   # rows: project_name, date, precip_mm
base_records   = []

for year in tqdm(all_years_needed):
    fpath = NC_DIR / f'rainfall_{year}.nc'
    if not fpath.exists():
        print(f"  WARNING: {fpath} not found — skipping year {year}")
        continue
    if fpath.stat().st_size == 0:
        print(f"  WARNING: {fpath} is 0 KB (corrupt/empty download) — skipping")
        continue

    try:
        ds = xr.open_dataset(fpath)
    except Exception as ex:
        print(f"  ERROR opening {fpath}: {ex}")
        continue

    # Identify the rainfall variable name (commonly 'rf' or 'RAINFALL' in IMD files)
    candidate_vars = [v for v in ds.data_vars
                       if ds[v].ndim >= 2 and 'time' in ds[v].dims]
    if not candidate_vars:
        print(f"  WARNING: no time-varying variable found in {fpath}, vars={list(ds.data_vars)}")
        ds.close()
        continue
    varname = candidate_vars[0]   # adjust manually if your file has multiple

    for _, row in df.iterrows():
        name   = row['project_name']
        lat    = row['Latitude']
        lon    = row['Longitude']
        radius = row['catchment_radius_km']

        needs_constr = year in dam_years_constr.get(name, [])
        needs_base   = year in dam_years_base.get(name, [])
        if not (needs_constr or needs_base):
            continue

        try:
            series = extract_year_for_dam(ds, lat, lon, radius, varname)
        except Exception as ex:
            print(f"  ERROR extracting {name} / {year}: {ex}")
            continue

        for date, val in series.items():
            rec = {'project_name': name, 'date': pd.Timestamp(date),
                   'precip_mm': float(val) if pd.notna(val) else np.nan}
            if needs_constr:
                constr_records.append(rec)
            if needs_base:
                base_records.append(rec)

    ds.close()

constr_df = pd.DataFrame(constr_records)
base_df   = pd.DataFrame(base_records)

constr_df.to_csv('raw_imd_construction_window.csv', index=False)
base_df.to_csv('raw_imd_base_period.csv', index=False)

print(f"\n   raw_imd_construction_window.csv ({len(constr_df):,} rows)")
print(f"   raw_imd_base_period.csv ({len(base_df):,} rows)")

