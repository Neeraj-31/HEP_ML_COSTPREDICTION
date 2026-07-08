"""
================================================================================
PILLAR 6 — Anthropogenic Factors
Census (2011) + Tourism Proximity Features
================================================================================
Produces 9 features per dam covering two anthropogenic risk dimensions:

PART A — CENSUS (district-level, 2011 PCA data you already have)
  These capture resettlement complexity and labour-market context —
  both documented cost-overrun drivers in Himalayan hydro projects:

  pop_density_district      District population / area proxy
                             (higher = more resettlement disputes,
                              NGO scrutiny, compensation renegotiation)
  st_pop_fraction           Scheduled Tribe fraction of district population
                             (ST land rights under Forest Rights Act 2006
                              trigger separate legal processes — a known
                              standalone delay cause in HP/UK projects)
  worker_ratio              Total workers / total population
                             (proxy for local labour availability;
                              low = harder to recruit unskilled workforce)
  household_size            Average household size (total pop / households)
                             (larger = more families displaced per village)
  non_worker_dependency     Non-working population fraction
                             (high = economically vulnerable communities,
                              more likely to contest displacement terms)

PART B — TOURISM PROXIMITY (hardcoded coordinates, no external data needed)
  These capture seasonal access constraints — heavy vehicle restrictions
  during pilgrim season on Char Dham routes are a direct work-halt cause:

  dist_to_nearest_pilgrim_km   Distance to nearest Char Dham/major pilgrimage
                                site (Badrinath, Kedarnath, Gangotri,
                                Yamunotri, Hemkund Sahib)
  dist_to_nearest_resort_km    Distance to nearest major tourist resort hub
                                (Manali, Shimla, Mussoorie, Nainital, Kullu)
  on_char_dham_corridor        1 if dam within 25km of a Char Dham highway,
                                0 otherwise (direct proxy for seasonal
                                heavy-vehicle restriction periods)
  tourism_pressure_score       Composite 0–1 score: inverse distance to
                                pilgrim sites + resort hubs, normalised


================================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 0. Load master dataset ────────────────────────────────────────────────────
dam_df = pd.read_csv(r'DA\dam_dataset_catchment_verified (2).csv', encoding='latin1')
print(f"Loaded {len(dam_df)} dams.")

# ── 1. District assignment (verified from lat/lon geography) ──────────────────
DAM_DISTRICT_MAP = {
    # Himachal Pradesh
    'alain duhangan'          : 'Kullu',
    'baira siul'              : 'Chamba',
    'nathpa jhakri'           : 'Kinnaur',
    'bhakra'                  : 'Bilaspur',
    'karcham wangtoo'         : 'Kinnaur',
    'koldam'                  : 'Bilaspur',
    'parbati II'              : 'Kullu',
    'parbati III'             : 'Kullu',
    'chamera I'               : 'Chamba',
    'chamera II'              : 'Chamba',
    'chamera III'             : 'Chamba',
    'bajoli holi'             : 'Chamba',
    'rampur'                  : 'Shimla',
    'malana I'                : 'Kullu',
    'malana II'               : 'Kullu',
    'sainj'                   : 'Kullu',
    'uhl III'                 : 'Mandi',
    'tidong-1'                : 'Kinnaur',
    'pong'                    : 'Kangra',
    'dehar'                   : 'Mandi',
    'sawra kuddu'             : 'Shimla',
    'kashang I'               : 'Kinnaur',
    'kutehr'                  : 'Chamba',
    'baspa'                   : 'Kinnaur',
    'Ranjit Sagar Dam (Thein)': 'Gurdaspur',
    'ghanvi'                  : 'Shimla',
    'larji'                   : 'Kullu',
    'Luhri Stage 1'           : 'Shimla',
    'Giri Bata'               : 'Sirmaur',
    'Shongtong Karcham'       : 'Kinnaur',
    # Uttarakhand
    'tehri'                   : 'Tehri Garhwal',
    'koteshwar'               : 'Tehri Garhwal',
    'dhauli ganga'            : 'Chamoli',
    'tanakpur'                : 'Champawat',
    'maneri bhali '           : 'Uttarkashi',
    'ramganga'                : 'Almora',
    'tiloth'                  : 'Uttarkashi',
    'vyasi'                   : 'Dehradun',
    'madhyamaheshwar'         : 'Rudraprayag',
    'singoli bhatwari'        : 'Rudraprayag',
    'Tapovan Vishnugad'       : 'Chamoli',
    'Vishnugad Pipalkoti'     : 'Chamoli',
    'Shrinagar'               : 'Pauri Garhwal',
    'Naitwar Mori'            : 'Uttarkashi',
    'Lata Tapovan'            : 'Chamoli',
    'Lakhwar'                 : 'Dehradun',
    'Kishau'                  : 'Uttarkashi',
    'Phata Byung'             : 'Rudraprayag',
    'Chibro'                  : 'Dehradun',
    'Khodri'                  : 'Dehradun',
    'chilla'                  : 'Haridwar',
    'vishnuprayag'            : 'Chamoli',
    'maneri_bhali_I'          : 'Uttarkashi',
    'Maneri Bhali Ii'         : 'Uttarkashi',
}

dam_df['district'] = dam_df['project_name'].map(DAM_DISTRICT_MAP)
unmatched = dam_df[dam_df['district'].isna()]['project_name'].tolist()
if unmatched:
    print(f"  WARNING: {len(unmatched)} dams with no district mapping: {unmatched}")

# ── 2. Load census files ───────────────────────────────────────────────────────
print("\nLoading census files …")

uk_pca = pd.read_csv(r'C:\Users\User\Downloads\f1a56af0-b9b0-43f8-98c4-f9b76351a379.csv', encoding='latin1')
hp_pca = pd.read_excel(r'C:\Users\User\Downloads\PCA0200_2011_MDDS.xls', engine='xlrd')

# Combine both states
combined_pca = pd.concat([uk_pca, hp_pca], ignore_index=True)

# Keep only DISTRICT level, Total TRU rows (not Rural/Urban splits)
census = combined_pca[
    (combined_pca['Level'] == 'DISTRICT') &
    (combined_pca['TRU'] == 'Total')
].copy()

census['Name_clean'] = census['Name'].str.strip().str.title()
print(f"  Districts in census data: {census['Name_clean'].tolist()}")

# ── 3. Compute census features per district ────────────────────────────────────
census['pop_per_household']    = census['Total Population Person'] / census['No of Households']
census['st_pop_fraction']      = census['Scheduled Tribes population Person'] / census['Total Population Person']
census['worker_ratio']         = census['Total Worker Population Person'] / census['Total Population Person']
census['household_size']       = census['Total Population Person'] / census['No of Households']
census['non_worker_dependency'] = census['Non Working Population Person'] / census['Total Population Person']
census['pop_density_district'] = census['Total Population Person']

CENSUS_FEATURES = ['Name_clean', 'pop_density_district', 'st_pop_fraction',
                   'worker_ratio', 'household_size', 'non_worker_dependency']
census_feats = census[CENSUS_FEATURES].rename(columns={'Name_clean': 'district'})

# ── 4. Join census to dams ─────────────────────────────────────────────────────
dam_df['district_clean'] = dam_df['district'].str.strip().str.title()
census_feats['district_clean'] = census_feats['district'].str.strip().str.title()

# Drop original 'district' from census_feats to avoid redundant column collision suffixes
census_feats = census_feats.drop(columns=['district'])

dam_census = dam_df.merge(census_feats, on='district_clean', how='left')

n_matched = dam_census['pop_density_district'].notna().sum()
print(f"\n  Census matched for {n_matched} / {len(dam_df)} dams")
if n_matched < len(dam_df):
    # Fixed KeyError by explicitly retrieving the valid mapped 'district_clean' or 'district' from dam_census
    missed = dam_census[dam_census['pop_density_district'].isna()][['project_name', 'district_clean']].values
    print(f"  Unmatched (district name mismatch):")
    for name, dist in missed:
        # Find closest census district name
        from difflib import get_close_matches
        close = get_close_matches(str(dist), census_feats['district_clean'].tolist(), n=1, cutoff=0.5)
        print(f"    {name} -> '{dist}' | closest match: {close}")

# ── 5. Tourism proximity features ─────────────────────────────────────────────
print("\nComputing tourism proximity features …")

PILGRIMAGE_SITES = {
    'Badrinath'      : (30.7433, 79.4938),
    'Kedarnath'      : (30.7346, 79.0669),
    'Gangotri'       : (30.9940, 78.9394),
    'Yamunotri'      : (31.0147, 78.4621),
    'Hemkund Sahib'  : (30.7083, 79.6483),
    'Vashisth Temple': (32.2640, 77.1936),  
    'Manikaran'      : (32.0292, 77.3429),  
    'Jwala Ji'       : (31.8726, 76.1553),  
    'Naina Devi'     : (31.2849, 76.5405),  
}

RESORT_HUBS = {
    'Manali'      : (32.2432, 77.1892),
    'Shimla'      : (31.1048, 77.1734),
    'Mussoorie'   : (30.4598, 78.0664),
    'Nainital'    : (29.3919, 79.4542),
    'Kullu'       : (31.9579, 77.1095),
    'Dharamsala'  : (32.2190, 76.3234),
    'Dalhousie'   : (32.5387, 75.9736),
    'Kasauli'     : (30.9024, 76.9668),
}

CHAR_DHAM_CORRIDOR_POINTS = [
    (30.0869, 78.2676),  
    (30.1295, 78.3390),  
    (30.3685, 78.7682),  
    (30.4133, 78.9629),  
    (30.5892, 79.0514),  
    (30.7433, 79.4938),  
    (30.7346, 79.0669),  
    (30.9940, 78.9394),  
    (31.0147, 78.4621),  
]

def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance between two lat/lon points in km."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def min_dist_to_sites(dam_lat, dam_lon, sites_dict):
    return min(haversine_km(dam_lat, dam_lon, lat, lon)
               for lat, lon in sites_dict.values())

def on_corridor(dam_lat, dam_lon, corridor_points, threshold_km=25):
    return int(any(haversine_km(dam_lat, dam_lon, lat, lon) <= threshold_km
                   for lat, lon in corridor_points))

tourism_rows = []
for _, row in dam_df.iterrows():
    lat, lon = row['Latitude'], row['Longitude']
    d_pilgrim = min_dist_to_sites(lat, lon, PILGRIMAGE_SITES)
    d_resort  = min_dist_to_sites(lat, lon, RESORT_HUBS)
    corridor  = on_corridor(lat, lon, CHAR_DHAM_CORRIDOR_POINTS)
    tourism_rows.append({
        'project_name'               : row['project_name'],
        'dist_to_nearest_pilgrim_km' : round(d_pilgrim, 2),
        'dist_to_nearest_resort_km'  : round(d_resort, 2),
        'on_char_dham_corridor'      : corridor,
    })

tourism_df = pd.DataFrame(tourism_rows)

def minmax(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

tourism_df['tourism_pressure_score'] = (
    0.50 * (1 - minmax(tourism_df['dist_to_nearest_pilgrim_km']))
  + 0.30 * (1 - minmax(tourism_df['dist_to_nearest_resort_km']))
  + 0.20 * tourism_df['on_char_dham_corridor']
).round(4)

# ── 6. Merge everything and save ───────────────────────────────────────────────
CENSUS_OUT_COLS = ['project_name', 'pop_density_district', 'st_pop_fraction',
                   'worker_ratio', 'household_size', 'non_worker_dependency']

final = (dam_df[['project_name','district']]
         .merge(dam_census[CENSUS_OUT_COLS], on='project_name', how='left')
         .merge(tourism_df, on='project_name', how='left'))

final.to_csv('dam_anthropogenic_features.csv', index=False)

print(f"\n{'='*60}")
print(f"Saved dam_anthropogenic_features.csv")
print(f"  Rows    : {final.shape[0]}")
print(f"  Columns : {final.shape[1] - 2} features (excl. project_name, district)")
print()
print("Census features (district-level 2011 PCA):")
print(final[['project_name','district','pop_density_district',
             'st_pop_fraction','worker_ratio','non_worker_dependency']].to_string(index=False))
print()
print("Tourism proximity features:")
print(tourism_df[['project_name','dist_to_nearest_pilgrim_km',
                   'dist_to_nearest_resort_km','on_char_dham_corridor',
                   'tourism_pressure_score']].to_string(index=False))