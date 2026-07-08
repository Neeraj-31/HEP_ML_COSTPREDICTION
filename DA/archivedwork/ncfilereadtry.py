
import xarray as xr
import matplotlib.pyplot as plt

# 1. Load the NetCDF file
file_path = r'C:\Users\User\.vscode\HEP_ML\DA\Rainfall\nc_yearwise\rainfall_2024.nc'
ds = xr.open_dataset(file_path)

# 2. Select a representative day (e.g., Day 215 is during the peak monsoon season)
# Change 215 to any day from 0 to 365
day_index = 215
single_day = ds['rain'].isel(time=day_index)

# 3. Plot the spatial map
plt.figure(figsize=(10, 8))

# 'robust=True' solves the color-washing issue by ignoring extreme outliers.
# 'cmap' sets a beautiful, industry-standard color progression for rainfall.
single_day.plot(robust=True, cmap='YlGnBu')

plt.title(f"Daily Rainfall Map - Day {day_index} of 2024", fontsize=14, fontweight='bold')
plt.xlabel("Longitude (°E)", fontsize=12)
plt.ylabel("Latitude (°N)", fontsize=12)
plt.tight_layout()

# 4. Save or display the map
plt.savefig('accurate_rainfall_map_xarray.png', dpi=300)
plt.show()