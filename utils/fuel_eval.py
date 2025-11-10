import rasterio
import numpy as np

fuel_band_index = 4  # assuming band 4 represents fuel

with rasterio.open("8900main.tif") as src:
    fuel_data = src.read(fuel_band_index)

print("Data type:", fuel_data.dtype)
print("Unique values:", np.unique(fuel_data))
