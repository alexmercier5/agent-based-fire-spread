import rasterio
import numpy as np
import os

fuel_band_index = 2  # assuming band 4 represents fuel


script_dir = os.path.dirname(os.path.abspath(__file__))
tif_path = os.path.join(script_dir, "8900main.tif")
with rasterio.open(tif_path) as src:
    fuel_data = src.read(fuel_band_index)
    meta = src.tags(fuel_band_index)

print(meta)
print("Data type:", fuel_data.dtype)
print("Unique values:", np.unique(fuel_data))
print("Length of unique values:", len(np.unique(fuel_data)))
