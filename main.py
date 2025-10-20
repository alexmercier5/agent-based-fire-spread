import rasterio
import matplotlib.pyplot as plt

# Path to your GeoTIFF file
tif_path = "./main.tif"

# Open with rasterio
with rasterio.open(tif_path) as src:
    data = src.read(1)  # Read the first band (use different index if multi-band)
    profile = src.profile  # metadata dictionary
    bounds = src.bounds
    res = src.res  # pixel size (xres, yres)
    crs = src.crs

print("CRS:", crs)
print("Pixel size (m):", res)
print("Shape (rows, cols):", data.shape)
print("Extent (minx, miny, maxx, maxy):", bounds)

# Plotting the raster data

plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='terrain', origin='upper')
plt.colorbar(label="Value")
plt.title("Raster Preview")
plt.xlabel("Column index")
plt.ylabel("Row index")
plt.show()

# Size of landscape

width_m = res[0] * data.shape[1]
height_m = abs(res[1]) * data.shape[0]

print(f"Landscape size: {width_m:.2f} m × {height_m:.2f} m")

