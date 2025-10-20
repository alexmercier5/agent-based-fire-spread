import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import numpy as np

def read_tif(tif_path):
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

def resample_tif(tif_path, out_path, target_pixel_size=100):
    """
    Resample a GeoTIFF to a new pixel size and save as a new file.

    Parameters:
        tif_path (str): Path to input GeoTIFF.
        out_path (str): Path to save the resampled GeoTIFF.
        target_pixel_size (float): Desired pixel size in meters (default 100).
    """

    with rasterio.open(tif_path) as src:
        # Calculate scale factor
        scale_factor = target_pixel_size / src.res[0]  # assuming square pixels

        new_width = int(src.width / scale_factor)
        new_height = int(src.height / scale_factor)

        # Resample data
        data_resampled = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.average
        )

        # Update metadata for the new file
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        profile = src.profile.copy()
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': transform
        })

        # Write new GeoTIFF
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(data_resampled, 1)

        print(f"Resampled GeoTIFF saved to {out_path}")
        print("Resampled shape:", data_resampled.shape)


if __name__ == "__main__":
    # Path to your GeoTIFF file
    tif_path = "./main.tif"
    out_path = "./resampled_main.tif"

    read_tif(out_path)
    read_tif(tif_path)
    #resample_tif(tif_path, out_path)