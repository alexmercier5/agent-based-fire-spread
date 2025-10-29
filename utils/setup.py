import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import numpy as np
import os

def read_tif(tif_path):
    print("Current working directory:", os.getcwd())
    print("Absolute path to TIFF:", os.path.abspath(tif_path))
    # Open with rasterio
    with rasterio.open(tif_path) as src:
        data = src.read(1)  # Read the first band
        data = src.read(2)  # Read the band, can be 1 through 10 for each tif layer
        print(data)
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

import rasterio
from rasterio.enums import Resampling

def resample_tif(tif_path, out_path, target_pixel_size=100):
    """
    Resample all bands of a multi-band GeoTIFF to a new pixel size and save as a new file.

    Parameters:
        tif_path (str): Path to input GeoTIFF.
        out_path (str): Path to save the resampled GeoTIFF.
        target_pixel_size (float): Desired pixel size in meters (default 100).
    """

    with rasterio.open(tif_path) as src:
        scale_factor = target_pixel_size / src.res[0]  # assuming square pixels

        new_width = int(src.width / scale_factor)
        new_height = int(src.height / scale_factor)

        # Read and resample all bands
        data_resampled = []
        for i in range(1, src.count + 1):
            band = src.read(
                i,
                out_shape=(new_height, new_width),
                resampling=Resampling.average
            )
            data_resampled.append(band)

        # Stack bands back into a single array with shape (bands, rows, cols)
        data_resampled = np.stack(data_resampled, axis=0)

        # Update metadata for the new file
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        profile = src.profile.copy()
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': transform,
            'count': src.count
        })

        # Write all bands to new GeoTIFF
        with rasterio.open(out_path, 'w', **profile) as dst:
            for i in range(src.count):
                dst.write(data_resampled[i], i + 1)
        print(data_resampled[3])
        print(f"Resampled GeoTIFF saved to {out_path}")
        print("Resampled shape (bands, rows, cols):", data_resampled.shape)



if __name__ == "__main__":
    # Path to your GeoTIFF file
    tif_path = "./main.tif"
    out_path = "./resampled_main.tif"

    read_tif(tif_path)
    # read_tif(out_path)
    resample_tif(tif_path, out_path)