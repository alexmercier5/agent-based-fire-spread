import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.transform import Affine
import numpy as np
import os
import math

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

def resample_tif(tif_path, out_path, target_pixel_size=100.0, snap_to_grid=True, resampling=Resampling.average):
    """
    Resample all bands of a multi-band GeoTIFF to exactly `target_pixel_size` x `target_pixel_size`.

    Parameters:
        tif_path (str): input GeoTIFF path
        out_path (str): output GeoTIFF path
        target_pixel_size (float): desired pixel size in same linear units as CRS (meters)
        snap_to_grid (bool): if True, snaps the left/top origin to nearest multiple of target_pixel_size
        resampling: rasterio.enums.Resampling method for continuous data (use Resampling.nearest for categorical)
    """
    with rasterio.open(tif_path) as src:
        # original bounds and CRS
        left, bottom, right, top = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
        crs = src.crs
        orig_xres, orig_yres = src.res  # note yres is negative in transform but src.res returns positive tuple

        # optionally snap origin to a grid multiple of target_pixel_size
        if snap_to_grid:
            # snap left to floor(left / target) * target
            snap_left = math.floor(left / target_pixel_size) * target_pixel_size
            # snap top to ceil(top / target) * target  (so top >= original top and aligns on grid)
            snap_top = math.ceil(top / target_pixel_size) * target_pixel_size
            # Keep right/bottom such that extent at least covers original
            snap_right = snap_left + math.ceil((right - snap_left) / target_pixel_size) * target_pixel_size
            snap_bottom = snap_top - math.ceil((snap_top - bottom) / target_pixel_size) * target_pixel_size
        else:
            snap_left, snap_top, snap_right, snap_bottom = left, top, right, bottom

        # compute new width/height from snapped extent
        new_width = int(round((snap_right - snap_left) / target_pixel_size))
        new_height = int(round((snap_top - snap_bottom) / target_pixel_size))

        if new_width <= 0 or new_height <= 0:
            raise ValueError("Computed new_width/new_height <= 0. Check bounds and target_pixel_size")

        print(f"Original pixel size: {orig_xres} x {orig_yres}")
        print(f"Snapped extent: left={snap_left}, bottom={snap_bottom}, right={snap_right}, top={snap_top}")
        print(f"New grid: {new_width} cols × {new_height} rows at {target_pixel_size} m")

        # Prepare array to hold resampled bands
        dtype = src.dtypes[0]
        data_resampled = np.empty((src.count, new_height, new_width), dtype=dtype)

        # compute new transform: top-left at (snap_left, snap_top)
        transform = Affine.translation(snap_left, snap_top) * Affine.scale(target_pixel_size, -target_pixel_size)

        # Read/resample each band
        for i in range(1, src.count + 1):
            band = src.read(
                i,
                out_shape=(new_height, new_width),
                resampling=resampling
            )
            data_resampled[i - 1] = band

        # Build profile for output
        profile = src.profile.copy()
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": transform,
            "crs": crs,
            "count": src.count,
            "driver": "GTiff"
        })

        # preserve nodata if present
        if 'nodata' in src.profile and src.profile['nodata'] is not None:
            profile['nodata'] = src.profile['nodata']

        # write
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data_resampled)

        print(f"✅ Resampled GeoTIFF saved to: {out_path}")
        print("Shape (bands, rows, cols):", data_resampled.shape)
        print("New pixel size (reported):", (target_pixel_size, target_pixel_size))
        print("Output bounds:", dst.bounds)



if __name__ == "__main__":
    # Path to your GeoTIFF file
    tif_path = "./8900main.tif"
    out_path = "./resampled_main.tif"

    read_tif(tif_path)
    # read_tif(out_path)
    resample_tif(tif_path, out_path)

    read_tif(out_path)