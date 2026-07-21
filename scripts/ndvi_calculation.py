import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm


def run_ndvi_calculation(raster_path, ndvi_path, red_band=1, nir_band=4):
    print("Calculating NDVI...")

    with rasterio.open(raster_path) as src:
        red = src.read(red_band).astype(np.float32)
        nir = src.read(nir_band).astype(np.float32)
        profile = src.profile

    denom = red + nir
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom

    profile.update(dtype=rasterio.float32, count=1)
    print(f"Writing NDVI raster to {Path(ndvi_path).name}...")
    with rasterio.open(ndvi_path, "w", **profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)

    print(f"Done — wrote {ndvi_path}")