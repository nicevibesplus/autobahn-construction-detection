# ====================================================
# FILE: ndvi_calculation.py
# ====================================================
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm

def run_ndvi_calculation(raster_path, ndvi_path, red_band=1, nir_band=4):
    with rasterio.open(raster_path) as src:
        red = src.read(red_band).astype(np.float32)
        nir = src.read(nir_band).astype(np.float32)
        profile = src.profile

    denom = red + nir
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom

    profile.update(dtype=rasterio.float32, count=1)
    
    # Added tqdm progress bar loop to simulate/wrap the file writing operation
    with rasterio.open(ndvi_path, "w", **profile) as dst:
        for _ in tqdm(range(1), desc=f"Writing NDVI raster ({Path(ndvi_path).name})"):
            dst.write(ndvi.astype(rasterio.float32), 1)

    print(f"Done — wrote {ndvi_path}")