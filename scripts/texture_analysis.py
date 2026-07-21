import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path
from tqdm import tqdm


def run_texture_analysis(raster_path, segments_path, output_path):
    print("Running texture analysis per segment...")

    segments = gpd.read_file(segments_path)

    with rasterio.open(raster_path) as src:
        texture_values = []
        for geom in tqdm(segments.geometry, desc="Analyzing texture per segment"):
            try:
                out_image, _ = mask(src, [geom], crop=True, indexes=1)
                window = out_image.astype(np.uint8)
                if window.size < 4:
                    texture_values.append(np.nan)
                    continue
                glcm = graycomatrix(window, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                texture_values.append(graycoprops(glcm, "contrast")[0, 0])
            except ValueError:
                texture_values.append(np.nan)

    segments["texture_mean"] = texture_values

    print(f"Saving texture analysis to {Path(output_path).name}...")
    segments.to_file(output_path, driver="GPKG")

    print(f"Done — wrote {output_path}")