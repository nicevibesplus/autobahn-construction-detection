# ====================================================
# FILE: scripts/quickshift_segmentation.py
# ====================================================
import rasterio
import rasterio.features
import numpy as np
import cv2
import geopandas as gpd
from skimage.segmentation import slic
from shapely.geometry import Polygon
from pathlib import Path
from tqdm import tqdm

def run_quickshift_segmentation(input_raster, output_gpkg, n_segments=10000, compactness=10.0):
    print("Opening raster for SLIC superpixel segmentation...")
    with rasterio.open(input_raster) as src:
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
        transform = src.transform
        crs = src.crs

    img = np.dstack([r, g, b])
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    print("Running SLIC superpixel segmentation (fast alternative)...")
    # SLIC is dramatically faster than quickshift for large geospatial rasters
    segments_ids = slic(img, n_segments=n_segments, compactness=compactness, start_label=1, channel_axis=2)

    print("Vectorizing segmentation mask...")
    geoms = []
    shapes_generator = list(rasterio.features.shapes(segments_ids.astype(np.int32), transform=transform))
    
    for geom_dict, label_val in tqdm(shapes_generator, desc="Converting segments to polygons"):
        if label_val == 0:
            continue
        polygon = Polygon(geom_dict['coordinates'][0])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            geoms.append({
                "geometry": polygon,
                "segment_id": int(label_val)
            })

    print(f"Creating GeoDataFrame with {len(geoms)} segments...")
    gdf = gpd.GeoDataFrame(geoms, crs=crs)
    
    for _ in tqdm(range(1), desc=f"Saving segmentation GPKG ({Path(output_gpkg).name})"):
        gdf.to_file(output_gpkg, driver="GPKG")

    print(f"Done — segmentation vector saved to {output_gpkg}")