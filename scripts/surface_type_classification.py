import rasterio
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
from sklearn.cluster import KMeans
from pathlib import Path
from tqdm import tqdm
import joblib


def zonal_stats_with_progress(gdf, raster, affine, stats, nodata, desc):
    BATCH_SIZE = 5000
    results = []
    for start in tqdm(range(0, len(gdf), BATCH_SIZE), desc=desc):
        batch = gdf.iloc[start:start + BATCH_SIZE]
        results.extend(zonal_stats(batch, raster, affine=affine, stats=stats, nodata=nodata))
    return results


def run_surface_classification(raster_path, ndvi_path, segments_path, output_path, vegetation_threshold=0.3, kmeans_model_path=None, fit_new_model=False):
    print("Running surface type classification...")

    segments = gpd.read_file(segments_path)
    print(f"Loaded {len(segments)} segments")

    with rasterio.open(raster_path) as src:
        transform = src.transform
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)

    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)
        ndvi_transform = src.transform

    r_stats = zonal_stats_with_progress(segments, red, transform, "mean", 0, "Red band")
    g_stats = zonal_stats_with_progress(segments, green, transform, "mean", 0, "Green band")
    b_stats = zonal_stats_with_progress(segments, blue, transform, "mean", 0, "Blue band")
    ndvi_stats = zonal_stats_with_progress(segments, ndvi, ndvi_transform, "mean", -9999, "NDVI")

    segments["mean_r"] = [s["mean"] for s in r_stats]
    segments["mean_g"] = [s["mean"] for s in g_stats]
    segments["mean_b"] = [s["mean"] for s in b_stats]
    segments["ndvi_mean"] = [s["mean"] for s in ndvi_stats]

    segments["is_vegetation"] = segments["ndvi_mean"] > vegetation_threshold
    print(f"Flagged {segments['is_vegetation'].sum()} of {len(segments)} objects as vegetation")


    #uncommented for testing
    # non_veg = segments[~segments["is_vegetation"]].copy()
    # features = non_veg[["mean_r", "mean_g", "mean_b"]].fillna(0)

    # kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

    # non_veg["surface_class"] = kmeans.fit_predict(features)


    #testing
    non_veg = segments[~segments["is_vegetation"]].copy()
    features = non_veg[["mean_r", "mean_g", "mean_b"]].fillna(0)

    kmeans_model_path = Path(kmeans_model_path)

    if fit_new_model or not kmeans_model_path.exists():
        print(f"Fitting new surface KMeans model -> {kmeans_model_path.name}")
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(features)
        kmeans_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(kmeans, kmeans_model_path)
    else:
        print(f"Loading existing surface KMeans model from {kmeans_model_path.name}")
        kmeans = joblib.load(kmeans_model_path)

    non_veg["surface_class"] = kmeans.predict(features)   # predict, not fit_predict



    segments["surface_class"] = -1
    segments.loc[non_veg.index, "surface_class"] = non_veg["surface_class"]

    print(f"Saving classified segments to {Path(output_path).name}...")
    segments.to_file(output_path, driver="GPKG")

    print(f"Done — wrote {output_path}")