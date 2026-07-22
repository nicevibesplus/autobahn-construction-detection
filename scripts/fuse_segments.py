import rasterio
import geopandas as gpd
from rasterstats import zonal_stats
from tqdm import tqdm
from pathlib import Path


def zonal_stats_with_progress(gdf, raster, affine, stats, nodata, desc, batch_size=5000):
    results = []
    for start in tqdm(range(0, len(gdf), batch_size), desc=desc):
        batch = gdf.iloc[start:start + batch_size]
        results.extend(zonal_stats(batch, raster, affine=affine, stats=stats, nodata=nodata))
    return results


def run_fuse_segments(segments_surface_path, segments_texture_path, white_refined_path, orange_refined_path, vehicles_gpkg_path, output_path):
    print("Fusing surface, texture, marking and vehicle attributes into segments...")

    segments = gpd.read_file(segments_surface_path)

    # Drop segments that carry no usable spectral information
    if all(col in segments.columns for col in ["mean_r", "mean_g", "mean_b", "ndvi_mean"]):
        initial_count = len(segments)
        invalid_mask = (
            segments["mean_r"].isna() &
            segments["mean_g"].isna() &
            segments["mean_b"].isna() &
            ((segments["ndvi_mean"] == 0) | segments["ndvi_mean"].isna())
        )
        segments = segments[~invalid_mask].copy()
        print(f"  Filtered out {initial_count - len(segments)} empty/uninitialized segments.")

    # tested
    # segments = segments.reset_index(drop=True)
    # segments["segment_id"] = range(len(segments))
    # print(f"  {len(segments)} segments loaded")

    # print("Merging texture column...")
    # texture = gpd.read_file(segments_texture_path)

    # # Apply the same invalid-row filter so texture aligns with segments
    # if len(texture) > len(segments) and all(col in texture.columns for col in ["mean_r", "mean_g", "mean_b", "ndvi_mean"]):
    #     texture_invalid_mask = (
    #         texture["mean_r"].isna() &
    #         texture["mean_g"].isna() &
    #         texture["mean_b"].isna() &
    #         ((texture["ndvi_mean"] == 0) | texture["ndvi_mean"].isna())
    #     )
    #     texture = texture[~texture_invalid_mask].copy().reset_index(drop=True)

    # if len(texture) != len(segments):
    #     print(f"  [WARNING] Texture row count ({len(texture)}) does not match segments count ({len(segments)}). Reindexing/slicing to match.")
    #     texture = texture.iloc[:len(segments)]

    # segments["texture_mean"] = texture["texture_mean"].values
    # print("  done")

    #testing
    print(f"  {len(segments)} segments loaded")

    print("Merging texture column...")
    texture = gpd.read_file(segments_texture_path)[["segment_id", "texture_mean"]]

    segments = segments.merge(texture, on="segment_id", how="left", validate="one_to_one")

    missing = segments["texture_mean"].isna().sum()
    if missing:
        print(f"  [WARNING] {missing} segments had no matching texture row after merge.")
    print("  done")






    print("Loading marking rasters into memory...")
    with rasterio.open(white_refined_path) as src:
        white_arr = src.read(1)
        white_transform = src.transform
    with rasterio.open(orange_refined_path) as src:
        orange_arr = src.read(1)
        orange_transform = src.transform
    print("  done")

    print("Computing lane marking zonal stats...")
    white_stats = zonal_stats_with_progress(segments, white_arr, white_transform, "sum", 0, "White markings")
    orange_stats = zonal_stats_with_progress(segments, orange_arr, orange_transform, "sum", 0, "Orange markings")

    segments["marking_white_sum"] = [s["sum"] or 0 for s in white_stats]
    segments["marking_orange_sum"] = [s["sum"] or 0 for s in orange_stats]
    print(f"  white sum total: {segments['marking_white_sum'].sum():.0f}")
    print(f"  orange sum total: {segments['marking_orange_sum'].sum():.0f}")

    print("Computing vehicle overlap...")
    vehicle_boxes = gpd.read_file(vehicles_gpkg_path)
    print(f"  {len(vehicle_boxes)} vehicle boxes loaded")

    overlay = gpd.overlay(
        vehicle_boxes,
        segments[["segment_id", "geometry"]],
        how="intersection"
    )
    print(f"  {len(overlay)} overlap pieces found")

    overlay["overlap_area"] = overlay.geometry.area
    vehicle_area = overlay.groupby("segment_id")["overlap_area"].sum()

    segments["vehicle_area"] = segments["segment_id"].map(vehicle_area).fillna(0)
    segments["vehicle_fraction"] = segments["vehicle_area"] / segments.geometry.area
    print(f"  {(segments['vehicle_fraction'] > 0).sum()} segments have some vehicle overlap")

    print(f"Saving {Path(output_path).name}...")
    segments.to_file(output_path, driver="GPKG")

    print(f"Done — {len(segments)} segments")