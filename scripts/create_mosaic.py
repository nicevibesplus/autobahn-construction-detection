import os
import glob
import time
import shutil
import numpy as np
import rasterio
from rasterio.merge import merge
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from tqdm import tqdm


def run_mosaic_extraction(input_dir, output_dir, roads_gpkg, grid_split, road_search_buffer_m, master_mosaic_path):
    print(f"[{time.strftime('%H:%M:%S')}] Starting Master Mosaic Extraction Pipeline...")

    roads_gpkg = Path(roads_gpkg)
    output_dir = Path(output_dir)
    temp_tiles_dir = output_dir / "temp_tiles"
    master_mosaic_path = Path(master_mosaic_path)

    if not roads_gpkg.exists():
        print(f"[ERROR] Missing GPKG: {roads_gpkg}")
        return

    if master_mosaic_path.exists():
        print(f"\n[{time.strftime('%H:%M:%S')}] [INFO] Master image already exists: {master_mosaic_path.name}")
        print(f"[{time.strftime('%H:%M:%S')}] Skipping generation. Using existing file.")
        print(f"[{time.strftime('%H:%M:%S')}] PIPELINE COMPLETE!")
        return

    if temp_tiles_dir.exists():
        shutil.rmtree(temp_tiles_dir)
    temp_tiles_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase 1: Bounding Box Culling ---")

    gdf_all_roads = gpd.read_file(str(roads_gpkg), layer='gis_osm_roads_free')
    gdf_all_roads = gdf_all_roads[gdf_all_roads['fclass'].isin(['motorway','motorway_link'])]

    jp2_files = glob.glob(os.path.join(input_dir, "*.jp2"))
    if not jp2_files:
        print(f"[ERROR] No .jp2 files found in {input_dir}!")
        return

    with rasterio.open(jp2_files[0]) as first_src:
        global_crs = first_src.crs

    minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
    for jp2 in tqdm(jp2_files, desc="Collecting JP2 bounds"):
        with rasterio.open(jp2) as src:
            b = src.bounds
            minx, miny = min(minx, b.left), min(miny, b.bottom)
            maxx, maxy = max(maxx, b.right), max(maxy, b.top)
    total_image_bbox = box(minx, miny, maxx, maxy)

    gdf_projected_roads = gdf_all_roads.to_crs(global_crs)
    gdf_culled_roads = gpd.clip(gdf_projected_roads, total_image_bbox)
    print(f" -> Culled road network to just {len(gdf_culled_roads)} relevant segments.")

    gdf_search_buffer = gdf_culled_roads.copy()
    gdf_search_buffer['geometry'] = gdf_search_buffer.geometry.buffer(road_search_buffer_m)

    print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase 2: Surgical Tile Extraction ---")
    tile_counter = 0

    for idx, jp2_path in enumerate(tqdm(jp2_files, desc="Extracting tiles"), start=1):
        with rasterio.open(jp2_path) as src:
            orig_w, orig_h = src.width, src.height
            src_transform = src.transform

            tile_w = int(np.ceil(orig_w / grid_split))
            tile_h = int(np.ceil(orig_h / grid_split))

            for t_y in range(0, orig_h, tile_h):
                for t_x in range(0, orig_w, tile_w):
                    w, h = min(tile_w, orig_w - t_x), min(tile_h, orig_h - t_y)
                    quad_window = rasterio.windows.Window(t_x, t_y, w, h)
                    quad_bounds = rasterio.windows.bounds(quad_window, src_transform)
                    quad_box = box(*quad_bounds)

                    if gdf_search_buffer.intersects(quad_box).any():
                        tile_counter += 1
                        quad_transform = rasterio.windows.transform(quad_window, src_transform)

                        tile_data = src.read(window=quad_window)
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": h,
                            "width": w,
                            "transform": quad_transform
                        })

                        temp_path = temp_tiles_dir / f"relevant_tile_{tile_counter}.tif"
                        with rasterio.open(temp_path, "w", **out_meta) as dest:
                            dest.write(tile_data)

    if tile_counter == 0:
        print("[ERROR] No tiles intersected the road buffer. No master image created.")
        return

    print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase 3: Merging {tile_counter} Tiles to Master Image ---")

    temp_tif_paths = glob.glob(str(temp_tiles_dir / "*.tif"))
    srcs_to_mosaic = [rasterio.open(p) for p in tqdm(temp_tif_paths, desc="Opening tiles for merge")]

    mosaic, out_trans = merge(srcs_to_mosaic)
    out_meta = srcs_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    with rasterio.open(master_mosaic_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in tqdm(srcs_to_mosaic, desc="Closing tile sources"):
        src.close()

    shutil.rmtree(temp_tiles_dir)
    print(f"[{time.strftime('%H:%M:%S')}] Master image successfully created: {master_mosaic_path.name}")
