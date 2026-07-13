import os
import glob
import cv2
import time
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.ops import substring, unary_union
from sklearn.cluster import KMeans
from scipy.ndimage import distance_transform_edt
from pathlib import Path

# ----------------------------------------------------
# 0. PATHS & CONFIGURATION
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
MODELS_DIR = BASE_DIR / "models"
ROADS_GPKG = BASE_DIR / "data" / "muenster-regbez.gpkg"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["ULTRALYTICS_CONFIG_DIR"] = str(MODELS_DIR / "ultralytics_config")

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# PERFORMANCE OPTIMIZATION TUNING:
# 2 = 2x2 split (4 tiles), 3 = 3x3 split (9 tiles)
GRID_SPLIT = 4 
YOLO_PATCH_SIZE = 640

def split_line_into_10m_segments(line_geom, segment_length=10.0):
    """Splits a Shapely LineString into 10-meter segments."""
    segments = []
    line_len = line_geom.length
    for i in np.arange(0, line_len, segment_length):
        sub_seg = substring(line_geom, i, min(i + segment_length, line_len))
        if not sub_seg.is_empty:
            segments.append(sub_seg)
    return segments

def main():
    device = "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] Starting Tiled Global-Aggregation Vector Feature Pipeline...")
    print(f"Performance settings: GRID_SPLIT={GRID_SPLIT} ({GRID_SPLIT**2} tiles per orthophoto)")
    
    if not ROADS_GPKG.exists():
        print(f"[ERROR] Missing GPKG: {ROADS_GPKG}")
        return

    roads_out_path = OUTPUT_DIR / "master_layer_active_road.gpkg"
    construction_out_path = OUTPUT_DIR / "master_layer_construction.gpkg"
    cars_out_path = OUTPUT_DIR / "master_layer_vehicles.gpkg"

    for path in [roads_out_path, construction_out_path, cars_out_path]:
        if path.exists(): path.unlink()

    # Global registers across ALL tiles and files
    global_segment_registry = {}  
    global_vehicles_list = []
    global_crs = None

    # 1. LOAD MODEL & VECTOR BASE
    print(f"\n[{time.strftime('%H:%M:%S')}] --- Step 1: Initializing Assets ---")
    yolo_weights_path = hf_hub_download(repo_id="dronefreak/visdrone-yolov8s", filename="best.pt", cache_dir=MODELS_DIR / "yolo")
    yolo_model = YOLO(yolo_weights_path)
    
    gdf_all_roads = gpd.read_file(str(ROADS_GPKG), layer='gis_osm_roads_free')
    gdf_all_roads = gdf_all_roads[gdf_all_roads['fclass'] == 'motorway']
    
    jp2_files = glob.glob(os.path.join(INPUT_DIR, "*.jp2"))
    if not jp2_files:
        print(f"[ERROR] No .jp2 files found in {INPUT_DIR}!")
        return

    # ----------------------------------------------------
    # PHASE 1: GLOBAL DATA HARVESTING VIA QUADRANT TILING
    # ----------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase 1: Harvesting Data with Grid Tiling ---")
    
    for idx, jp2_path in enumerate(jp2_files, start=1):
        file_name = os.path.basename(jp2_path)
        print("\n" + "="*80)
        print(f"[{idx}/{len(jp2_files)}] LOADING MASTER TILE MAP SHEET: {file_name}")
        print("="*80)
        
        try:
            with rasterio.open(jp2_path) as src:
                orig_w, orig_h = src.width, src.height
                src_crs = src.crs
                src_transform = src.transform
                if global_crs is None: global_crs = src_crs
                
                # Dynamically calculate the pixel sizes of our sub-tile grid divisions
                tile_w = int(np.ceil(orig_w / GRID_SPLIT))
                tile_h = int(np.ceil(orig_h / GRID_SPLIT))
                
                sub_tile_idx = 0
                total_sub_tiles = GRID_SPLIT * GRID_SPLIT

                for t_y in range(0, orig_h, tile_h):
                    for t_x in range(0, orig_w, tile_w):
                        sub_tile_idx += 1
                        
                        # Handle exact pixel window crop boundaries
                        w = min(tile_w, orig_w - t_x)
                        h = min(tile_h, orig_h - t_y)
                        quad_window = rasterio.windows.Window(t_x, t_y, w, h)
                        
                        # Compute the spatial coordinate bounds and transform mapping of this specific sub-tile quadrant
                        quad_bounds = rasterio.windows.bounds(quad_window, src_transform)
                        quad_box = box(*quad_bounds)
                        quad_transform = rasterio.windows.transform(quad_window, src_transform)

                        # Strict crop of vector layer centerlines to this sub-tile's bounding box
                        gdf_local = gdf_all_roads.to_crs(src_crs)
                        gdf_local = gdf_local[gdf_local.geometry.intersects(quad_box)].copy()
                        if gdf_local.empty:
                            continue

                        print(f" -> [{sub_tile_idx}/{total_sub_tiles}] Processing Quadrant Sub-Tile (Width: {w}px, Height: {h}px)...")

                        # Read only the selected sub-tile pixel data window from disk into memory
                        rgb_quad = src.read([1, 2, 3], window=quad_window)
                        rgb_quad = np.moveaxis(rgb_quad, 0, -1)
                        gray_quad = cv2.cvtColor(rgb_quad, cv2.COLOR_RGB2GRAY)

                        # Generate unique deterministic segment keys for sections passing inside this window
                        segment_lines = []
                        for _, road_row in gdf_local.iterrows():
                            osm_id = road_row.get('osm_id', 'road')
                            # Clip the geometry strictly to the quadrant boundary to keep calculations accurate
                            clipped_geom = road_row.geometry.intersection(quad_box)
                            
                            if clipped_geom.is_empty:
                                continue
                                
                            ten_meter_lines = split_line_into_10m_segments(clipped_geom, segment_length=10.0)
                            for seg_idx, line_seg in enumerate(ten_meter_lines):
                                unique_key = f"{osm_id}_{seg_idx}"
                                segment_lines.append((line_seg, unique_key))

                        if not segment_lines:
                            continue

                        # High-Speed Raster Voronoi Allocation on the smaller sub-tile array
                        shapes = [(line, i) for i, (line, _) in enumerate(segment_lines)]
                        local_id_map = features.rasterize(
                            shapes=shapes, out_shape=(h, w),
                            transform=quad_transform, fill=-1, dtype=np.int32
                        )
                        
                        distances, indices = distance_transform_edt((local_id_map == -1), return_distances=True, return_indices=True)
                        nearest_indices = local_id_map[indices[0], indices[1]]
                        pixel_cutoff = 10.0 / quad_transform[0] 
                        local_id_map = np.where(distances <= pixel_cutoff, nearest_indices, -1)

                        # Extract metrics out of the sub-tile
                        for local_idx, (line, unique_key) in enumerate(segment_lines):
                            pixel_coords = (local_id_map == local_idx)
                            if not np.any(pixel_coords): 
                                continue

                            geom_gen = features.shapes(np.where(pixel_coords, np.uint8(1), np.uint8(0)), mask=pixel_coords, transform=quad_transform)
                            poly_geom = Polygon(next(geom_gen)[0]['coordinates'][0])

                            if unique_key not in global_segment_registry:
                                global_segment_registry[unique_key] = {"polys": [], "gray": [], "red": [], "blue": [], "cars": 0}

                            global_segment_registry[unique_key]["polys"].append(poly_geom)
                            global_segment_registry[unique_key]["gray"].extend(gray_quad[pixel_coords].tolist())
                            global_segment_registry[unique_key]["red"].extend(rgb_quad[:, :, 0][pixel_coords].tolist())
                            global_segment_registry[unique_key]["blue"].extend(rgb_quad[:, :, 2][pixel_coords].tolist())

                        # Micro-Scale YOLO Vehicle Scanning (640x640 loops inside the sub-tile window)
                        for y_yolo in range(0, h, YOLO_PATCH_SIZE):
                            for x_yolo in range(0, w, YOLO_PATCH_SIZE):
                                w_yolo = min(YOLO_PATCH_SIZE, w - x_yolo)
                                h_yolo = min(YOLO_PATCH_SIZE, h - y_yolo)
                                
                                if np.all(local_id_map[y_yolo:y_yolo+h_yolo, x_yolo:x_yolo+w_yolo] == -1): 
                                    continue

                                bgr_chunk = cv2.cvtColor(rgb_quad[y_yolo:y_yolo+h_yolo, x_yolo:x_yolo+w_yolo], cv2.COLOR_RGB2BGR)
                                if h_yolo < YOLO_PATCH_SIZE or w_yolo < YOLO_PATCH_SIZE:
                                    padded = np.zeros((YOLO_PATCH_SIZE, YOLO_PATCH_SIZE, 3), dtype=bgr_chunk.dtype)
                                    padded[:h_yolo, :w_yolo, :] = bgr_chunk
                                    bgr_chunk = padded

                                yolo_results = yolo_model.predict(source=bgr_chunk, device=device, verbose=False)[0]
                                for box_obj in yolo_results.boxes:
                                    x1, y1, x2, y2 = map(int, box_obj.xyxy[0].cpu().numpy())
                                    if yolo_results.names[int(box_obj.cls[0].cpu().numpy())] in ['car', 'van', 'truck', 'bus']:
                                        cx, cy = x_yolo + int((x1 + x2) / 2), y_yolo + int((y1 + y2) / 2)
                                        if cx >= w or cy >= h: 
                                            continue

                                        target_local_idx = local_id_map[cy, cx]
                                        if target_local_idx != -1:
                                            u_key = segment_lines[target_local_idx][1]
                                            global_segment_registry[u_key]["cars"] += 1
                                            
                                            # Map patch coordinates directly back to world coordinates via quad_transform
                                            gx1, gy1 = quad_transform * (x_yolo + x1, y_yolo + y1)
                                            gx2, gy2 = quad_transform * (x_yolo + x2, y_yolo + y2)
                                            global_vehicles_list.append({"geometry": box(gx1, gy2, gx2, gy1), "type": "vehicle"})

        except Exception as e:
            print(f" -> [ERROR] Failed to compile tiled loop for sheet {file_name}: {e}")

    # ----------------------------------------------------
    # PHASE 2: GLOBAL FEATURE FUSION & DISSOLVE
    # ----------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase 2: Processing Global Feature Fusion & Dissolving Edges ---")
    
    final_features_list = []
    final_metadata_list = []

    for unique_key, data in global_segment_registry.items():
        if not data["polys"]: continue

        # Unified Edge-Dissolve: Merge boundary clipped sub-tile elements seamlessly
        unified_geom = unary_union(data["polys"])
        
        arr_gray = np.array(data["gray"], dtype=np.float32)
        arr_red = np.array(data["red"], dtype=np.float32)
        arr_blue = np.array(data["blue"], dtype=np.float32)
        arr_blue[arr_blue == 0] = 1.0

        mean_gray = np.mean(arr_gray) if len(arr_gray) > 0 else 0
        std_gray = np.std(arr_gray) if len(arr_gray) > 0 else 0
        soil_idx = np.mean(arr_red) / np.mean(arr_blue) if len(arr_red) > 0 else 1.0

        feat_vector = [mean_gray, soil_idx, std_gray]
        final_features_list.append(feat_vector)
        
        final_metadata_list.append({
            "geometry": unified_geom,
            "cars_counted": data["cars"],
            "features": feat_vector,
            "cluster_label": None
        })

    if not final_features_list:
        print("[ERROR] No valid segment data could be compiled globally. Terminating.")
        return

    # ----------------------------------------------------
    # PHASE 3: GLOBAL UNSUPERVISED CLUSTERING & EXPORT
    # ----------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase 3: Executing Unsupervised Grouping and Export ---")
    
    X = np.array(final_features_list)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
    
    for idx, cluster_label in enumerate(kmeans.labels_):
        final_metadata_list[idx]["cluster_label"] = cluster_label

    cars_c0 = sum(m["cars_counted"] for m in final_metadata_list if m["cluster_label"] == 0)
    cars_c1 = sum(m["cars_counted"] for m in final_metadata_list if m["cluster_label"] == 1)
    
    active_cluster, construction_cluster = (0, 1) if cars_c0 >= cars_c1 else (1, 0)

    active_road_geoms = [{"geometry": m["geometry"]} for m in final_metadata_list if m["cluster_label"] == active_cluster]
    construction_geoms = [{"geometry": m["geometry"]} for m in final_metadata_list if m["cluster_label"] == construction_cluster]

    if global_crs is None: 
        global_crs = "EPSG:25832"

    if active_road_geoms:
        gpd.GeoDataFrame(active_road_geoms, crs=global_crs).to_file(str(roads_out_path), layer="active_roads", driver="GPKG")
        print(f" -> Saved: {roads_out_path.name} ({len(active_road_geoms)} features)")
    if construction_geoms:
        gpd.GeoDataFrame(construction_geoms, crs=global_crs).to_file(str(construction_out_path), layer="construction_zones", driver="GPKG")
        print(f" -> Saved: {construction_out_path.name} ({len(construction_geoms)} features)")
    if global_vehicles_list:
        gpd.GeoDataFrame(global_vehicles_list, crs=global_crs).to_file(str(cars_out_path), layer="detected_vehicles", driver="GPKG")
        print(f" -> Saved: {cars_out_path.name} ({len(global_vehicles_list)} features)")

    print(f"\n[{time.strftime('%H:%M:%S')}] PIECEWISE GRID TILING WORK FINISHED! ALL EDGE SEGMENTS FIXED.")

if __name__ == "__main__":
    main()