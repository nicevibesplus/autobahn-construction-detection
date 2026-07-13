import os
import glob
import cv2
import time
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import substring
from sklearn.cluster import KMeans
from scipy.ndimage import distance_transform_edt
from pathlib import Path

# ----------------------------------------------------
# 0. PATHS & CONFIGURATION
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
PATCHES_DIR = OUTPUT_DIR / "patches"
MODELS_DIR = BASE_DIR / "models"
ROADS_GPKG = BASE_DIR / "data" / "muenster-regbez.gpkg"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PATCHES_DIR.mkdir(parents=True, exist_ok=True)
os.environ["ULTRALYTICS_CONFIG_DIR"] = str(MODELS_DIR / "ultralytics_config")

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

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
    print(f"[{time.strftime('%H:%M:%S')}] Starting Autobahn Construction Detection Pipeline...")
    print(f"Using device for YOLO: {device}")

    if not ROADS_GPKG.exists():
        print(f"[ERROR] Missing GPKG: {ROADS_GPKG}")
        return

    # 1. LOAD MODEL
    t_start = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] --- Step 1: Initializing Vehicle Model ---")
    yolo_weights_path = hf_hub_download(
        repo_id="dronefreak/visdrone-yolov8s", 
        filename="best.pt", 
        cache_dir=MODELS_DIR / "yolo"
    )
    yolo_model = YOLO(yolo_weights_path)
    print(f"[{time.strftime('%H:%M:%S')}] YOLO Model loaded successfully in {time.time() - t_start:.2f}s.")

    # 2. LOAD HIGHWAY LAYER
    t_start = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] --- Step 2: Loading Autobahn Vector Database ---")
    gdf_all_roads = gpd.read_file(str(ROADS_GPKG), layer='gis_osm_roads_free')
    
    # Isolate motorway geometries
    gdf_all_roads = gdf_all_roads[gdf_all_roads['fclass'] == 'motorway']
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(gdf_all_roads)} motorway features in {time.time() - t_start:.2f}s.")

    # Find raw files to analyze
    jp2_files = glob.glob(os.path.join(INPUT_DIR, "*.jp2"))
    if not jp2_files:
        print(f"[ERROR] No .jp2 files found in {INPUT_DIR}!")
        return
        
    print(f"[{time.strftime('%H:%M:%S')}] Found {len(jp2_files)} JP2 files to process.")
    patch_size = 640  

    for idx, jp2_path in enumerate(jp2_files, start=1):
        file_name = os.path.basename(jp2_path)
        base_name = os.path.splitext(file_name)[0]
        
        print("\n" + "="*80)
        print(f"[{idx}/{len(jp2_files)}] STARTING WORK ON: {file_name}")
        print("="*80)
        
        tile_start_time = time.time()

        try:
            with rasterio.open(jp2_path) as src:
                orig_w, orig_h = src.width, src.height
                src_crs = src.crs
                src_transform = src.transform
                input_profile = src.profile.copy()
                
                print(f"[{time.strftime('%H:%M:%S')}] Orthophoto size: {orig_w}x{orig_h} | CRS: {src_crs}")
                
                # Spatial crop of the vector roads to the current tile limits
                print(f"[{time.strftime('%H:%M:%S')}] Cropping motorway layer to image boundaries...")
                img_box = box(*src.bounds)
                gdf_local = gdf_all_roads.to_crs(src_crs)
                gdf_local = gdf_local[gdf_local.geometry.intersects(img_box)].copy()

                if gdf_local.empty:
                    print(f"[{time.strftime('%H:%M:%S')}] -> No vector highways cross this tile. Skipping.")
                    continue

                # Read full image arrays
                t_io = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Loading high-res image bands into memory...")
                rgb_full = src.read([1, 2, 3])
                rgb_full = np.moveaxis(rgb_full, 0, -1)
                gray_full = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2GRAY)
                bgr_full = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)
                print(f"[{time.strftime('%H:%M:%S')}] Loaded full resolution layers in {time.time() - t_io:.2f}s.")

                # ----------------------------------------------------
                # PHASE A: 10M SEGMENT SPLITTING
                # ----------------------------------------------------
                t_split = time.time()
                print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase A: Dividing Highways into 10-Meter Segments ---")
                segment_lines = [] 
                segment_counter = 0

                for _, road_row in gdf_local.iterrows():
                    line_geom = road_row.geometry
                    ten_meter_lines = split_line_into_10m_segments(line_geom, segment_length=10.0)
                    
                    for line_seg in ten_meter_lines:
                        segment_lines.append((line_seg, segment_counter))
                        segment_counter += 1

                print(f"[{time.strftime('%H:%M:%S')}] Generated {len(segment_lines)} individual 10m centerline elements in {time.time() - t_split:.2f}s.")

                if not segment_lines:
                    print(f"[{time.strftime('%H:%M:%S')}] -> Zero segments matched. Skipping.")
                    continue

                # ----------------------------------------------------
                # PHASE B: ZERO-OVERLAP RASTER VORONOI ALLOCATION (10M MAX WIDTH)
                # ----------------------------------------------------
                t_vor = time.time()
                print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase B: Calculating Non-Overlapping Voronoi Lanes ---")
                
                # Draw lines as 1-pixel markers on a master ID matrix
                print(f"[{time.strftime('%H:%M:%S')}]   * Painting centerlines onto master raster...")
                shapes = [(line, seg_id) for line, seg_id in segment_lines]
                master_id_map = features.rasterize(
                    shapes=shapes,
                    out_shape=(orig_h, orig_w),
                    transform=src_transform,
                    fill=-1,
                    dtype=np.int32
                )

                # Compute Euclidean Distance Transform using SciPy
                print(f"[{time.strftime('%H:%M:%S')}]   * Running Euclidean Distance Transform (SciPy EDT)...")
                mask_unassigned = (master_id_map == -1)
                
                # return_distances calculates distance in pixels; return_indices tracks coordinates of nearest line pixel
                distances, indices = distance_transform_edt(
                    mask_unassigned, 
                    return_distances=True, 
                    return_indices=True
                )

                # Map every single pixel in the image to the ID of the nearest road segment
                print(f"[{time.strftime('%H:%M:%S')}]   * Allocating spatial buffers dynamically...")
                nearest_segment_ids = master_id_map[indices[0], indices[1]]

                # Convert target 10.0 meters into pixels: (Resolution of DOP10 is 0.1m/pixel, so 10m = 100 pixels)
                pixel_cutoff = 10.0 / src_transform[0] 

                # Apply cutoff: pixels further than 10 meters are set back to unassigned (-1)
                master_id_map = np.where(distances <= pixel_cutoff, nearest_segment_ids, -1)
                print(f"[{time.strftime('%H:%M:%S')}] Allocation finished! Zero-overlap partition completed in {time.time() - t_vor:.2f}s.")

                # ----------------------------------------------------
                # PHASE C: TEXTURE & COLOR FEATURE EXTRACTION
                # ----------------------------------------------------
                t_feat = time.time()
                print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase C: Extracting Spectral & Textural Signatures ---")
                features_list = []
                segment_metadata = {}

                # Loop through and calculate properties for each segment in the allocated space
                for line, seg_id in segment_lines:
                    pixel_coords = (master_id_map == seg_id)
                    if not np.any(pixel_coords):
                        continue

                    # Retrieve isolated pixel values
                    pixels_red = rgb_full[:, :, 0][pixel_coords].astype(np.float32)
                    pixels_blue = rgb_full[:, :, 2][pixel_coords].astype(np.float32)
                    pixels_gray = gray_full[pixel_coords].astype(np.float32)

                    mean_red = np.mean(pixels_red)
                    mean_blue = np.mean(pixels_blue) if np.mean(pixels_blue) != 0 else 1.0
                    mean_gray = np.mean(pixels_gray)
                    std_gray = np.std(pixels_gray)

                    # Red/Blue Ratio highlights dirt/sand
                    soil_idx = mean_red / mean_blue
                    feat_vector = [mean_gray, soil_idx, std_gray]
                    features_list.append(feat_vector)

                    segment_metadata[seg_id] = {
                        "poly": line,
                        "coords": pixel_coords,
                        "cars_counted": 0,
                        "features": feat_vector,
                        "cluster_label": None
                    }

                print(f"[{time.strftime('%H:%M:%S')}] Completed feature mining for {len(features_list)} segments in {time.time() - t_feat:.2f}s.")

                # ----------------------------------------------------
                # PHASE D: UNSUPERVISED 2-CLASS CLUSTERING
                # ----------------------------------------------------
                t_cluster = time.time()
                print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase D: Running Unsupervised K-Means ---")
                X = np.array(features_list)
                
                # Cluster all 10m segments based purely on their appearance signatures
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
                cluster_assignments = kmeans.labels_

                for index, (seg_id, meta) in enumerate(segment_metadata.items()):
                    meta["cluster_label"] = cluster_assignments[index]

                print(f"[{time.strftime('%H:%M:%S')}] Class clustering finished in {time.time() - t_cluster:.2f}s.")

                # ----------------------------------------------------
                # PHASE E: TILED HIGH-RES VEHICLE DETECTION
                # ----------------------------------------------------
                t_yolo = time.time()
                print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase E: Performing Tiled YOLO Vehicle Search ---")
                
                # Count patches to provide a real-time progress indicator
                cols = int(np.ceil(orig_w / patch_size))
                rows = int(np.ceil(orig_h / patch_size))
                total_patches = cols * rows
                processed_patches = 0
                car_hits = 0

                for y in range(0, orig_h, patch_size):
                    for x in range(0, orig_w, patch_size):
                        processed_patches += 1
                        w = min(patch_size, orig_w - x)
                        h = min(patch_size, orig_h - y)
                        
                        # Optimization: check if there is any road present in this grid patch
                        local_ids = master_id_map[y:y+h, x:x+w]
                        if np.all(local_ids == -1):
                            continue

                        # Extract local image patch
                        bgr_chunk = bgr_full[y:y+h, x:x+w]
                        if h < patch_size or w < patch_size:
                            padded = np.zeros((patch_size, patch_size, 3), dtype=bgr_chunk.dtype)
                            padded[:h, :w, :] = bgr_chunk
                            bgr_chunk = padded

                        # Standard terminal print replacement line (\r) to show current progress status
                        percent_done = (processed_patches / total_patches) * 100
                        print(f"\r[{time.strftime('%H:%M:%S')}]   * Progress: {processed_patches}/{total_patches} patches scanned ({percent_done:.1f}%) | Cars: {car_hits}", end="", flush=True)

                        # Run YOLO inference
                        yolo_results = yolo_model.predict(source=bgr_chunk, device=device, verbose=False)[0]

                        for box_obj in yolo_results.boxes:
                            x1, y1, x2, y2 = map(int, box_obj.xyxy[0].cpu().numpy())
                            cls_id = int(box_obj.cls[0].cpu().numpy())
                            label = yolo_results.names[cls_id]

                            if label in ['car', 'van', 'truck', 'bus']:
                                global_cx = x + int((x1 + x2) / 2)
                                global_cy = y + int((y1 + y2) / 2)

                                if global_cx >= orig_w or global_cy >= orig_h:
                                    continue

                                # Get target ID directly from our master ID map
                                target_id = master_id_map[global_cy, global_cx]
                                if target_id in segment_metadata:
                                    segment_metadata[target_id]["cars_counted"] += 1
                                    car_hits += 1
                                    # Draw detected car bounding box onto main background matrix
                                    cv2.rectangle(bgr_full, (x + x1, y + y1), (x + x2, y + y2), (255, 0, 0), 2)

                print(f"\n[{time.strftime('%H:%M:%S')}] Completed YOLO scanning loop in {time.time() - t_yolo:.2f}s.")

                # ----------------------------------------------------
                # PHASE F: BEHAVIORAL HEURISTIC VOTING
                # ----------------------------------------------------
                t_vote = time.time()
                print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase F: Applying Unsupervised Vote Engine ---")
                
                cars_in_class_0 = sum(m["cars_counted"] for m in segment_metadata.values() if m["cluster_label"] == 0)
                cars_in_class_1 = sum(m["cars_counted"] for m in segment_metadata.values() if m["cluster_label"] == 1)

                print(f"[{time.strftime('%H:%M:%S')}]   * Total vehicles found on Class 0 terrain: {cars_in_class_0}")
                print(f"[{time.strftime('%H:%M:%S')}]   * Total vehicles found on Class 1 terrain: {cars_in_class_1}")

                if cars_in_class_0 >= cars_in_class_1:
                    active_road_cluster = 0
                    construction_cluster = 1
                else:
                    active_road_cluster = 1
                    construction_cluster = 0

                num_active = sum(1 for m in segment_metadata.values() if m["cluster_label"] == active_road_cluster)
                num_const = sum(1 for m in segment_metadata.values() if m["cluster_label"] == construction_cluster)

                print(f"[{time.strftime('%H:%M:%S')}]   * Result: Class {active_road_cluster} is ACTIVE ROAD ({num_active} segments)")
                print(f"[{time.strftime('%H:%M:%S')}]   * Result: Class {construction_cluster} is CONSTRUCTION ({num_const} segments)")

                # ----------------------------------------------------
                # PHASE G: RENDERING & EXPORT
                # ----------------------------------------------------
                t_render = time.time()
                print(f"\n[{time.strftime('%H:%M:%S')}] --- Phase G: Generating Visual Overlays & Export ---")
                
                vis_output = bgr_full.copy()
                red_overlay = np.zeros_like(vis_output)
                green_overlay = np.zeros_like(vis_output)

                for seg_id, meta in segment_metadata.items():
                    coords = meta["coords"]
                    label = meta["cluster_label"]

                    if label == construction_cluster:
                        red_overlay[coords] = [0, 0, 255] # Red BGR
                    else:
                        green_overlay[coords] = [0, 255, 0] # Green BGR

                # Apply transparent colored overlays
                cv2.addWeighted(red_overlay, 0.4, vis_output, 1.0, 0, vis_output)
                cv2.addWeighted(green_overlay, 0.2, vis_output, 1.0, 0, vis_output)

                # Transform BGRA canvas layout for georeferencing compatibility
                bgra_output = cv2.cvtColor(vis_output, cv2.COLOR_BGR2BGRA)
                bgra_output[:, :, 3] = 255
                rgba_output = cv2.cvtColor(bgra_output, cv2.COLOR_BGRA2RGBA)
                export_data = np.moveaxis(rgba_output, -1, 0)

                output_file_path = OUTPUT_DIR / f"{base_name}_adaptive_OBIA.tif"
                input_profile.update(
                    driver='GTiff',
                    count=4,
                    dtype=rasterio.ubyte,
                    nodata=0
                )

                # Save final georeferenced output file
                with rasterio.open(output_file_path, 'w', **input_profile) as dst:
                    dst.write(export_data)

                print(f"[{time.strftime('%H:%M:%S')}] Georeferenced map saved successfully: {output_file_path}")
                print(f"[{time.strftime('%H:%M:%S')}] Completed processing for {file_name} in {time.time() - tile_start_time:.2f}s.")

        except Exception as e:
            print(f"\n[{time.strftime('%H:%M:%S')}] [ERROR] Aborted processing loop for {file_name}: {e}")

    print("\n" + "="*80)
    print(f"[{time.strftime('%H:%M:%S')}] PIPELINE JOB COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()