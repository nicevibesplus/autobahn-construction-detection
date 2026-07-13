import os
import glob
import cv2
import torch
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

# ----------------------------------------------------
# 0. PATHS & CACHE MANAGEMENT
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
PATCHES_DIR = OUTPUT_DIR / "patches"
MODELS_DIR = BASE_DIR / "models"
ROADS_GPKG = BASE_DIR / "data" / "muenster-regbez.gpkg"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PATCHES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["ULTRALYTICS_CONFIG_DIR"] = str(MODELS_DIR / "ultralytics_config")

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

def main():
    device = "cpu"  
    print(f"Using device for YOLO: {device}")

    # 1. LOAD GEOPACKAGE ROAD LAYER
    if not ROADS_GPKG.exists():
        print(f"[ERROR] Missing road GeoPackage at: {ROADS_GPKG}.")
        return
        
    print("Loading OpenStreetMap roads from GeoPackage...")
    try:
        gdf_all_roads = gpd.read_file(str(ROADS_GPKG), layer='gis_osm_roads_free')
        
        if 'fclass' in gdf_all_roads.columns:
            # STRICT FILTER: Keep ONLY the German Autobahn network and ramps
            valid_highways = ['motorway', 'motorway_link']
            gdf_all_roads = gdf_all_roads[gdf_all_roads['fclass'].isin(valid_highways)]
            print(f" -> Successfully filtered {len(gdf_all_roads)} Autobahn segments.")
        else:
            print(" -> [WARNING] 'fclass' column not found in layer. Using all vector lines.")
            
    except Exception as e:
        print(f"[ERROR] Could not read layer 'gis_osm_roads_free' from GeoPackage: {e}")
        return
    
    # 2. LOAD CAR DETECTION MODEL
    print("--- Initializing Vehicle Model ---")
    yolo_weights_path = hf_hub_download(
        repo_id="dronefreak/visdrone-yolov8s", 
        filename="best.pt", 
        cache_dir=MODELS_DIR / "yolo"
    )
    yolo_model = YOLO(yolo_weights_path)
    print("Model loaded successfully.\n")

    # 3. DISCOVER JP2 IMAGES
    jp2_files = glob.glob(os.path.join(INPUT_DIR, "*.jp2"))
    if not jp2_files:
        print(f"No .jp2 files found in {INPUT_DIR}.")
        return

    print(f"Found {len(jp2_files)} files to process.")

    # 4. PIPELINE LOOP USING VECTOR SELECTION
    tile_size = 2560  

    for idx, jp2_path in enumerate(jp2_files, start=1):
        file_name = os.path.basename(jp2_path)
        base_name = os.path.splitext(file_name)[0]
        print(f"\n=========================================")
        print(f"[{idx}/{len(jp2_files)}] Processing Mosaic Tile: {file_name}")
        print(f"=========================================")

        try:
            with rasterio.open(jp2_path) as src:
                orig_w, orig_h = src.width, src.height
                src_crs = src.crs
                src_transform = src.transform
                print(f" -> Resolution: {orig_w}x{orig_h} pixels, CRS: {src_crs}")
                
                # Get image boundary limits
                img_box = box(*src.bounds)
                
                # Reproject road vectors to match your EPSG:25832 data projection
                gdf_local_roads = gdf_all_roads.to_crs(src_crs)
                
                # Filter down only to vectors cutting across this map sheet
                gdf_local_roads = gdf_local_roads[gdf_local_roads.geometry.intersects(img_box)]
                
                if gdf_local_roads.empty:
                    print(" -> No matching vector highways cross this image space. Skipping tile completely.")
                    continue
                
                # Buffer the lines by 20 meters to lock in the lane corridors
                gdf_buffered = gdf_local_roads.copy()
                gdf_buffered.geometry = gdf_local_roads.geometry.buffer(20.0)
                
                # Burn vector layer directly into high-res mask arrays matching your image canvas
                print(" -> Rasterizing vector layers into high-res pixel canvas masks...")
                shapes = [(geom, 255) for geom in gdf_buffered.geometry]
                master_road_mask = features.rasterize(
                    shapes=shapes,
                    out_shape=(orig_h, orig_w),
                    transform=src_transform,
                    fill=0,
                    dtype=np.uint8
                )

                # Blank 10000x10000 target canvas deployment (BGRA with completely transparent alpha)
                canvas = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)

                cols = int(np.ceil(orig_w / tile_size))
                rows = int(np.ceil(orig_h / tile_size))
                total_patches = cols * rows
                current_patch_idx = 0

                for y in range(0, orig_h, tile_size):
                    for x in range(0, orig_w, tile_size):
                        current_patch_idx += 1
                        w = min(tile_size, orig_w - x)
                        h = min(tile_size, orig_h - y)
                        window = rasterio.windows.Window(x, y, w, h)

                        print(f"\r   -> [{current_patch_idx}/{total_patches}] Scanning grid space window (x: {x}, y: {y})...", end="", flush=True)

                        # Skip patch ONLY if no vector roads cross this coordinate window
                        local_mask_chunk = master_road_mask[y:y+h, x:x+w]
                        if np.sum(local_mask_chunk) == 0:
                            continue

                        # Extract high-res image data
                        rgb_data = src.read([1, 2, 3], window=window)
                        rgb_data = np.moveaxis(rgb_data, 0, -1)

                        if h < tile_size or w < tile_size:
                            padded = np.zeros((tile_size, tile_size, 3), dtype=rgb_data.dtype)
                            padded[:h, :w, :] = rgb_data
                            rgb_data = padded
                            
                            padded_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
                            padded_mask[:h, :w] = local_mask_chunk
                            local_mask_chunk = padded_mask

                        print(f"\n      * HIT! Vector corridor crosses patch ({x}, {y}). Processing...")

                        # Send to YOLO to check for vehicles in this road patch
                        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
                        yolo_results = yolo_model.predict(source=bgr_data, device=device, verbose=False)[0]

                        vis_image = bgr_data.copy()
                        green_overlay = np.zeros_like(vis_image)
                        green_overlay[local_mask_chunk == 255] = [0, 255, 0] 
                        cv2.addWeighted(green_overlay, 0.25, vis_image, 1.0, 0, vis_image)

                        # Trace vehicles that reside strictly within the road buffer zone
                        for box_obj in yolo_results.boxes:
                            x1, y1, x2, y2 = map(int, box_obj.xyxy[0].cpu().numpy())
                            conf = float(box_obj.conf[0].cpu().numpy())
                            cls_id = int(box_obj.cls[0].cpu().numpy())
                            label = yolo_results.names[cls_id]

                            if label in ['car', 'van', 'truck', 'bus']:
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                
                                # Vector boundary check
                                if local_mask_chunk[center_y, center_x] == 255:
                                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 3) 
                                    cv2.putText(vis_image, f"{label} {conf:.2f}", (x1, max(y1 - 10, 20)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2) 

                        # Export patch to local temp file (no geo metadata needed for raw patch clips)
                        patch_cropped = vis_image[:h, :w]
                        patch_file_name = f"{base_name}_x{x}_y{y}.png"
                        cv2.imwrite(str(PATCHES_DIR / patch_file_name), patch_cropped)

                        # Reconstruct patch back onto the master canvas coordinates
                        bgra_patch = cv2.cvtColor(vis_image, cv2.COLOR_BGR2BGRA)
                        bgra_patch[:, :, 3] = 255
                        canvas[y:y+h, x:x+w] = bgra_patch[:h, :w]

                print("\n   -> Grid looping finished for this mosaic map sheet.")
                
                # ----------------------------------------------------
                # 4. EXPORT GEOTIFF COMPILATION
                # ----------------------------------------------------
                output_file_path = OUTPUT_DIR / f"{base_name}_stitched.tif"
                print(f" -> Compiling full layout... Saving georeferenced GeoTIFF...")

                # A. Convert standard OpenCV BGRA canvas to standard GIS RGBA representation
                rgba_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)

                # B. Move channel axis to the front as required by Rasterio (H, W, 4) -> (4, H, W)
                export_data = np.moveaxis(rgba_canvas, -1, 0)

                # C. Write GeoTIFF using the original spatial coordinates and CRS definitions
                with rasterio.open(
                    output_file_path,
                    'w',
                    driver='GTiff',
                    height=orig_h,
                    width=orig_w,
                    count=4,
                    dtype=rasterio.ubyte,
                    crs=src_crs,
                    transform=src_transform,
                    nodata=0  # Signals QGIS that 0-value pixels are completely transparent alpha
                ) as dst:
                    dst.write(export_data)

                print(f" -> Output exported successfully with EPSG:25832 tags: {output_file_path}")

        except Exception as e:
            print(f"\n [ERROR] Aborted file loop for {file_name}: {e}")

    print("\n--- Pipeline Job Complete ---")

if __name__ == "__main__":
    main()