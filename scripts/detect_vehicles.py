import os
import time
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from tqdm import tqdm


def run_vehicle_detection(master_mosaic_path, vehicles_out_path, models_dir, device="cpu"):
    print(f"[{time.strftime('%H:%M:%S')}] Starting Standalone Vehicle Detection...")

    models_dir = Path(models_dir)
    os.environ["ULTRALYTICS_CONFIG_DIR"] = str(models_dir / "ultralytics_config")

    YOLO_PATCH_SIZE = 640
    TARGET_CLASSES = ['car', 'van', 'truck', 'bus']

    master_mosaic_path = Path(master_mosaic_path)
    vehicles_out_path = Path(vehicles_out_path)

    if not master_mosaic_path.exists():
        print(f"[ERROR] Master image not found: {master_mosaic_path}")
        return

    if vehicles_out_path.exists():
        vehicles_out_path.unlink()

    global_vehicles_list = []

    print(f"\n[{time.strftime('%H:%M:%S')}] --- Loading YOLOv8 Model ---")
    yolo_weights_path = hf_hub_download(repo_id="dronefreak/visdrone-yolov8s", filename="best.pt", cache_dir=models_dir / "yolo")
    yolo_model = YOLO(yolo_weights_path)

    print(f"\n[{time.strftime('%H:%M:%S')}] --- Running Inference on Master Image ---")

    with rasterio.open(master_mosaic_path) as src:
        orig_w = src.width
        orig_h = src.height
        src_crs = src.crs
        src_transform = src.transform

        windows = []
        for y_offset in range(0, orig_h, YOLO_PATCH_SIZE):
            for x_offset in range(0, orig_w, YOLO_PATCH_SIZE):
                w = min(YOLO_PATCH_SIZE, orig_w - x_offset)
                h = min(YOLO_PATCH_SIZE, orig_h - y_offset)
                windows.append((x_offset, y_offset, w, h))

        for x_offset, y_offset, w, h in tqdm(windows, desc="Running YOLO inference on patches"):
            window = Window(x_offset, y_offset, w, h)

            rgb_chunk = src.read([1, 2, 3], window=window)
            if not np.any(rgb_chunk):
                continue

            rgb_chunk = np.moveaxis(rgb_chunk, 0, -1)
            bgr_chunk = cv2.cvtColor(rgb_chunk, cv2.COLOR_RGB2BGR)

            if h < YOLO_PATCH_SIZE or w < YOLO_PATCH_SIZE:
                padded = np.zeros((YOLO_PATCH_SIZE, YOLO_PATCH_SIZE, 3), dtype=bgr_chunk.dtype)
                padded[:h, :w, :] = bgr_chunk
                bgr_chunk = padded

            yolo_results = yolo_model.predict(source=bgr_chunk, device=device, verbose=False)[0]

            for box_obj in yolo_results.boxes:
                class_idx = int(box_obj.cls[0].cpu().numpy())
                class_name = yolo_results.names[class_idx]

                if class_name in TARGET_CLASSES:
                    x1, y1, x2, y2 = map(int, box_obj.xyxy[0].cpu().numpy())
                    if x1 >= w or y1 >= h:
                        continue
                    x2 = min(x2, w)
                    y2 = min(y2, h)

                    gx1, gy1 = x_offset + x1, y_offset + y1
                    gx2, gy2 = x_offset + x2, y_offset + y2
                    mx1, my1 = src_transform * (gx1, gy1)
                    mx2, my2 = src_transform * (gx2, gy2)

                    vehicle_geom = box(mx1, my2, mx2, my1)
                    global_vehicles_list.append({
                        "geometry": vehicle_geom,
                        "type": class_name,
                        "confidence": float(box_obj.conf[0].cpu().numpy())
                    })

    print(f"\n[{time.strftime('%H:%M:%S')}] --- Exporting Results ---")
    if not global_vehicles_list:
        print("[WARNING] No vehicles detected in the entire image.")
        return

    gdf = gpd.GeoDataFrame(global_vehicles_list, crs=src_crs)
    gdf.to_file(str(vehicles_out_path), layer="detected_vehicles", driver="GPKG")
    print(f"[{time.strftime('%H:%M:%S')}] PIPELINE COMPLETE! File saved to: {vehicles_out_path.name}")
