# ====================================================
# FILE: pipeline.py
# ====================================================
import time
from pathlib import Path

# Import steps from your separate script files
from scripts.create_mosaic import run_mosaic_extraction
from scripts.ndvi_calculation import run_ndvi_calculation
from scripts.detect_lane_markings import run_lane_marking_detection
from scripts.detect_vehicles import run_vehicle_detection
from scripts.surface_type_classification import run_surface_classification
from scripts.texture_analysis import run_texture_analysis
from scripts.fuse_segments import run_fuse_segments
from scripts.train_random_forest import run_random_forest_training
from scripts.predict_random_forest import run_random_forest_prediction
from scripts.quickshift_segmentation import run_quickshift_segmentation
from scripts.download_dop import download_dop_tiles
from scripts.download_osm import download_osm_gpkg

# ----------------------------------------------------
# CONFIGURATION & BASE PATHS
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
MODELS_DIR = BASE_DIR / "models"

# Dynamic Base Name Configuration
BASE_NAME = "example_bottrop"

# External static inputs
ROADS_GPKG = BASE_DIR / "data" / "duesseldorf-regbez.gpkg"
LABELED_GPKG = BASE_DIR / "segments_to_label.gpkg"  # Required if running training

# Pipeline Parameters
GRID_SPLIT = 10
ROAD_SEARCH_BUFFER_M = 30.0
VEGETATION_THRESHOLD = 0.3
DEVICE = "cpu"  # Change to "cuda" if using an NVIDIA GPU
RUN_TRAINING = False  # Set to True if you want to retrain the model in the pipeline


def main():
    start_time = time.time()
    print(
        f"[{time.strftime('%H:%M:%S')}] Starting Master Remote Sensing Pipeline for '{BASE_NAME}'..."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Automatically generated file paths based on BASE_NAME
    master_mosaic = OUTPUT_DIR / f"{BASE_NAME}.tif"
    ndvi_path = OUTPUT_DIR / f"{BASE_NAME}_ndvi.tif"
    white_line_path = OUTPUT_DIR / f"{BASE_NAME}_lane_markings_white.tif"
    orange_line_path = OUTPUT_DIR / f"{BASE_NAME}_lane_markings_orange.tif"
    white_ref_path = OUTPUT_DIR / f"{BASE_NAME}_lane_markings_white_refined.tif"
    orange_ref_path = OUTPUT_DIR / f"{BASE_NAME}_lane_markings_orange_refined.tif"
    vehicles_gpkg = OUTPUT_DIR / f"{BASE_NAME}_detected_vehicles.gpkg"
    segmentation_gpkg = OUTPUT_DIR / f"{BASE_NAME}_segments.gpkg"
    segments_surface = OUTPUT_DIR / f"{BASE_NAME}_segments_with_surface.gpkg"
    segments_texture = OUTPUT_DIR / f"{BASE_NAME}_segments_with_texture.gpkg"
    segments_full = OUTPUT_DIR / f"{BASE_NAME}_segments_full.gpkg"
    rf_model_path = OUTPUT_DIR / f"{BASE_NAME}_construction_rf.pkl"
    classified_gpkg = OUTPUT_DIR / f"{BASE_NAME}_segments_classified.gpkg"

    download_dop_tiles(
        preset_name=BASE_NAME,
        output_dir=INPUT_DIR
    )
    
    download_osm_gpkg(
        profile_name=BASE_NAME,
        output_dir=DATA_DIR
    )
    
    
    # ----------------------------------------------------
    # STEP 1: Master Mosaic Extraction
    # ----------------------------------------------------
    run_mosaic_extraction(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        roads_gpkg=ROADS_GPKG,
        grid_split=GRID_SPLIT,
        road_search_buffer_m=ROAD_SEARCH_BUFFER_M,
        master_mosaic_path=master_mosaic,
    )

    # ----------------------------------------------------
    # STEP 2: NDVI Calculation
    # ----------------------------------------------------
    run_ndvi_calculation(
        raster_path=master_mosaic, ndvi_path=ndvi_path, red_band=1, nir_band=4
    )

    # ----------------------------------------------------
    # STEP 3: Lane Marking Detection & Refinement
    # ----------------------------------------------------
    run_lane_marking_detection(
        raster_path=master_mosaic,
        white_out=white_line_path,
        orange_out=orange_line_path,
        white_ref_out=white_ref_path,
        orange_ref_out=orange_ref_path,
    )

    # ----------------------------------------------------
    # STEP 4: Vehicle Detection
    # ----------------------------------------------------
    run_vehicle_detection(
        master_mosaic_path=master_mosaic,
        vehicles_out_path=vehicles_gpkg,
        models_dir=MODELS_DIR,
        device=DEVICE,
    )

    run_quickshift_segmentation(
        input_raster=master_mosaic,
        output_gpkg=segmentation_gpkg,
    )

    # ----------------------------------------------------
    # STEP 5: Surface Classification & Feature Extraction
    # ----------------------------------------------------
    run_surface_classification(
        raster_path=master_mosaic,
        ndvi_path=ndvi_path,
        segments_path=segmentation_gpkg,
        output_path=segments_surface,
        vegetation_threshold=VEGETATION_THRESHOLD,
    )

    run_texture_analysis(
        raster_path=master_mosaic,
        segments_path=segmentation_gpkg,
        output_path=segments_texture,
    )

    run_fuse_segments(
        segments_surface_path=segments_surface,
        segments_texture_path=segments_texture,
        white_refined_path=white_ref_path,
        orange_refined_path=orange_ref_path,
        vehicles_gpkg_path=vehicles_gpkg,
        output_path=segments_full,
    )

    # ----------------------------------------------------
    # STEP 6: Random Forest Training (Optional) & Prediction
    # ----------------------------------------------------
    """
    if RUN_TRAINING:
        run_random_forest_training(
            labeled_gpkg=LABELED_GPKG,
            model_output_path=rf_model_path
        )

    run_random_forest_prediction(
        segments_full_path=segments_full,
        model_path=rf_model_path,
        output_path=classified_gpkg
    )
    """

    elapsed = time.time() - start_time
    print(
        f"\n[{time.strftime('%H:%M:%S')}] PIPELINE COMPLETE SUCCESSFULLY in {elapsed:.2f} seconds!"
    )


if __name__ == "__main__":
    main()
