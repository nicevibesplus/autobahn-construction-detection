# ====================================================
# FILE: predict_random_forest.py
# ====================================================
import joblib
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

def run_random_forest_prediction(segments_full_path, model_path, output_path):
    FEATURES = [
        "surface_class",
        "ndvi_mean",
        "texture_mean",
        "marking_white_sum",
        "marking_orange_sum",
        "vehicle_fraction"
    ]

    CLASS_NAMES = {
        0: "other",
        1: "road",
        2: "construction"
    }

    print("Loading trained Random Forest...")
    clf = joblib.load(model_path)
    print("Loaded trained Random Forest")

    print("Loading full segments file...")
    full = gpd.read_file(segments_full_path)
    X = full[FEATURES].fillna(0)

    print("Running predictions...")
    full["predicted_class"] = clf.predict(X)
    full["predicted_name"] = full["predicted_class"].map(CLASS_NAMES)

    # Added tqdm progress bar loop to wrap the file saving operation
    for _ in tqdm(range(1), desc=f"Saving predictions ({Path(output_path).name})"):
        full.to_file(output_path, driver="GPKG")

    print(f"Predicted {len(full)} segments and saved to {output_path}")