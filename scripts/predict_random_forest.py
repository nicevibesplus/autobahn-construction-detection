# ====================================================
# FILE: predict_random_forest.py
# ====================================================
import joblib
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

def run_random_forest_prediction(segments_full_path, model_path, output_path, confidence_threshold=0.80):
    FEATURES = [
        "surface_class",
        "ndvi_mean",
        "texture_mean",
        "marking_white_sum",
        "marking_orange_sum",
        "vehicle_fraction"
    ]

    # Updated class mapping to match your training labels: 1=road, 2=construction, 3=other
    CLASS_NAMES = {
        1: "road",
        2: "construction",
        3: "other"
    }

    print("Loading trained Random Forest...")
    clf = joblib.load(model_path)
    print("Loaded trained Random Forest")

    print("Loading full segments file...")
    full = gpd.read_file(segments_full_path)
    X = full[FEATURES].fillna(0)

    print("Running predictions and probability calculations...")
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)
    
    # Get the maximum probability for each prediction (confidence score)
    max_probs = probabilities.max(axis=1)

    # Apply confidence thresholding to prevent false construction classification
    # If the model predicts construction (2) but confidence is below the threshold, fall back to road (1) or other (3)
    adjusted_predictions = []
    for pred, prob in zip(predictions, max_probs):
        if pred == 2 and prob < confidence_threshold:
            # Fall back to road or other based on secondary probability or safe default
            adjusted_predictions.append(1)  # Fall back to road (or 3 for 'other')
        else:
            adjusted_predictions.append(pred)

    full["predicted_class"] = adjusted_predictions
    full["prediction_confidence"] = max_probs
    full["predicted_name"] = full["predicted_class"].map(CLASS_NAMES)

    for _ in tqdm(range(1), desc=f"Saving predictions ({Path(output_path).name})"):
        full.to_file(output_path, driver="GPKG")

    print(f"Predicted {len(full)} segments and saved to {output_path}")