# ====================================================
# FILE: train_random_forest.py
# ====================================================
import joblib
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def run_random_forest_training(labeled_gpkg, model_output_path):
    FEATURES = [
        "surface_class",
        "ndvi_mean",
        "texture_mean",
        "marking_white_sum",
        "marking_orange_sum",
        "vehicle_fraction"
    ]

    # Map class labels correctly: 1=road, 2=construction, 3=other
    # Order matches indices/labels if encoded, or we supply target_names ordered by class labels (1, 2, 3)
    CLASS_NAMES = ["road", "construction", "other"]

    # ----------------------------
    # Load labelled training data
    # ----------------------------
    labeled = gpd.read_file(labeled_gpkg)

    print(f"Training on {len(labeled)} labelled segments")

    X = labeled[FEATURES].fillna(0)
    y = labeled["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # ----------------------------
    # Train
    # ----------------------------
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    clf.fit(X_train, y_train)

    # ----------------------------
    # Evaluate
    # ----------------------------
    y_pred = clf.predict(X_test)

    print("\nClassification report")
    # labels=[1, 2, 3] ensures proper alignment with target_names since classes are numeric integers
    print(classification_report(y_test, y_pred, labels=[1, 2, 3], target_names=CLASS_NAMES))

    print("\nConfusion matrix")
    print(confusion_matrix(y_test, y_pred, labels=[1, 2, 3]))

    print("\nFeature importance")
    for name, imp in sorted(zip(FEATURES, clf.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"{name:25s} {imp:.3f}")

    # ----------------------------
    # Save trained model
    # ----------------------------
    joblib.dump(clf, model_output_path)

    print(f"\nModel saved as {model_output_path}")