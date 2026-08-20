# Autobahn Construction Detection Pipeline

An automated remote sensing and machine learning workflow to detect and map construction zones along the German Autobahn network using high-resolution Digital Orthophotos (DOP) and OpenStreetMap (OSM) data.

---

## Project & Authors

* **Authors:** Andreas Rademaker & Florian Thiemann 
* **Course:** Analysis of High Resultion Remote Sensing Imagery

## Local Setup
```
# Clone the repository
git clone https://github.com/nicevibesplus/autobahn-construction-detection
cd autobahn-construction-detection

# Create a local virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
1. Configure Profiles: Define your target tiles and OSM regions in ``profiles.py``
2. Adjust Pipeline Flags: Set ``BASE_NAME``, ``DEVICE``, ``RUN_TRAINING`` directly in ``pipeline.py``
3. Execute ``python pipeline.py``
4. All generated vectors, rasters, and final predictions are stored under ``data/output/<BASE_NAME>/``.