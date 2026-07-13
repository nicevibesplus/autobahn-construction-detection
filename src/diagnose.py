import os
import glob
import cv2
import numpy as np
import rasterio
from pathlib import Path
from huggingface_hub import hf_hub_download
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"
MODELS_DIR = BASE_DIR / "models"

# 1. Load Model
unet_file_path = hf_hub_download(
    repo_id="spectrewolf8/aerial-image-road-segmentation-with-U-NET-xp",
    filename="aerial-image-road-segmentation-xp.keras",
    cache_dir=MODELS_DIR / "keras_unet"
)
unet_model = tf.keras.models.load_model(unet_file_path, compile=False)

# 2. Grab first JP2
jp2_files = glob.glob(os.path.join(INPUT_DIR, "*.jp2"))
if not jp2_files:
    print("No JP2 files found.")
    exit()

with rasterio.open(jp2_files[0]) as src:
    # Read a single 2560x2560 window from the top corner
    window = rasterio.windows.Window(0, 0, 2560, 2560)
    rgb_data = src.read([1, 2, 3], window=window)
    rgb_data = np.moveaxis(rgb_data, 0, -1)

# Downsample to 256
img_downsampled = cv2.resize(rgb_data, (256, 256), interpolation=cv2.INTER_AREA)

# Run test prediction on standard 0-1 normalization
img_norm_1 = img_downsampled.astype(np.float32) / 255.0
pred_1 = unet_model.predict(np.expand_dims(img_norm_1, axis=0), verbose=0)

print("\n=== DIAGNOSTICS ===")
print(f"Raw Output Shape from Model: {pred_1.shape}")
print(f"Max confidence value: {pred_1.max():.4f}")
print(f"Min confidence value: {pred_1.min():.4f}")
print(f"Mean confidence value: {pred_1.mean():.4f}")