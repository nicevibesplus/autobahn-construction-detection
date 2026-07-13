import rasterio

file_path = "data/input/dop10rgbi_32_401_5746_1_nw_2024.jp2"

with rasterio.open(file_path) as src:
    print("--- Channel Interpretation ---")
    for i in range(1, src.count + 1):
        # Accessing the tuple using square brackets
        color_interp = src.colorinterp[i - 1]
        print(f"Band {i}: {color_interp.name}")