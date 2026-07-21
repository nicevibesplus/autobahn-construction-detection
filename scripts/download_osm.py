import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


def download_osm_gpkg(regbez, output_dir):
    print(f"Starting Geofabrik download for region '{regbez}'...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://download.geofabrik.de/europe/germany/nordrhein-westfalen/{regbez}-latest-free.gpkg.zip"
    zip_filename = f"{regbez}-latest-free.gpkg.zip"
    zip_path = output_dir / zip_filename

    if (output_dir / f"{regbez}.gpkg").exists():
        print(f"  [SKIP] Geofabrik .gpkg already exists for {regbez}. Skipping download.")
        return

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(zip_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=zip_filename
        ) as bar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"Extracting .gpkg file from {zip_filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith('.gpkg'):
                    extracted_path = output_dir / Path(file_info.filename).name
                    with zip_ref.open(file_info) as source, open(extracted_path, "wb") as target:
                        target.write(source.read())
                    print(f"  [SUCCESS] Extracted: {extracted_path.name}")

        zip_path.unlink()
        print("Geofabrik download and extraction complete!")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to download Geofabrik data for {regbez}: {e}")
    except zipfile.BadZipFile:
        print("[ERROR] Downloaded file is not a valid zip archive.")