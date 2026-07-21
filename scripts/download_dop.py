import requests
from pathlib import Path
from tqdm import tqdm


def download_dop_tiles(tiles, output_dir):
    print(f"Starting DOP download ({len(tiles)} tiles)...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    years_to_try = [2026, 2025, 2024, 2023]  # newest year first
    base_url_template = "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/dop10rgbi_32_{tile}_1_nw_{year}.jp2"

    for tile in tqdm(tiles, desc="Downloading DOP tiles"):
        downloaded = False

        for year in years_to_try:
            url = base_url_template.format(tile=tile, year=year)
            filename = url.split("/")[-1]
            destination = output_dir / filename

            if destination.exists():
                print(f"  [SKIP] File already exists: {filename}")
                downloaded = True
                break

            print(f"  Checking/Downloading {filename} for year {year}...")
            try:
                response = requests.get(url, stream=True)

                if response.status_code == 200:
                    total_size = int(response.headers.get("content-length", 0))
                    block_size = 1024

                    with open(destination, "wb") as f, tqdm(
                        total=total_size, unit="B", unit_scale=True, desc=filename, leave=False
                    ) as bar:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))

                    print(f"  [SUCCESS] Downloaded {filename} (Year: {year})")
                    downloaded = True
                    break
                else:
                    print(f"  [INFO] Year {year} not available for tile {tile} (Status code: {response.status_code})")
            except requests.exceptions.RequestException as e:
                print(f"  [WARNING] Connection error for {filename} (Year {year}): {e}")

        if not downloaded:
            print(f"  [ERROR] Could not find any valid image file for tile {tile} across years {years_to_try}")

    print("Download process complete!")