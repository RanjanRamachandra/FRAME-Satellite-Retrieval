
import os
import glob
import csv
import re
import numpy as np

from satpy import Scene
from pyresample import create_area_def

INPUT_DIR = "data/processed/multiday"
OUTPUT_DIR = "data/processed/satellite_images/7days"

BANDS = ["B07", "B08", "B13"]
EXPECTED_SHAPE = (1000, 1000)

os.makedirs(OUTPUT_DIR, exist_ok=True)

asia = create_area_def(
    "asia",
    {"proj": "longlat", "datum": "WGS84"},
    area_extent=(90, -10, 160, 60),
    shape=EXPECTED_SHAPE
)

folders = sorted(
    p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
    if os.path.isdir(p)
)

metadata = []
errors = []

print("Observation folders found:", len(folders))

for index, folder in enumerate(folders, 1):

    folder_name = os.path.basename(folder)

    # Support both folder naming patterns
    if re.fullmatch(r"\d{4}", folder_name):
        observation = "20220115_" + folder_name
    else:
        observation = folder_name

    print(f"\n[{index}/{len(folders)}] Processing {observation}")

    files = sorted(glob.glob(os.path.join(folder, "*.DAT")))

    if len(files) != 30:
        message = f"Expected 30 DAT files, found {len(files)}"
        print("SKIPPED:", message)
        errors.append((observation, message))
        continue

    try:
        for band in BANDS:
            count = sum(
                f"_{band}_" in os.path.basename(path)
                for path in files
            )
            if count != 10:
                raise ValueError(
                    f"{band}: expected 10 segments, found {count}"
                )

        scene = Scene(
            filenames=files,
            reader="ahi_hsd"
        )

        scene.load(BANDS)

        resampled = scene.resample(
            asia,
            resampler="nearest"
        )

        arrays = {}

        for band in BANDS:
            arr = np.ma.asarray(resampled[band].values)
            arr = arr.filled(np.nan).astype(np.float32)

            if arr.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"{band} incorrect shape: {arr.shape}"
                )

            if not np.isfinite(arr).any():
                raise ValueError(f"{band} has no valid pixels")

            arrays[band] = arr

        output_file = os.path.join(
            OUTPUT_DIR,
            f"himawari_asia_{observation}.npz"
        )

        np.savez_compressed(output_file, **arrays)

        metadata.append({
            "observation": observation,
            "date": observation[:8],
            "time_utc": observation[9:],
            "npz_path": output_file,
            "height": 1000,
            "width": 1000,
            "channels": 3
        })

        print("SUCCESS:", output_file)

    except Exception as e:
        print("ERROR:", str(e))
        errors.append((observation, str(e)))

# Save metadata
if metadata:
    with open(
        os.path.join(OUTPUT_DIR, "metadata.csv"),
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

# Save errors
with open(
    os.path.join(OUTPUT_DIR, "errors.csv"),
    "w",
    newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow(["observation", "error"])
    writer.writerows(errors)

print("\n========== SUMMARY ==========")
print("Folders:", len(folders))
print("Processed:", len(metadata))
print("Failed:", len(errors))
print("Output:", OUTPUT_DIR)

