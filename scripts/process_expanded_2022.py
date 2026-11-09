import os
import glob
import csv
import numpy as np
from satpy import Scene
from pyresample import create_area_def

INPUT_DIR = "data/expanded_2022/processed"
OUTPUT_DIR = "data/expanded_2022/npz"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BANDS = ["B07", "B08", "B13"]

asia = create_area_def(
    "asia",
    {
        "proj": "longlat",
        "datum": "WGS84",
        "ellps": "WGS84"
    },
    area_extent=(90, -10, 160, 60),
    shape=(1000, 1000)
)

metadata = []
errors = []

folders = sorted(glob.glob(os.path.join(INPUT_DIR, "*")))

for folder in folders:

    if not os.path.isdir(folder):
        continue

    observation = os.path.basename(folder)

    files = glob.glob(os.path.join(folder, "*.DAT"))

    if len(files) == 0:
        print("SKIP:", observation, "- empty folder")
        continue

    if len(files) != 30:
        errors.append({
            "observation": observation,
            "error": f"Expected 30 DAT files, found {len(files)}"
        })
        print("ERROR:", observation, "- incorrect file count")
        continue

    output_file = os.path.join(
        OUTPUT_DIR,
        f"himawari_asia_{observation}.npz"
    )

    if os.path.exists(output_file):
        print("SKIP:", observation, "- NPZ already exists")
        continue

    print("\nProcessing:", observation)

    try:
        scn = Scene(
            filenames=files,
            reader="ahi_hsd"
        )

        scn.load(BANDS)

        resampled = scn.resample(
            asia,
            resampler="nearest"
        )

        arrays = {}

        for band in BANDS:

            data = np.ma.asarray(
                resampled[band].values
            )

            image = np.asarray(
                np.ma.filled(data, np.nan),
                dtype=np.float32
            )

            if image.shape != (1000, 1000):
                raise ValueError(
                    f"{band} unexpected shape: {image.shape}"
                )

            if not np.isfinite(image).any():
                raise ValueError(
                    f"{band} contains no valid pixels"
                )

            arrays[band] = image

            valid_fraction = np.isfinite(image).mean()

            print(
                band,
                "Shape:", image.shape,
                "Valid:", round(valid_fraction * 100, 2), "%"
            )

        np.savez_compressed(
            output_file,
            B07=arrays["B07"],
            B08=arrays["B08"],
            B13=arrays["B13"]
        )

        date = observation[:8]
        time_utc = observation[9:]

        metadata.append({
            "observation": observation,
            "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
            "time_utc": time_utc,
            "npz_path": output_file,
            "height": 1000,
            "width": 1000,
            "channels": 3
        })

        print("Saved:", output_file)

    except Exception as error:

        errors.append({
            "observation": observation,
            "error": str(error)
        })

        print("ERROR processing", observation, ":", error)


metadata_path = os.path.join(
    OUTPUT_DIR,
    "expanded_metadata.csv"
)

with open(metadata_path, "w", newline="") as csvfile:

    fields = [
        "observation",
        "date",
        "time_utc",
        "npz_path",
        "height",
        "width",
        "channels"
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(metadata)


error_path = os.path.join(
    OUTPUT_DIR,
    "processing_errors.csv"
)

with open(error_path, "w", newline="") as csvfile:

    writer = csv.DictWriter(
        csvfile,
        fieldnames=["observation", "error"]
    )

    writer.writeheader()
    writer.writerows(errors)


print("\nProcessing completed.")
print("Successful new observations:", len(metadata))
print("Processing errors:", len(errors))
print("Metadata:", metadata_path)
print("Error report:", error_path)
