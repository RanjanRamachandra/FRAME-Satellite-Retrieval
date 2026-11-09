import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

from satpy import Scene
from pyresample import create_area_def

INPUT_DIR = "data/processed/multiday"
OUTPUT_DIR = "data/processed/satellite_images/multiday"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMES = [
    "0000", "0300", "0600", "0900",
    "1200", "1500", "1800", "2100"
]

BANDS = ["B07", "B08", "B13"]

# Asia geographic area
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

for time in TIMES:

    print("\nProcessing observation:", time)

    folder = os.path.join(INPUT_DIR, time)
    files = glob.glob(os.path.join(folder, "*.DAT"))

    print("Files found:", len(files))

    if len(files) != 30:
        print("Skipping", time, "- expected 30 files")
        continue

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
                    f"{band} has unexpected shape {image.shape}"
                )

            if not np.isfinite(image).any():
                raise ValueError(
                    f"{band} contains no valid pixels"
                )

            arrays[band] = image

            valid_fraction = float(
                np.isfinite(image).mean()
            )

            print(
                band,
                "Shape:", image.shape,
                "Valid:", round(valid_fraction * 100, 2), "%"
            )

        # Save three-channel numerical data
        output_file = os.path.join(
            OUTPUT_DIR,
            f"himawari_asia_20220115_{time}.npz"
        )

        np.savez_compressed(
            output_file,
            B07=arrays["B07"],
            B08=arrays["B08"],
            B13=arrays["B13"]
        )

        # Save preview for the first observation only
        if time == "0000":
            for band in BANDS:
                plt.figure(figsize=(8, 6))
                plt.imshow(arrays[band], cmap="gray")
                plt.title(f"Himawari-8 {band} - Asia - {time}")
                plt.colorbar()
                plt.savefig(
                    os.path.join(
                        OUTPUT_DIR,
                        f"{band}_preview.png"
                    ),
                    dpi=120,
                    bbox_inches="tight"
                )
                plt.close()

        metadata.append({
            "date": "2022-01-15",
            "time_utc": time,
            "npz_path": output_file,
            "height": 1000,
            "width": 1000,
            "channels": 3
        })

        print("Saved:", output_file)

    except Exception as error:
        print("ERROR processing", time, ":", error)

# Save metadata CSV
csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")

with open(csv_path, "w", newline="") as csvfile:

    writer = csv.DictWriter(
        csvfile,
        fieldnames=[
            "date", "time_utc", "npz_path",
            "height", "width", "channels"
        ]
    )

    writer.writeheader()
    writer.writerows(metadata)

print("\nProcessing completed.")
print("Successful observations:", len(metadata))
print("Metadata:", csv_path)
