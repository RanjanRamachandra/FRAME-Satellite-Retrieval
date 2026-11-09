import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from satpy import Scene
from pyresample import create_area_def

# Project paths
INPUT_DIR = "data/processed/hsd"
OUTPUT_DIR = "data/processed/satellite_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all decompressed HSD files
files = glob.glob(os.path.join(INPUT_DIR, "*.DAT"))

print("Files found:", len(files))

if len(files) != 30:
    raise ValueError("Expected 30 HSD files. Check the input folder.")

# Load satellite data
scn = Scene(
    filenames=files,
    reader="ahi_hsd"
)

bands = ["B07", "B08", "B13"]
scn.load(bands)

# Define Asia geographical region
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

# Reproject all bands onto the same grid
print("Reprojecting satellite images...")

resampled = scn.resample(
    asia,
    resampler="nearest"
)

arrays = {}

for band in bands:

    data = np.ma.asarray(resampled[band].values)

    image = np.asarray(
        np.ma.filled(data, np.nan),
        dtype=np.float32
    )

    arrays[band] = image

    valid = np.isfinite(image)

    print("\nBand:", band)
    print("Shape:", image.shape)
    print("Valid pixels:", round(valid.mean() * 100, 2), "%")

    if valid.any():
        print("Minimum:", np.nanmin(image))
        print("Maximum:", np.nanmax(image))
        print("Mean:", np.nanmean(image))

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap="gray")
    plt.title("Himawari-8 " + band + " Asia")
    plt.colorbar(label="Brightness temperature (K)")
    plt.savefig(
        os.path.join(OUTPUT_DIR, band + "_asia.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

# Save aligned three-channel data
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "himawari_asia_20220115_0400.npz"),
    B07=arrays["B07"],
    B08=arrays["B08"],
    B13=arrays["B13"]
)

print("\nPreprocessing completed!")
