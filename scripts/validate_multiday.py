
import glob
import os
import numpy as np

DATA_DIR = "data/processed/satellite_images/multiday"
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))

print(f"\nTotal observations: {len(files)}")
print("-" * 75)

expected_shape = (1000, 1000)
channels = ["B07", "B08", "B13"]

for file in files:
    data = np.load(file)

    print(f"\nFile: {os.path.basename(file)}")

    for band in channels:
        if band not in data:
            print(f"  {band}: MISSING")
            continue

        arr = data[band].astype(np.float32)
        valid = np.isfinite(arr)

        if arr.shape != expected_shape:
            print(f"  {band}: WRONG SHAPE {arr.shape}")
            continue

        if not valid.any():
            print(f"  {band}: NO VALID PIXELS")
            continue

        values = arr[valid]

        print(f"  {band}:")
        print(f"    Shape: {arr.shape}")
        print(f"    Valid pixels: {valid.mean() * 100:.2f}%")
        print(f"    Min: {values.min():.2f} K")
        print(f"    Max: {values.max():.2f} K")
        print(f"    Mean: {values.mean():.2f} K")

    data.close()

print("\nValidation completed.")

