import os
import json
import numpy as np
import pandas as pd

MANIFEST = "data/expanded_2022/metadata/expanded_dataset_manifest.csv"
OUTPUT = "data/expanded_2022/metadata/normalization_stats.json"

BANDS = ["B07", "B08", "B13"]
EXPECTED_SHAPE = (1000, 1000)

df = pd.read_csv(MANIFEST)
train_df = df[df["split"] == "train"]

if len(train_df) != 112:
    raise ValueError(f"Expected 112 training observations, found {len(train_df)}")

stats = {}

for band in BANDS:
    total = 0.0
    total_squared = 0.0
    count = 0

    print(f"\nCalculating statistics for {band}")

    for path in train_df["npz_path"]:
        with np.load(path) as data:
            arr = data[band].astype(np.float64)

        if arr.shape != EXPECTED_SHAPE:
            raise ValueError(f"Incorrect shape in {path}: {arr.shape}")

        valid = np.isfinite(arr)
        values = arr[valid]

        if len(values) == 0:
            continue

        total += values.sum()
        total_squared += np.square(values).sum()
        count += len(values)

    if count == 0:
        raise ValueError(f"No valid training pixels for {band}")

    mean = total / count
    variance = max(total_squared / count - mean ** 2, 0.0)
    std = np.sqrt(variance)

    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid standard deviation for {band}: {std}")

    stats[band] = {
        "mean_kelvin": float(mean),
        "std_kelvin": float(std),
        "valid_pixel_count": int(count)
    }

    print(f"Mean: {mean:.4f} K")
    print(f"Std: {std:.4f} K")
    print(f"Valid pixels: {count}")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(OUTPUT, "w") as f:
    json.dump(stats, f, indent=4)

print("\nEXPANDED NORMALIZATION STATISTICS SAVED")
print(OUTPUT)
print("Training observations used:", len(train_df))
