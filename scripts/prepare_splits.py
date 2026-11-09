
import os
import glob
import csv
import json
import numpy as np

DATA_DIR = "data/processed/satellite_images/7days"
OUTPUT_DIR = "data/processed/satellite_images/7days"

BANDS = ["B07", "B08", "B13"]
EXPECTED_SHAPE = (1000, 1000)

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))

if len(files) != 56:
    raise ValueError(
        f"Expected 56 observations, found {len(files)}"
    )

# Chronological 80/20 split
split_index = int(round(len(files) * 0.8))

train_files = files[:split_index]
test_files = files[split_index:]

print("Training observations:", len(train_files))
print("Testing observations:", len(test_files))

# Save chronological split
split_file = os.path.join(OUTPUT_DIR, "dataset_split.csv")

with open(split_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "split"])

    for path in train_files:
        writer.writerow([os.path.basename(path), "train"])

    for path in test_files:
        writer.writerow([os.path.basename(path), "test"])

# Calculate normalization statistics using training data only
stats = {}

for band in BANDS:

    total = 0.0
    total_squared = 0.0
    count = 0

    print(f"\nCalculating statistics for {band}")

    for path in train_files:

        with np.load(path) as data:
            arr = data[band].astype(np.float64)

        if arr.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Incorrect shape in {path}: {arr.shape}"
            )

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

    stats[band] = {
        "mean_kelvin": float(mean),
        "std_kelvin": float(std),
        "valid_pixel_count": int(count)
    }

    print(f"  Mean: {mean:.4f} K")
    print(f"  Std:  {std:.4f} K")
    print(f"  Valid pixels: {count}")

stats_file = os.path.join(
    OUTPUT_DIR,
    "normalization_stats.json"
)

with open(stats_file, "w") as f:
    json.dump(stats, f, indent=4)

print("\n========== DATASET PREPARATION COMPLETE ==========")
print("Total observations:", len(files))
print("Training:", len(train_files))
print("Testing:", len(test_files))
print("Split file:", split_file)
print("Normalization file:", stats_file)

print("\nFirst training observation:", os.path.basename(train_files[0]))
print("Last training observation:", os.path.basename(train_files[-1]))
print("First testing observation:", os.path.basename(test_files[0]))
print("Last testing observation:", os.path.basename(test_files[-1]))

