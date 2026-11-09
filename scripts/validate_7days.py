
import os
import glob
import csv
import numpy as np

DATA_DIR = "data/processed/satellite_images/7days"
BANDS = ["B07", "B08", "B13"]
EXPECTED_SHAPE = (1000, 1000)

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))

report = []
errors = []

print("Total NPZ files:", len(files))

for path in files:
    filename = os.path.basename(path)
    print("\nChecking:", filename)

    try:
        with np.load(path) as data:

            if set(data.files) != set(BANDS):
                raise ValueError(
                    f"Incorrect channels: {data.files}"
                )

            for band in BANDS:
                arr = data[band]

                if arr.shape != EXPECTED_SHAPE:
                    raise ValueError(
                        f"{band}: incorrect shape {arr.shape}"
                    )

                valid = np.isfinite(arr)
                valid_pct = valid.mean() * 100

                if not valid.any():
                    raise ValueError(
                        f"{band}: no valid pixels"
                    )

                values = arr[valid]

                minimum = float(values.min())
                maximum = float(values.max())
                mean = float(values.mean())

                print(
                    f" {band}: valid={valid_pct:.2f}%, "
                    f"min={minimum:.2f} K, "
                    f"max={maximum:.2f} K, "
                    f"mean={mean:.2f} K"
                )

                report.append({
                    "file": filename,
                    "band": band,
                    "valid_percent": round(valid_pct, 3),
                    "minimum_K": round(minimum, 3),
                    "maximum_K": round(maximum, 3),
                    "mean_K": round(mean, 3)
                })

    except Exception as e:
        errors.append({
            "file": filename,
            "error": str(e)
        })
        print("FAILED:", e)

report_file = os.path.join(DATA_DIR, "quality_report.csv")

with open(report_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file", "band", "valid_percent",
            "minimum_K", "maximum_K", "mean_K"
        ]
    )
    writer.writeheader()
    writer.writerows(report)

print("\n========== VALIDATION SUMMARY ==========")
print("NPZ files:", len(files))
print("Channel records:", len(report))
print("Failed files:", len(errors))
print("Expected channel records:", 168)

if errors:
    print("\nErrors:")
    for error in errors:
        print(error)

print("\nQuality report:", report_file)

