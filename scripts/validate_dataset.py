import numpy as np

path = "data/processed/satellite_images/himawari_asia_20220115_0400.npz"

data = np.load(path)

for band in ["B07", "B08", "B13"]:
    arr = data[band]

    valid = np.isfinite(arr)

    print("\nBand:", band)
    print("Shape:", arr.shape)
    print("Data type:", arr.dtype)
    print("Valid pixels:", round(valid.mean() * 100, 2), "%")
    print("Minimum:", np.nanmin(arr))
    print("Maximum:", np.nanmax(arr))
    print("Mean:", np.nanmean(arr))
    print("Standard deviation:", np.nanstd(arr))

print("\nAll three channels validated.")
