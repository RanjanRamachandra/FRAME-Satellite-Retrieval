from glob import glob
from satpy import Scene

folder = "data/processed/hsd"

files = glob(folder + "/*.DAT")

print("Total files:", len(files))

scn = Scene(
    filenames=files,
    reader="ahi_hsd"
)

print("\nAvailable datasets:")
print(scn.available_dataset_names())

bands = ["B07", "B08", "B13"]

print("\nLoading satellite bands...")

scn.load(bands)

for band in bands:
    data = scn[band]
    print("\nBand:", band)
    print("Dimensions:", data.shape)
    print("Units:", data.attrs.get("units"))
    print("Calibration:", data.attrs.get("calibration"))
