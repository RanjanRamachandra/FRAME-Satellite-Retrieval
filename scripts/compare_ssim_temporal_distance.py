import pandas as pd
from datetime import datetime

df = pd.read_csv("results/visible_ssim_baseline.csv")

top1 = df[df["rank"] == 1].copy()

def get_time(filename):
    value = filename.replace("himawari_asia_", "").replace(".npz", "")
    return datetime.strptime(value, "%Y%m%d_%H%M")

top1["query_time"] = top1["query"].apply(get_time)
top1["retrieved_time"] = top1["retrieved_image"].apply(get_time)

top1["distance_hours"] = (
    top1["query_time"] - top1["retrieved_time"]
).dt.total_seconds().abs() / 3600

print("\nSSIM TOP-1 TEMPORAL DISTANCE")
print(top1["distance_hours"].describe())

print("\nPER-QUERY RESULTS")
print(
    top1[["query", "retrieved_image", "ssim", "distance_hours"]]
    .to_string(index=False)
)

print("\nSSIM TEMPORAL SUMMARY")
print("Mean:", round(top1["distance_hours"].mean(), 2))
print("Median:", round(top1["distance_hours"].median(), 2))
print("Minimum:", top1["distance_hours"].min())
print("Maximum:", top1["distance_hours"].max())

print("\nSaved analysis complete.")
