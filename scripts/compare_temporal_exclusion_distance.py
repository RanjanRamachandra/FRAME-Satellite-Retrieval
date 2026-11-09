import pandas as pd
from datetime import datetime

df = pd.read_csv(
    "results/temporal_exclusion_12h_rankings.csv"
)

def get_time(filename):
    value = filename.replace(
        "himawari_asia_", ""
    ).replace(".npz", "")

    return datetime.strptime(value, "%Y%m%d_%H%M")

top1 = df[df["rank"] == 1].copy()

top1["query_time"] = top1["query"].apply(get_time)
top1["retrieved_time"] = top1["retrieved_image"].apply(get_time)

top1["distance_hours"] = (
    top1["query_time"] - top1["retrieved_time"]
).abs().dt.total_seconds() / 3600

print("\nTOP-1 TEMPORAL DISTANCE AFTER 12-HOUR EXCLUSION")

print(
    top1.groupby("method")["distance_hours"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

print("\nPER-QUERY TEMPORAL DISTANCES")

print(
    top1.pivot(
        index="query",
        columns="method",
        values="distance_hours"
    ).to_string()
)

print("\nSaved analysis complete.")
