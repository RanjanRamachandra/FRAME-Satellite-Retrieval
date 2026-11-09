import pandas as pd

retrieval = pd.read_csv("results/retrieval_results.csv")
pixel = pd.read_csv("results/all_pixel_baseline_comparison.csv")

def get_hours(query, retrieved):
    q = pd.to_datetime(
        query.replace("himawari_asia_", "").replace(".npz", ""),
        format="%Y%m%d_%H%M"
    )
    r = pd.to_datetime(
        retrieved.replace("himawari_asia_", "").replace(".npz", ""),
        format="%Y%m%d_%H%M"
    )
    return abs((q-r).total_seconds()) / 3600

rows = []

for _, row in pixel[pixel["method"] == "Teacher"].iterrows():
    query = row["query"]

    for method, retrieved in [
        ("Pixel", row["pixel_top1"]),
        ("Teacher", row["method_top1"])
    ]:
        rows.append({
            "query": query,
            "method": method,
            "retrieved": retrieved,
            "hours": get_hours(query, retrieved)
        })

    frame_row = pixel[
        (pixel["query"] == query) &
        (pixel["method"] == "FRAME")
    ].iloc[0]

    rows.append({
        "query": query,
        "method": "FRAME",
        "retrieved": frame_row["method_top1"],
        "hours": get_hours(query, frame_row["method_top1"])
    })

result = pd.DataFrame(rows)

print("\nTOP-1 TEMPORAL DISTANCE")
print(
    result.groupby("method")["hours"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

print("\nPER-QUERY COMPARISON")
print(
    result.pivot(
        index="query",
        columns="method",
        values="hours"
    ).to_string()
)

result.to_csv(
    "results/top1_temporal_comparison.csv",
    index=False
)

print("\nSaved: results/top1_temporal_comparison.csv")
