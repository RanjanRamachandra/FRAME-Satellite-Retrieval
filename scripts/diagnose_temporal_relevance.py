import pandas as pd

df = pd.read_csv("results/retrieval_results.csv")

teacher = df[df["method"] == "Teacher"].copy()

teacher["query_time"] = pd.to_datetime(
    teacher["query"].str.extract(
        r"(\d{8}_\d{4})"
    )[0],
    format="%Y%m%d_%H%M"
)

teacher["retrieved_time"] = pd.to_datetime(
    teacher["retrieved_image"].str.extract(
        r"(\d{8}_\d{4})"
    )[0],
    format="%Y%m%d_%H%M"
)

teacher["time_difference_hours"] = (
    teacher["query_time"] - teacher["retrieved_time"]
).dt.total_seconds().abs() / 3600

print("\nTEACHER TEMPORAL DISTANCE")
print(
    teacher.groupby("rank")["time_difference_hours"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

print("\nOVERALL")
print(
    teacher["time_difference_hours"]
    .describe()
    .round(2)
)

print("\nTOP-1 RETRIEVAL PER QUERY")
print(
    teacher[teacher["rank"] == 1][
        ["query", "retrieved_image", "time_difference_hours"]
    ].to_string(index=False)
)

teacher.to_csv(
    "results/teacher_temporal_relevance.csv",
    index=False
)
