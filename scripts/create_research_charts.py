
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("results/expanded_2022")
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(BASE / "final_test_method_summary.csv")
pairs = pd.read_csv(BASE / "final_test_pairwise_comparison.csv")
queries = pd.read_csv(BASE / "final_test_per_query.csv")

methods = ["Teacher", "FRAME", "Autoencoder", "Pixel", "SSIM"]

# 1. Unique Top-1 database matches
summary["method"] = pd.Categorical(
    summary["method"], categories=methods, ordered=True
)
summary = summary.sort_values("method")

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(summary["method"].astype(str), summary["unique_top1_matches"])
ax.set_title("Unique Top-1 Matches by Retrieval Method")
ax.set_ylabel("Unique database observations")
ax.set_xlabel("Retrieval method")
ax.set_ylim(0, 56)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "01_unique_top1_matches.png", dpi=300)
plt.close(fig)

# Create consistent unordered pair labels
pairs["pair"] = pairs.apply(
    lambda r: " vs ".join(sorted([r["method_a"], r["method_b"]])),
    axis=1
)
pair_order = sorted(pairs["pair"].unique())

# 2. Pairwise Top-1 agreement
top1 = (
    pairs.groupby("pair", observed=True)["top1_agreement"]
    .mean()
    .reindex(pair_order) * 100
)

fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(top1.index, top1.values)
ax.set_title("Pairwise Top-1 Agreement")
ax.set_xlabel("Queries with identical Top-1 result (%)")
ax.set_xlim(0, 100)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "02_pairwise_top1_agreement.png", dpi=300)
plt.close(fig)

# 3. Pairwise average Top-5 overlap
top5 = (
    pairs.groupby("pair", observed=True)["top5_overlap"]
    .mean()
    .reindex(pair_order)
)

fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(top5.index, top5.values)
ax.set_title("Pairwise Average Top-5 Overlap")
ax.set_xlabel("Average shared observations in Top-5 (0 to 5)")
ax.set_xlim(0, 5)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "03_pairwise_top5_overlap.png", dpi=300)
plt.close(fig)

# 4. Temporal separation of Top-1 retrievals
queries["query_time"] = pd.to_datetime(
    queries["query"], format="%Y%m%d_%H%M"
)
queries["top1_time"] = pd.to_datetime(
    queries["top1"], format="%Y%m%d_%H%M"
)
queries["temporal_distance_hours"] = (
    queries["query_time"] - queries["top1_time"]
).abs().dt.total_seconds() / 3600

temporal = (
    queries.groupby("method")["temporal_distance_hours"]
    .agg(["mean", "median"])
    .reindex(methods)
)

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(methods))
width = 0.35
ax.bar(
    [i - width / 2 for i in x],
    temporal["mean"],
    width,
    label="Mean"
)
ax.bar(
    [i + width / 2 for i in x],
    temporal["median"],
    width,
    label="Median"
)
ax.set_xticks(list(x))
ax.set_xticklabels(methods)
ax.set_title("Temporal Separation of Top-1 Retrievals")
ax.set_ylabel("Absolute time difference (hours)")
ax.set_xlabel("Retrieval method")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "04_temporal_separation.png", dpi=300)
plt.close(fig)

print("CHART GENERATION COMPLETED")
for path in sorted(OUT.glob("*.png")):
    print(path)
print("\nTEMPORAL SUMMARY")
print(temporal.round(2))
print("\nPAIRWISE SUMMARY")
print(pd.DataFrame({
    "top1_agreement_percent": top1,
    "mean_top5_overlap": top5
}).round(3))
