import torch
import pandas as pd
from satellite_dataset import HimawariDataset

DEVICE = torch.device("cpu")

train = HimawariDataset(split="train", image_size=200)
dev = HimawariDataset(split="test", image_size=200)

database = {
    filename: image
    for image, filename in train
}

retrieval = pd.read_csv("results/retrieval_results.csv")

rows = []

for i, (query, filename) in enumerate(dev):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1000 + i)

    mask = torch.ones(1, 1, 200, 200)

    side = torch.randint(40, 101, (1,), generator=generator).item()
    top = torch.randint(0, 200-side+1, (1,), generator=generator).item()
    left = torch.randint(0, 200-side+1, (1,), generator=generator).item()

    mask[:, :, top:top+side, left:left+side] = 0
    visible = mask[0].expand_as(query).bool()

    scores = []

    for db_name, db_image in database.items():
        mse = ((query - db_image) ** 2)[visible].mean().item()
        scores.append((db_name, mse))

    pixel_ranked = sorted(scores, key=lambda x: x[1])
    pixel_top5 = [x[0] for x in pixel_ranked[:5]]

    for method in ["Teacher", "FRAME"]:
        method_df = retrieval[
            (retrieval["query"] == filename) &
            (retrieval["method"] == method)
        ].sort_values("rank")

        method_top5 = method_df["retrieved_image"].tolist()

        overlap = len(set(pixel_top5) & set(method_top5))

        rows.append({
            "query": filename,
            "method": method,
            "pixel_top1": pixel_top5[0],
            "method_top1": method_top5[0],
            "top1_agreement": pixel_top5[0] == method_top5[0],
            "top5_overlap": overlap
        })

results = pd.DataFrame(rows)

print("\nALL-QUERY PIXEL BASELINE COMPARISON")

print("\nTOP-1 AGREEMENT")
print(
    results.groupby("method")["top1_agreement"]
    .agg(["sum", "count", "mean"])
)

print("\nAVERAGE TOP-5 OVERLAP")
print(
    results.groupby("method")["top5_overlap"]
    .mean()
)

print("\nPER-QUERY RESULTS")
print(results.to_string(index=False))

results.to_csv(
    "results/all_pixel_baseline_comparison.csv",
    index=False
)

print("\nSaved: results/all_pixel_baseline_comparison.csv")
