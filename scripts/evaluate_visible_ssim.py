import torch
import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity
from scipy.ndimage import binary_erosion

from satellite_dataset import HimawariDataset

QUERY_RESULTS = "results/retrieval_results.csv"
OUTPUT = "results/visible_ssim_baseline.csv"

# Load datasets in chronological order
train = HimawariDataset(split="train", image_size=200)
dev = HimawariDataset(split="test", image_size=200)

database = {
    filename: image.numpy()
    for image, filename in train
}

retrieval = pd.read_csv(QUERY_RESULTS)

rows = []

for query_index, (query_tensor, query_name) in enumerate(dev):

    query = query_tensor.numpy()

    # Reproduce the exact evaluation mask
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1000 + query_index)

    mask = torch.ones(1, 1, 200, 200)

    side = torch.randint(
        40, 101, (1,), generator=generator
    ).item()

    top = torch.randint(
        0, 200 - side + 1, (1,), generator=generator
    ).item()

    left = torch.randint(
        0, 200 - side + 1, (1,), generator=generator
    ).item()

    mask[:, :, top:top+side, left:left+side] = 0

    visible = mask[0, 0].numpy().astype(bool)

    # Keep only locations whose complete 7x7 window is visible
    valid = binary_erosion(
        visible,
        structure=np.ones((7, 7), dtype=bool),
        border_value=0
    )

    if valid.sum() == 0:
        raise ValueError("No valid visible SSIM locations.")

    candidate_scores = []

    for db_name, db_image in database.items():

        channel_scores = []

        for channel in range(3):

            _, similarity_map = structural_similarity(
                query[channel],
                db_image[channel],
                data_range=10.0,
                win_size=7,
                full=True
            )

            channel_scores.append(
                similarity_map[valid].mean()
            )

        score = float(np.mean(channel_scores))

        candidate_scores.append((db_name, score))

    ranked = sorted(
        candidate_scores,
        key=lambda x: x[1],
        reverse=True
    )

    for rank, (db_name, score) in enumerate(ranked, start=1):
        rows.append({
            "query": query_name,
            "rank": rank,
            "retrieved_image": db_name,
            "ssim": score,
            "mask_size": side
        })

    print(
        f"Processed {query_index + 1}/11: "
        f"{query_name}"
    )

results = pd.DataFrame(rows)

results.to_csv(OUTPUT, index=False)

# Compare Top-5 results against Teacher and FRAME
print("\nVISIBLE SSIM BASELINE COMPARISON")

comparisons = []

for query_name in dev:
    filename = query_name[1]

    ssim_top5 = set(
        results[
            (results["query"] == filename) &
            (results["rank"] <= 5)
        ]["retrieved_image"]
    )

    for method in ["Teacher", "FRAME"]:

        method_top5 = set(
            retrieval[
                (retrieval["query"] == filename) &
                (retrieval["method"] == method)
            ]["retrieved_image"]
        )

        comparisons.append({
            "method": method,
            "query": filename,
            "top5_overlap": len(ssim_top5 & method_top5)
        })

comparison_df = pd.DataFrame(comparisons)

print("\nAVERAGE TOP-5 OVERLAP")
print(
    comparison_df.groupby("method")["top5_overlap"]
    .mean()
)

print("\nTOP-1 AGREEMENT")
for method in ["Teacher", "FRAME"]:

    matches = 0

    for query_name in dev:
        filename = query_name[1]

        ssim_top1 = results[
            (results["query"] == filename) &
            (results["rank"] == 1)
        ]["retrieved_image"].iloc[0]

        method_top1 = retrieval[
            (retrieval["query"] == filename) &
            (retrieval["method"] == method) &
            (retrieval["rank"] == 1)
        ]["retrieved_image"].iloc[0]

        matches += int(ssim_top1 == method_top1)

    print(f"{method}: {matches}/11 = {matches/11:.2%}")

comparison_df.to_csv(
    "results/ssim_teacher_frame_overlap.csv",
    index=False
)

print("\nSaved:", OUTPUT)
print("Saved: results/ssim_teacher_frame_overlap.csv")
print("\nSSIM EVALUATION COMPLETE")
