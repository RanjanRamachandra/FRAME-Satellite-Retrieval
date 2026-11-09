import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from satellite_dataset import HimawariDataset

RANKINGS = "results/temporal_exclusion_12h_rankings.csv"
OUTPUT = "results/human_relevance_labels.csv"

SEED = 20261109

def create_mask(height, width, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    mask = torch.ones(1, 1, height, width)

    side = torch.randint(
        40, 101, (1,), generator=generator
    ).item()

    top = torch.randint(
        0, height - side + 1, (1,), generator=generator
    ).item()

    left = torch.randint(
        0, width - side + 1, (1,), generator=generator
    ).item()

    mask[:, :, top:top+side, left:left+side] = 0

    return mask[0, 0].numpy()


def display_image(image):
    # Three channels: B07, B08, B13.
    # Stretch each channel independently for visualization.
    rgb = np.zeros(
        (image.shape[1], image.shape[2], 3),
        dtype=np.float32
    )

    for channel in range(3):
        band = image[channel]
        low, high = np.percentile(band, [2, 98])

        rgb[:, :, channel] = np.clip(
            (band - low) / max(high - low, 1e-6),
            0, 1
        )

    return rgb


print("Loading chronological datasets...")

train = HimawariDataset(
    split="train", image_size=200
)

dev = HimawariDataset(
    split="test", image_size=200
)

database = {
    filename: image.numpy()
    for image, filename in train
}

queries = {
    filename: image.numpy()
    for image, filename in dev
}

rankings = pd.read_csv(RANKINGS)

# Build one unique candidate list per query from all four methods.
tasks = []

for query_index, query_name in enumerate(queries):

    selected = rankings[
        (rankings["query"] == query_name) &
        (rankings["rank"] <= 5)
    ]

    candidates = sorted(
        set(selected["retrieved_image"])
    )

    tasks.extend([
        (query_index, query_name, candidate)
        for candidate in candidates
    ])

# Randomize presentation order without exposing method identities.
rng = random.Random(SEED)
rng.shuffle(tasks)

if os.path.exists(OUTPUT):
    labels = pd.read_csv(OUTPUT)
    completed = set(
        zip(labels["query"], labels["candidate"])
    )
else:
    labels = pd.DataFrame(
        columns=["query", "candidate", "rating"]
    )
    completed = set()

print("\nBLIND HUMAN RELEVANCE ANNOTATION")
print("Total unique query-candidate pairs:", len(tasks))
print("Already rated:", len(completed))
print("Remaining:", len(tasks) - len(completed))

print("\nRating guide:")
print("0 = Not relevant")
print("1 = Partially similar")
print("2 = Strong cloud-pattern match")
print("s = Skip for now")
print("q = Save and quit")

for task_number, (query_index, query_name, candidate) in enumerate(tasks, 1):

    if (query_name, candidate) in completed:
        continue

    query = queries[query_name]

    mask = create_mask(
        query.shape[1],
        query.shape[2],
        seed=1000 + query_index
    )

    masked_query = query * mask[np.newaxis, :, :]

    candidate_image = database[candidate]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(display_image(query))
    axes[0].set_title("Original query")

    axes[1].imshow(display_image(masked_query))
    axes[1].set_title("Masked query")

    axes[2].imshow(display_image(candidate_image))
    axes[2].set_title("Historical candidate")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Blind candidate {task_number} / {len(tasks)}",
        fontsize=13
    )

    plt.tight_layout()
    plt.show()
   
   
    completed.add((query_name, candidate))

    while True:
        answer = input(
            "\nRate cloud-pattern similarity (0/1/2, s, q): "
        ).strip().lower()

        if answer == "q":
            labels.to_csv(OUTPUT, index=False)
            print("\nSaved. You can resume this evaluation later.")
            raise SystemExit

        if answer == "s":
            break

        if answer in ["0", "1", "2"]:
            labels.loc[len(labels)] = [
                query_name,
                candidate,
                int(answer)
            ]

            labels.to_csv(OUTPUT, index=False)

            completed.add((query_name, candidate))
            break

        print("Enter 0, 1, 2, s or q.")

print("\nANNOTATION SESSION COMPLETE")
print("Total ratings:", len(labels))
print("Saved:", OUTPUT)
print("\nDo not inspect method identities until all ratings are complete.")
