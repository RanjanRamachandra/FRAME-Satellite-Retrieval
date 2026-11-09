
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from satellite_dataset import create_dataloaders


# Load dataset
train_loader, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

# Build image lookup
images_by_name = {}

for images, filenames in train_loader:
    images_by_name[filenames[0]] = images[0]

for images, filenames in test_loader:
    images_by_name[filenames[0]] = images[0]


# Load retrieval results
results = pd.read_csv("results/retrieval_results.csv")

queries = results["query"].unique()[:5]

os.makedirs("results", exist_ok=True)


def create_mask(height, width, seed):

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    mask = torch.ones(1, height, width)

    side = torch.randint(
        40, 101, (1,), generator=generator
    ).item()

    top = torch.randint(
        0, height - side + 1, (1,), generator=generator
    ).item()

    left = torch.randint(
        0, width - side + 1, (1,), generator=generator
    ).item()

    mask[:, top:top + side, left:left + side] = 0

    return mask


def display_image(ax, tensor, title):

    # Convert normalized three-channel data to display range
    image = ((tensor.permute(1, 2, 0).numpy() + 3) / 6)
    image = image.clip(0, 1)

    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


fig, axes = plt.subplots(
    len(queries), 4,
    figsize=(14, 3.5 * len(queries))
)

if len(queries) == 1:
    axes = [axes]


for row, query in enumerate(queries):

    query_image = images_by_name[query]

    # Recreate the same mask used during evaluation
    test_index = list(queries).index(query)

    # Find its actual position in the complete test set
    all_test_names = [
        filename
        for _, filenames in test_loader
        for filename in filenames
    ]

    actual_index = all_test_names.index(query)

    mask = create_mask(
        query_image.shape[1],
        query_image.shape[2],
        seed=1000 + actual_index
    )

    masked_query = query_image * mask

    query_results = results[
        results["query"] == query
    ]

    frame_top1 = query_results[
        query_results["method"] == "FRAME"
    ].sort_values("rank").iloc[0]["retrieved_image"]

    teacher_top1 = query_results[
        query_results["method"] == "Teacher"
    ].sort_values("rank").iloc[0]["retrieved_image"]

    display_image(
        axes[row][0],
        query_image,
        "Original Query"
    )

    display_image(
        axes[row][1],
        masked_query,
        "Masked Query"
    )

    display_image(
        axes[row][2],
        images_by_name[frame_top1],
        "FRAME Top-1"
    )

    display_image(
        axes[row][3],
        images_by_name[teacher_top1],
        "Teacher Top-1"
    )

    print("\nQuery:", query)
    print("FRAME:", frame_top1)
    print("Teacher:", teacher_top1)


plt.suptitle(
    "Himawari-8 Region-Specific Retrieval Comparison",
    fontsize=14
)

plt.tight_layout()

output = "results/retrieval_visual_inspection.png"

plt.savefig(
    output,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nVisualization saved:", output)

