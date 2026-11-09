
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

all_test_names = []

for images, filenames in test_loader:
    images_by_name[filenames[0]] = images[0]
    all_test_names.append(filenames[0])

# Load V4 retrieval results
results = pd.read_csv("results/retrieval_results_v4.csv")

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
    image = (
        (tensor.detach().cpu().permute(1, 2, 0).numpy() + 3) / 6
    )

    image = image.clip(0, 1)

    ax.imshow(image)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


# 9 columns:
# Original query, masked query,
# Original Top-1, V4 Full Top-1,
# V4 Masked Top-1 through Top-5

fig, axes = plt.subplots(
    len(queries),
    9,
    figsize=(23, 3.5 * len(queries)),
    squeeze=False
)

for row, query in enumerate(queries):

    query_image = images_by_name[query]

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

    def get_retrieved(method, rank=1):
        selected = query_results[
            (query_results["method"] == method) &
            (query_results["rank"] == rank)
        ]

        return selected.iloc[0]["retrieved_image"]

    original_top1 = get_retrieved("Original")
    v4_full_top1 = get_retrieved("V4_Full")

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
        images_by_name[original_top1],
        "Original Top-1"
    )

    display_image(
        axes[row][3],
        images_by_name[v4_full_top1],
        "V4 Full Top-1"
    )

    for rank in range(1, 6):

        retrieved = get_retrieved("V4_Masked", rank)

        display_image(
            axes[row][3 + rank],
            images_by_name[retrieved],
            f"V4 Masked Top-{rank}"
        )

    print("\nQuery:", query)
    print("Original Top-1:", original_top1)
    print("V4 Full Top-1:", v4_full_top1)

    for rank in range(1, 6):
        print(
            f"V4 Masked Top-{rank}:",
            get_retrieved("V4_Masked", rank)
        )

plt.suptitle(
    "Himawari-8 V4 Retrieval Visual Inspection",
    fontsize=15
)

plt.tight_layout()

output = "results/retrieval_visual_inspection_v4.png"

plt.savefig(
    output,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nV4 visualization saved:", output)

