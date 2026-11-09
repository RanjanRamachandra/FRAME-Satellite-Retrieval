import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from expanded_satellite_dataset import ExpandedHimawariDataset

BASE = "results/expanded_2022"
OUTPUT = os.path.join(BASE, "blind_review")
os.makedirs(OUTPUT, exist_ok=True)

METHODS = ["Teacher", "FRAME", "Autoencoder", "Pixel", "SSIM"]
rng = np.random.default_rng(42)

print("Loading training and test images...")

train = ExpandedHimawariDataset(split="train", image_size=200)
test = ExpandedHimawariDataset(split="test", image_size=200)

database = {name: image.numpy() for image, name in train}
queries = {name: image.numpy() for image, name in test}

rankings = pd.read_csv(os.path.join(BASE, "final_test_rankings.csv"))

selected_queries = sorted(
    rng.choice(sorted(queries.keys()), size=10, replace=False).tolist()
)

key_rows = []

def display_image(ax, image, title):
    # Consistent display scaling across all images.
    rgb = np.transpose(np.clip((image + 5.0) / 10.0, 0, 1), (1, 2, 0))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=8)
    ax.axis("off")

for number, query_name in enumerate(selected_queries, start=1):
    fig, axes = plt.subplots(6, 6, figsize=(18, 17))

    fig.suptitle(
        "BLIND VISUAL REVIEW | Query " + str(number) + "\n" + query_name,
        fontsize=13
    )

    # Query image in the centre of the first row.
    for ax in axes[0]:
        ax.axis("off")

    display_image(axes[0, 3], queries[query_name], "QUERY IMAGE")

    # Randomize method order separately for each query.
    shuffled_methods = rng.permutation(METHODS).tolist()

    for row, method in enumerate(shuffled_methods, start=1):
        blind_code = chr(64 + row)

        key_rows.append({
            "query": query_name,
            "blind_code": blind_code,
            "actual_method": method
        })

        selected = rankings[
            (rankings["query"] == query_name) &
            (rankings["method"] == method)
        ].sort_values("rank").head(5)

        if len(selected) != 5:
            raise ValueError(
                f"Expected five results for {query_name}, {method}"
            )

        axes[row, 0].text(
            0.5, 0.5, blind_code,
            ha="center", va="center",
            fontsize=16, fontweight="bold"
        )
        axes[row, 0].axis("off")

        for column, (_, result) in enumerate(selected.iterrows(), start=1):
            image_name = result["retrieved_image"]

            if image_name not in database:
                raise ValueError(f"Missing database image: {image_name}")

            display_image(
                axes[row, column],
                database[image_name],
                "Rank " + str(column)
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = os.path.join(
        OUTPUT, f"blind_review_{number:02d}.png"
    )
    fig.savefig(output_file, dpi=120)
    plt.close(fig)

    print(f"Created {output_file}")

pd.DataFrame(key_rows).to_csv(
    os.path.join(OUTPUT, "method_key.csv"), index=False
)

pd.DataFrame({"query": selected_queries}).to_csv(
    os.path.join(OUTPUT, "selected_queries.csv"), index=False
)

print("\nBLINDED VISUAL REVIEW CREATED")
print("Review sheets:", len(selected_queries))
print("Images per sheet: 26 (1 query + 25 retrieved)")
print("Method mapping saved separately.")
print("Output directory:", OUTPUT)
