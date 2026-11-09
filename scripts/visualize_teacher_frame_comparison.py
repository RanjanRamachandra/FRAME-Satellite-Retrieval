import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from satellite_dataset import create_dataloaders

QUERY_NAME = "himawari_asia_20220120_1500.npz"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the recorded retrieval rankings
df = pd.read_csv("results/retrieval_results.csv")

query_results = df[df["query"] == QUERY_NAME]

teacher = (
    query_results[query_results["method"] == "Teacher"]
    .sort_values("rank")
)

frame = (
    query_results[query_results["method"] == "FRAME"]
    .sort_values("rank")
)

if len(teacher) != 5 or len(frame) != 5:
    raise ValueError("Expected exactly five results for each method.")

# Load images and preserve their filenames
train_loader, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

database_images = {}
query_images = {}

for batch, filenames in train_loader:
    for image, filename in zip(batch, filenames):
        database_images[filename] = image

for batch, filenames in test_loader:
    for image, filename in zip(batch, filenames):
        query_images[filename] = image

if QUERY_NAME not in query_images:
    raise ValueError(f"Query image not found: {QUERY_NAME}")

# Reproduce the exact evaluation mask
generator = torch.Generator(device="cpu")
generator.manual_seed(1000)

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

mask[:, :, top:top + side, left:left + side] = 0

print("Mask size:", side)
print("Mask position: top =", top, "left =", left)

def to_rgb(tensor):
    image = tensor.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image[:, :, [2, 1, 0]]
    return np.clip((image + 5) / 10, 0, 1)

query = query_images[QUERY_NAME]

original_rgb = to_rgb(query)

masked_rgb = original_rgb.copy()
mask_2d = mask[0, 0].numpy()
masked_rgb[mask_2d == 0] = 0

# Prepare all 12 panels
panels = [
    ("Original Query", QUERY_NAME, original_rgb),
    ("Masked Query", f"Hidden square: {side}x{side}", masked_rgb)
]

for method_name, results in [
    ("Teacher", teacher),
    ("FRAME", frame)
]:
    for _, row in results.iterrows():
        filename = row["retrieved_image"]

        if filename not in database_images:
            raise ValueError(
                f"Retrieved image not found in database: {filename}"
            )

        rgb = to_rgb(database_images[filename])

        title = (
            f"{method_name} Top-{int(row['rank'])}\n"
            f"{filename.replace('.npz', '')}\n"
            f"Cosine: {row['similarity']:.6f}"
        )

        panels.append((title, filename, rgb))

fig, axes = plt.subplots(4, 3, figsize=(19, 22))
axes = axes.flatten()

for ax, (title, _, image) in zip(axes, panels):
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.suptitle(
    "Himawari-8 Retrieval Comparison\n"
    "Query: " + QUERY_NAME,
    fontsize=15
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

os.makedirs("results", exist_ok=True)

output = "results/teacher_frame_visual_comparison.png"

plt.savefig(output, dpi=160, bbox_inches="tight")
plt.close()

print("\nSaved:", output)
print("Panels:", len(panels))
print("\nVISUAL COMPARISON COMPLETE")
