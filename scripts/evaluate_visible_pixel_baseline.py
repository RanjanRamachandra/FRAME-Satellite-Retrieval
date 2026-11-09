import torch
import pandas as pd
from satellite_dataset import HimawariDataset

QUERY_NAME = "himawari_asia_20220120_1500.npz"

# Load datasets in fixed chronological order
train = HimawariDataset(split="train", image_size=200)
dev = HimawariDataset(split="test", image_size=200)

database = {}
for image, filename in train:
    database[filename] = image

queries = {}
for image, filename in dev:
    queries[filename] = image

query = queries[QUERY_NAME]

# Exact evaluation mask
generator = torch.Generator(device="cpu")
generator.manual_seed(1000)

mask = torch.ones(1, 1, 200, 200)

side = torch.randint(40, 101, (1,), generator=generator).item()
top = torch.randint(0, 200-side+1, (1,), generator=generator).item()
left = torch.randint(0, 200-side+1, (1,), generator=generator).item()

mask[:, :, top:top+side, left:left+side] = 0

# Visible region only
visible = mask[0].expand_as(query).bool()

scores = []

for filename, image in database.items():
    mse = ((query - image) ** 2)[visible].mean().item()

    scores.append({
        "retrieved_image": filename,
        "visible_pixel_mse": mse
    })

results = pd.DataFrame(scores).sort_values(
    "visible_pixel_mse"
).reset_index(drop=True)

results.insert(0, "rank", range(1, len(results)+1))

print("\nVISIBLE-PIXEL RETRIEVAL BASELINE")
print(results.head(10).to_string(index=False))

print("\nMASK DETAILS")
print("Side:", side)
print("Top:", top)
print("Left:", left)
print("Visible pixels:", visible.sum().item())

results.to_csv(
    "results/visible_pixel_baseline.csv",
    index=False
)

print("\nSaved: results/visible_pixel_baseline.csv")
