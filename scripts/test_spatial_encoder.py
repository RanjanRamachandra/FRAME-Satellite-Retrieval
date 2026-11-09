import torch
import numpy as np

from satellite_dataset import create_dataloaders
from models.spatial_encoder import SpatialSEResEncoder

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", DEVICE)

train_loader, test_loader = create_dataloaders(
    batch_size=2,
    image_size=200
)

model = SpatialSEResEncoder().to(DEVICE)
model.eval()

images, filenames = next(iter(test_loader))
images = images.to(DEVICE)

with torch.no_grad():

    embeddings, attention = model(
        images,
        return_attention=True
    )

print("\nREAL SATELLITE SPATIAL TEST")
print("================================")

print("Input:", images.shape)
print("Embedding:", embeddings.shape)
print("Attention:", attention.shape)

print(
    "Attention sum:",
    attention.sum(dim=(2, 3)).cpu().numpy()
)

print(
    "Attention minimum:",
    attention.min().item()
)

print(
    "Attention maximum:",
    attention.max().item()
)

print("Files:", filenames)

print("\nREAL SATELLITE TEST PASSED")
