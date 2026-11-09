
import torch

from satellite_dataset import create_dataloaders
from train_masked_encoder import random_square_mask


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

train_loader, _ = create_dataloaders(
    batch_size=2,
    image_size=200
)

images, filenames = next(iter(train_loader))
images = images.to(device)

mask = random_square_mask(
    batch_size=images.size(0),
    height=images.size(2),
    width=images.size(3),
    device=device
)

masked_images = images * mask

print("Original image shape:", images.shape)
print("Mask shape:", mask.shape)
print("Masked image shape:", masked_images.shape)

for i in range(images.size(0)):

    hidden_pixels = (mask[i, 0] == 0).sum().item()
    total_pixels = mask[i, 0].numel()

    print(
        f"Image {i + 1}: "
        f"Hidden pixels = {hidden_pixels}/{total_pixels} "
        f"({hidden_pixels / total_pixels * 100:.2f}%)"
    )

# Verify that one mask is shared across all three channels
assert mask.shape == (2, 1, 200, 200)

assert torch.equal(
    masked_images[:, 0] == 0,
    masked_images[:, 1] == 0
)

assert torch.equal(
    masked_images[:, 1] == 0,
    masked_images[:, 2] == 0
)

assert torch.isfinite(masked_images).all()

print("\nMASKING PIPELINE TEST PASSED")

