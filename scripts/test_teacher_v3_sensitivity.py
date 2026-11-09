import torch
import torch.nn.functional as F

from satellite_dataset import create_dataloaders
from models.spatial_autoencoder import SpatialSEResAutoencoder

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINT = "checkpoints/best_spatial_teacher_v3.pth"

_, test_loader = create_dataloaders(
    batch_size=2,
    image_size=200
)

model = SpatialSEResAutoencoder().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

mask_sizes = [40, 80, 120, 160, 180]

positions = [
    "top_left",
    "center",
    "bottom_right"
]


def create_mask(images, side, position):

    batch, _, height, width = images.shape

    mask = torch.ones(
        batch, 1, height, width,
        device=DEVICE
    )

    if position == "top_left":
        top, left = 0, 0

    elif position == "center":
        top = (height - side) // 2
        left = (width - side) // 2

    else:
        top = height - side
        left = width - side

    mask[:, :, top:top+side, left:left+side] = 0

    return mask


cosine_values = {}
attention_values = {}

with torch.no_grad():

    for images, filenames in test_loader:

        images = images.to(DEVICE)

        _, full_embedding = model(images)

        for side in mask_sizes:

            embeddings = []

            for position in positions:

                mask = create_mask(
                    images, side, position
                )

                masked_images = images * mask

                _, embedding, attention = model(
                    masked_images,
                    return_attention=True
                )

                embeddings.append(embedding)

                key = (side, position)

                attention_values.setdefault(key, []).append(
                    attention.mean().item()
                )

            for i in range(len(positions)):

                for j in range(i + 1, len(positions)):

                    similarity = F.cosine_similarity(
                        embeddings[i],
                        embeddings[j],
                        dim=1
                    ).mean().item()

                    key = (side, positions[i], positions[j])

                    cosine_values.setdefault(key, []).append(
                        similarity
                    )

print("\nV3 MASK LOCATION SENSITIVITY")
print("================================")

for key, values in cosine_values.items():

    print(
        f"Mask {key[0]:3d} | "
        f"{key[1]} vs {key[2]} | "
        f"Cosine: {sum(values)/len(values):.6f}"
    )

print("\nV3 ATTENTION STATISTICS")

for key, values in attention_values.items():

    print(
        f"Mask {key[0]:3d} | "
        f"{key[1]} | "
        f"Mean attention: {sum(values)/len(values):.6f}"
    )
