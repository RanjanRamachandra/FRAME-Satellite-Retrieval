
import torch
import torch.nn.functional as F

from satellite_dataset import create_dataloaders
from models.spatial_autoencoder_v4 import SpatialSEResAutoencoderV4

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINT = "checkpoints/best_spatial_teacher_v4.pth"

_, test_loader = create_dataloaders(
    batch_size=2,
    image_size=200
)

model = SpatialSEResAutoencoderV4().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

mask_sizes = [40, 80, 120, 160, 180]

positions = ["top_left", "center", "bottom_right"]

results = {}
attention_results = {}
hidden_attention = []


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

    mask[:, :, top:top + side, left:left + side] = 0

    return mask


with torch.no_grad():

    for images, filenames in test_loader:

        images = images.to(DEVICE)

        for side in mask_sizes:

            embeddings = {}
            attentions = {}

            for position in positions:

                mask = create_mask(images, side, position)

                masked_images = images * mask

                _, embedding, attention = model(
                    masked_images,
                    mask,
                    return_attention=True
                )

                embeddings[position] = embedding
                attentions[position] = attention

                hidden = F.interpolate(
                    mask,
                    size=(13, 13),
                    mode="nearest"
                ) < 0.5

                hidden_mass = (
                    attention * hidden.float()
                ).sum(dim=(1, 2, 3))

                hidden_attention.extend(
                    hidden_mass.cpu().tolist()
                )

                entropy = -(
                    attention.clamp_min(1e-12)
                    * attention.clamp_min(1e-12).log()
                ).sum(dim=(1, 2, 3))

                attention_results.setdefault(
                    (side, position), []
                ).extend(entropy.cpu().tolist())

            pairs = [
                ("top_left", "center"),
                ("top_left", "bottom_right"),
                ("center", "bottom_right")
            ]

            for first, second in pairs:

                cosine = F.cosine_similarity(
                    embeddings[first],
                    embeddings[second],
                    dim=1
                )

                distance = torch.norm(
                    embeddings[first] - embeddings[second],
                    dim=1
                )

                key = (side, first, second)

                results.setdefault(key, []).append(
                    (cosine.mean().item(), distance.mean().item())
                )

print("\nV4 MASK LOCATION SENSITIVITY")
print("================================")

for key, values in results.items():

    avg_cosine = sum(v[0] for v in values) / len(values)
    avg_distance = sum(v[1] for v in values) / len(values)

    print(
        f"Mask {key[0]:3d} | "
        f"{key[1]} vs {key[2]} | "
        f"Cosine: {avg_cosine:.6f} | "
        f"Distance: {avg_distance:.6f}"
    )

print("\nV4 ATTENTION ENTROPY")

for key, values in attention_results.items():

    print(
        f"Mask {key[0]:3d} | {key[1]} | "
        f"Entropy: {sum(values)/len(values):.6f}"
    )

print("\nHIDDEN REGION ATTENTION")
print(
    "Maximum hidden attention mass:",
    max(hidden_attention)
)

print(
    "Average hidden attention mass:",
    sum(hidden_attention) / len(hidden_attention)
)

