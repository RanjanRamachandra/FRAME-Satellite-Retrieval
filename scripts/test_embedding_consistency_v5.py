
import torch
import torch.nn.functional as F
from satellite_dataset import create_dataloaders
from models.spatial_autoencoder_v4 import SpatialSEResAutoencoderV4

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINT = "checkpoints/best_spatial_teacher_v5.pth"

_, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

model = SpatialSEResAutoencoderV4().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

positions = ["top_left", "center", "bottom_right"]
mask_sizes = [80, 120, 160]

embeddings_by_image = {}


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
        filename = filenames[0]

        # Full-image embedding
        full_mask = torch.ones(
            images.shape[0], 1,
            images.shape[2], images.shape[3],
            device=DEVICE
        )

        full_embedding = model.encoder(
            images, full_mask
        )

        full_embedding = F.normalize(
            full_embedding, dim=1
        )

        masked_embeddings = {}

        for side in mask_sizes:

            for position in positions:

                mask = create_mask(
                    images, side, position
                )

                masked_images = images * mask

                embedding = model.encoder(
                    masked_images, mask
                )

                embedding = F.normalize(
                    embedding, dim=1
                )

                masked_embeddings[
                    (side, position)
                ] = embedding

        embeddings_by_image[filename] = {
            "full": full_embedding,
            "masked": masked_embeddings
        }


print("\nV5 SAME-IMAGE MASK CONSISTENCY")
print("================================")

for side in mask_sizes:

    pair_scores = []

    for filename, data in embeddings_by_image.items():

        positions_pairs = [
            ("top_left", "center"),
            ("top_left", "bottom_right"),
            ("center", "bottom_right")
        ]

        for first, second in positions_pairs:

            a = data["masked"][(side, first)]
            b = data["masked"][(side, second)]

            similarity = F.cosine_similarity(
                a, b, dim=1
            ).item()

            pair_scores.append(similarity)

    print(
        f"Mask {side:3d} | "
        f"Mean cosine: "
        f"{sum(pair_scores)/len(pair_scores):.6f}"
    )


print("\nV5 DIFFERENT-IMAGE SIMILARITY")
print("================================")

full_embeddings = []

for filename, data in embeddings_by_image.items():
    full_embeddings.append(data["full"])

full_embeddings = torch.cat(full_embeddings, dim=0)

similarity_matrix = torch.mm(
    full_embeddings,
    full_embeddings.T
)

n = similarity_matrix.shape[0]

different_image_scores = similarity_matrix[
    ~torch.eye(n, dtype=torch.bool, device=DEVICE)
]

print("Number of test images:", n)
print(
    "Mean different-image cosine:",
    different_image_scores.mean().item()
)
print(
    "Minimum different-image cosine:",
    different_image_scores.min().item()
)
print(
    "Maximum different-image cosine:",
    different_image_scores.max().item()
)
print(
    "Standard deviation:",
    different_image_scores.std().item()
)

