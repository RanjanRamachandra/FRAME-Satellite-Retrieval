import os
import csv
import torch
import torch.nn.functional as F

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResEncoder

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

EMBEDDING_DIM = 512
MASK_SIZES = [40, 80, 120, 160, 180]

print("Device:", DEVICE)

# Load test dataset
_, test_loader = create_dataloaders(
    batch_size=2,
    image_size=200
)

# Load V2 teacher
checkpoint = torch.load(
    "checkpoints/best_masked_encoder_aligned.pth",
    map_location="cpu"
)

teacher = SEResEncoder(EMBEDDING_DIM).to(DEVICE)

teacher.load_state_dict(
    checkpoint["encoder_state_dict"]
)

teacher.eval()

results = []

positions = ["top_left", "center", "bottom_right"]

with torch.no_grad():

    for images, filenames in test_loader:

        images = images.to(DEVICE)

        batch_size, channels, height, width = images.shape

        full_embeddings = teacher(images)

        for i in range(batch_size):

            image = images[i:i+1]

            full_embedding = full_embeddings[i:i+1]

            for side in MASK_SIZES:

                for position in positions:

                    mask = torch.ones(
                        1, 1, height, width,
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

                    mask[
                        :, :,
                        top:top + side,
                        left:left + side
                    ] = 0

                    masked_image = image * mask

                    masked_embedding = teacher(masked_image)

                    cosine = F.cosine_similarity(
                        full_embedding,
                        masked_embedding,
                        dim=1
                    ).item()

                    distance = torch.norm(
                        full_embedding - masked_embedding,
                        p=2,
                        dim=1
                    ).item()

                    results.append({
                        "filename": filenames[i],
                        "mask_size": side,
                        "position": position,
                        "cosine_similarity": cosine,
                        "embedding_distance": distance
                    })


# Save results
os.makedirs("results", exist_ok=True)

output_file = "results/teacher_mask_sensitivity.csv"

with open(output_file, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# Print averages
print("\nTEACHER MASK SENSITIVITY")
print("=" * 50)

for side in MASK_SIZES:

    subset = [
        row for row in results
        if row["mask_size"] == side
    ]

    avg_cosine = sum(
        row["cosine_similarity"] for row in subset
    ) / len(subset)

    avg_distance = sum(
        row["embedding_distance"] for row in subset
    ) / len(subset)

    print(
        f"Mask {side}x{side} | "
        f"Cosine: {avg_cosine:.6f} | "
        f"Distance: {avg_distance:.6f}"
    )

print("\nObservations:", len(test_loader.dataset))
print("Results saved:", output_file)
print("MASK SENSITIVITY TEST COMPLETE")
