import torch
import torch.nn.functional as F

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResEncoder

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

SIZES = [80, 120, 160, 180]

_, test_loader = create_dataloaders(
    batch_size=2,
    image_size=200
)

checkpoint = torch.load(
    "checkpoints/best_masked_encoder_aligned.pth",
    map_location="cpu"
)

teacher = SEResEncoder(512).to(DEVICE)
teacher.load_state_dict(checkpoint["encoder_state_dict"])
teacher.eval()

positions = ["top_left", "center", "bottom_right"]

all_results = []

with torch.no_grad():

    for images, filenames in test_loader:

        images = images.to(DEVICE)
        batch_size, channels, height, width = images.shape

        for i in range(batch_size):

            image = images[i:i+1]
            embeddings = {}

            for side in SIZES:

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

                    mask[:, :, top:top+side, left:left+side] = 0

                    embedding = teacher(image * mask)

                    embeddings[(side, position)] = embedding

            for side in SIZES:

                pairs = [
                    ("top_left", "center"),
                    ("top_left", "bottom_right"),
                    ("center", "bottom_right")
                ]

                for first, second in pairs:

                    a = embeddings[(side, first)]
                    b = embeddings[(side, second)]

                    cosine = F.cosine_similarity(
                        a, b, dim=1
                    ).item()

                    distance = torch.norm(
                        a - b, p=2, dim=1
                    ).item()

                    all_results.append({
                        "mask_size": side,
                        "position_a": first,
                        "position_b": second,
                        "cosine": cosine,
                        "distance": distance
                    })


print("\nTEACHER LOCATION SENSITIVITY")
print("=" * 55)

for side in SIZES:

    subset = [
        r for r in all_results
        if r["mask_size"] == side
    ]

    avg_cosine = sum(
        r["cosine"] for r in subset
    ) / len(subset)

    avg_distance = sum(
        r["distance"] for r in subset
    ) / len(subset)

    print(
        f"Mask {side}x{side} | "
        f"Location cosine: {avg_cosine:.6f} | "
        f"Distance: {avg_distance:.6f}"
    )

print("\nLOCATION SENSITIVITY TEST COMPLETE")
