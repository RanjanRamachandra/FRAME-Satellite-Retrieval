import csv
import torch
import torch.nn.functional as F
from pathlib import Path

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResEncoder
from models.spatial_autoencoder_v4 import SpatialSEResAutoencoderV4

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

_, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

original_checkpoint = torch.load(
    ROOT / "checkpoints/original_encoder.pth",
    map_location="cpu"
)

original_model = SEResEncoder(embedding_dim=512)
original_model.load_state_dict(
    original_checkpoint["encoder_state_dict"]
)
original_model.to(DEVICE).eval()

v5_checkpoint = torch.load(
    ROOT / "checkpoints/best_spatial_teacher_v5.pth",
    map_location="cpu"
)

v5_model = SpatialSEResAutoencoderV4()
v5_model.load_state_dict(v5_checkpoint["model_state_dict"])
v5_model.to(DEVICE).eval()

rows = []

with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(DEVICE)

        full_mask = torch.ones(
            images.size(0), 1,
            images.size(2), images.size(3),
            device=DEVICE
        )

        original = original_model(images)
        v5 = v5_model.encoder(images, full_mask)

        original_norm = F.normalize(original, dim=1)
        v5_norm = F.normalize(v5, dim=1)

        cosine = F.cosine_similarity(
            original_norm, v5_norm, dim=1
        ).item()

        distance = torch.norm(
            original_norm - v5_norm, dim=1
        ).item()

        rows.append({
            "filename": filenames[0],
            "paired_cosine": cosine,
            "normalized_l2": distance,
            "original_norm": torch.norm(original).item(),
            "v5_norm": torch.norm(v5).item()
        })

cosines = torch.tensor([r["paired_cosine"] for r in rows])
distances = torch.tensor([r["normalized_l2"] for r in rows])

print("\nV5 PAIRED EMBEDDING ALIGNMENT")
print("=============================")
print("Observations:", len(rows))
print("Mean paired cosine:", cosines.mean().item())
print("Minimum paired cosine:", cosines.min().item())
print("Maximum paired cosine:", cosines.max().item())
print("Mean normalized L2:", distances.mean().item())
print("Maximum normalized L2:", distances.max().item())

print("\nPER-IMAGE RESULTS")
for r in rows:
    print(
        r["filename"],
        "| cosine:", round(r["paired_cosine"], 6),
        "| L2:", round(r["normalized_l2"], 6)
    )

output = ROOT / "results/paired_alignment_v5.csv"
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", output)
