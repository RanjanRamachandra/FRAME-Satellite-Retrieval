import torch
import torch.nn.functional as F

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResEncoder

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

_, test_loader = create_dataloaders(batch_size=2, image_size=200)
train_loader, _ = create_dataloaders(batch_size=2, image_size=200)

checkpoint = torch.load(
    "checkpoints/best_masked_encoder.pth",
    map_location="cpu"
)

encoder = SEResEncoder(embedding_dim=512)

encoder.load_state_dict(
    checkpoint["encoder_state_dict"]
)

encoder.to(DEVICE)
encoder.eval()

embeddings = []

with torch.no_grad():
    for images, filenames in train_loader:
        images = images.to(DEVICE)

        output = encoder(images)

        embeddings.append(output)

embeddings = torch.cat(embeddings, dim=0)

normalized = F.normalize(embeddings, dim=1)

similarities = normalized @ normalized.T

mask = ~torch.eye(
    similarities.shape[0],
    dtype=torch.bool,
    device=DEVICE
)

values = similarities[mask]

print("\nDATABASE EMBEDDING DIAGNOSTICS")
print("================================")

print("Embedding shape:", embeddings.shape)

print(
    "Embedding standard deviation:",
    embeddings.std().item()
)

print(
    "Average pairwise cosine similarity:",
    values.mean().item()
)

print(
    "Minimum pairwise cosine similarity:",
    values.min().item()
)

print(
    "Maximum pairwise cosine similarity:",
    values.max().item()
)
