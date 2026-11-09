
from pathlib import Path
import torch

from models.se_res_autoencoder import SEResEncoder


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = ROOT / "checkpoints/best_autoencoder.pth"
OUTPUT = ROOT / "checkpoints/original_encoder.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Loading best autoencoder checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

full_state = checkpoint["model_state_dict"]

# Extract only the original encoder weights
encoder_state = {
    key.replace("encoder.", "", 1): value
    for key, value in full_state.items()
    if key.startswith("encoder.")
}

encoder = SEResEncoder(embedding_dim=512)

encoder.load_state_dict(encoder_state)
encoder = encoder.to(device)
encoder.eval()

# Freeze all encoder parameters
for parameter in encoder.parameters():
    parameter.requires_grad = False

# Verify embedding generation
sample = torch.randn(2, 3, 200, 200, device=device)

with torch.no_grad():
    embeddings = encoder(sample)

assert embeddings.shape == (2, 512)
assert torch.isfinite(embeddings).all()

torch.save(
    {
        "encoder_state_dict": encoder_state,
        "embedding_dim": 512,
        "best_epoch": checkpoint["epoch"],
        "validation_loss": checkpoint["validation_loss"],
    },
    OUTPUT
)

print("\nOriginal encoder export successful")
print("Embedding shape:", embeddings.shape)
print("Best epoch:", checkpoint["epoch"])
print("Validation loss:", checkpoint["validation_loss"])
print("Device:", device)
print("Frozen parameters:", all(
    not p.requires_grad for p in encoder.parameters()
))
print("Saved:", OUTPUT)

