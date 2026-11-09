
import torch
from pathlib import Path

from models.spatial_autoencoder_v4 import SpatialSEResAutoencoderV4

ROOT = Path(__file__).resolve().parents[1]

V4_PATH = ROOT / "checkpoints/best_spatial_teacher_v4.pth"
V5_PATH = ROOT / "checkpoints/spatial_teacher_v5_initial.pth"

device = torch.device("cpu")

v4 = torch.load(V4_PATH, map_location=device)

model = SpatialSEResAutoencoderV4(embedding_dim=512)

state = v4["model_state_dict"]

# Transfer the complete trained V4 model.
model.load_state_dict(state)

# Reinitialize only the final attention convolution.
with torch.no_grad():
    final_attention = model.encoder.spatial_pool.attention[2]

    torch.nn.init.normal_(
        final_attention.weight,
        mean=0.0,
        std=0.01
    )

    torch.nn.init.zeros_(final_attention.bias)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "embedding_dim": 512,
        "architecture": "SpatialSEResAutoencoderV5",
        "initialization": "V4 trained checkpoint with random attention logits"
    },
    V5_PATH
)

print("V5 INITIALIZATION COMPLETE")
print("Source: Best trained V4 checkpoint")
print("Backbone: Transferred")
print("Decoder: Transferred")
print("Projection: Transferred")
print("Attention final layer: Random initialization")
print("Attention weight norm:",
      final_attention.weight.norm().item())
print("Saved:", V5_PATH)

