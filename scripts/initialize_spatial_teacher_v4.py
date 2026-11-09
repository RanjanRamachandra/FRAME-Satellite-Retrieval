
import torch
from pathlib import Path

from models.spatial_autoencoder_v4 import SpatialSEResAutoencoderV4

ROOT = Path(__file__).resolve().parents[1]

V3_PATH = ROOT / "checkpoints/best_spatial_teacher_v3.pth"
V4_PATH = ROOT / "checkpoints/spatial_teacher_v4_initial.pth"

device = torch.device("cpu")

v3 = torch.load(V3_PATH, map_location=device)
model = SpatialSEResAutoencoderV4(embedding_dim=512)

v3_state = v3["model_state_dict"]
v4_state = model.state_dict()

# Transfer matching encoder backbone weights.
backbone = {}

for key, value in v3_state.items():

    if key.startswith("encoder."):

        new_key = key.replace("encoder.", "", 1)

        if new_key.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4.")):
            backbone["encoder." + new_key] = value

result = model.load_state_dict(backbone, strict=False)

# Transfer compatible embedding projection weights.
model.encoder.spatial_pool.projection.load_state_dict(
    {
        "weight": v3_state["encoder.spatial_pool.projection.weight"],
        "bias": v3_state["encoder.spatial_pool.projection.bias"]
    }
)

print("Projection weights transferred: Yes")

# Transfer decoder weights.
decoder = {
    key.replace("decoder.", "", 1): value
    for key, value in v3_state.items()
    if key.startswith("decoder.")
}

decoder_result = model.decoder.load_state_dict(decoder)

# Start attention with zero logits (uniform over visible cells).
with torch.no_grad():
    model.encoder.spatial_pool.attention[2].weight.zero_()
    model.encoder.spatial_pool.attention[2].bias.zero_()

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "embedding_dim": 512,
        "architecture": "SpatialSEResAutoencoderV4"
    },
    V4_PATH
)

print("V4 INITIALIZATION COMPLETE")
print("Backbone tensors transferred:", len(backbone))
print("Decoder transfer:", decoder_result)
print("Attention initialized: Zero logits")
print("Missing new parameters:", result.missing_keys)
print("Saved:", V4_PATH)

