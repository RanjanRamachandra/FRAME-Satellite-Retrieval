import torch

from models.spatial_encoder import SpatialSEResEncoder

CHECKPOINT = "checkpoints/best_masked_encoder_aligned.pth"
OUTPUT = "checkpoints/spatial_teacher_v3_initial.pth"

checkpoint = torch.load(CHECKPOINT, map_location="cpu")

old_weights = checkpoint["encoder_state_dict"]

model = SpatialSEResEncoder(embedding_dim=512)

new_weights = model.state_dict()

# Transfer all matching backbone weights.
transferred = []

for key in new_weights:

    if key in old_weights:
        if new_weights[key].shape == old_weights[key].shape:
            new_weights[key] = old_weights[key]
            transferred.append(key)

model.load_state_dict(new_weights)

# Preserve the original learned embedding projection.
model.spatial_pool.projection.weight.data.copy_(
    old_weights["embedding.weight"]
)

model.spatial_pool.projection.bias.data.copy_(
    old_weights["embedding.bias"]
)
# Start with uniform spatial attention.
# The attention will learn during V3 training.
with torch.no_grad():
    model.spatial_pool.attention[2].weight.zero_()
    model.spatial_pool.attention[2].bias.zero_()

torch.save(
    {
        "encoder_state_dict": model.state_dict(),
        "embedding_dim": 512,
        "source": "V2 aligned teacher"
    },
    OUTPUT
)

print("V3 INITIALIZATION COMPLETE")
print("Transferred backbone tensors:", len(transferred))
print("Projection weights transferred: Yes")
print("New attention layers: Random initialization")
print("Saved:", OUTPUT)
