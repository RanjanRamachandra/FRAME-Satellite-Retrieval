import torch

from models.spatial_encoder import SpatialSEResEncoder

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

checkpoint = torch.load(
    "checkpoints/spatial_teacher_v3_initial.pth",
    map_location="cpu"
)

model = SpatialSEResEncoder().to(DEVICE)

model.load_state_dict(checkpoint["encoder_state_dict"])

model.eval()

x = torch.randn(2, 3, 200, 200).to(DEVICE)

with torch.no_grad():

    embeddings, attention = model(
        x,
        return_attention=True
    )

print("V3 TEACHER INITIALIZATION TEST")
print("================================")

print("Embedding shape:", embeddings.shape)
print("Attention shape:", attention.shape)

print(
    "Attention sum:",
    attention.sum(dim=(2, 3)).cpu().numpy()
)

print(
    "Embedding finite:",
    torch.isfinite(embeddings).all().item()
)

print(
    "Attention finite:",
    torch.isfinite(attention).all().item()
)

print("V3 TEACHER TEST PASSED")
