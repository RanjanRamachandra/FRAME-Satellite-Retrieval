import torch

from models.frame_model import FRAMEModel

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)

# Create FRAME model
model = FRAMEModel(embedding_dim=512).to(device)

# Simulate original image embeddings
original_embeddings = torch.randn(2, 512).to(device)

# Create two sample masks
masks = torch.ones(2, 1, 200, 200).to(device)

# Hide a square region
masks[0, :, 40:80, 50:90] = 0
masks[1, :, 70:120, 60:110] = 0

# Forward pass
predicted_embeddings = model(
    original_embeddings,
    masks
)

print("Original embedding shape:", original_embeddings.shape)
print("Mask shape:", masks.shape)
print("Predicted embedding shape:", predicted_embeddings.shape)

# Test cosine similarity loss
target_embeddings = torch.randn(2, 512).to(device)

loss = 1 - torch.nn.functional.cosine_similarity(
    predicted_embeddings,
    target_embeddings,
    dim=1
).mean()

print("Cosine loss:", loss.item())

# Backward pass
loss.backward()

print("Backward pass: successful")

# Verify gradients
gradients_finite = all(
    p.grad is None or torch.isfinite(p.grad).all()
    for p in model.parameters()
)

print("Gradients finite:", gradients_finite)

assert predicted_embeddings.shape == (2, 512)
assert gradients_finite

print("\nFRAME MODEL TEST PASSED")
