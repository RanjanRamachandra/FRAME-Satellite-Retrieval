
import torch
import torch.nn.functional as F
from models.spatial_autoencoder_v4 import SpatialSEResAutoencoderV4
from satellite_dataset import create_dataloaders

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

checkpoint = torch.load(
    "checkpoints/best_spatial_teacher_v5.pth",
    map_location=DEVICE
)

model = SpatialSEResAutoencoderV4(512).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

_, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

print("\nV5 ATTENTION LOGIT DIAGNOSTICS")

with torch.no_grad():
    for index, (images, filenames) in enumerate(test_loader):

        if index >= 3:
            break

        images = images.to(DEVICE)

        mask = torch.ones(
            1, 1, images.shape[2], images.shape[3],
            device=DEVICE
        )

        feature_map = model.encoder.stem(images)
        feature_map = model.encoder.layer1(feature_map)
        feature_map = model.encoder.layer2(feature_map)
        feature_map = model.encoder.layer3(feature_map)
        feature_map = model.encoder.layer4(feature_map)

        logits = model.encoder.spatial_pool.attention(
            feature_map
        )

        weights = torch.softmax(
            logits.flatten(2), dim=2
        )

        print("\nQuery:", filenames[0])
        print("Logit mean:", logits.mean().item())
        print("Logit std:", logits.std().item())
        print("Logit min:", logits.min().item())
        print("Logit max:", logits.max().item())
        print("Attention max:", weights.max().item())
        print("Attention min:", weights.min().item())
        print("Attention std:", weights.std().item())

