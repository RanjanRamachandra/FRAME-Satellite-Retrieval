
import torch
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

temperatures = [1.0, 0.5, 0.25, 0.1]

print("\nATTENTION TEMPERATURE EXPERIMENT")

with torch.no_grad():
    images, filenames = next(iter(test_loader))
    images = images.to(DEVICE)

    feature_map = model.encoder.stem(images)
    feature_map = model.encoder.layer1(feature_map)
    feature_map = model.encoder.layer2(feature_map)
    feature_map = model.encoder.layer3(feature_map)
    feature_map = model.encoder.layer4(feature_map)

    logits = model.encoder.spatial_pool.attention(
        feature_map
    )

    for temperature in temperatures:

        attention = torch.softmax(
            logits.flatten(2) / temperature,
            dim=2
        )

        entropy = -(
            attention * torch.log(attention + 1e-12)
        ).sum(dim=2)

        normalized_entropy = (
            entropy / torch.log(
                torch.tensor(
                    attention.shape[-1],
                    device=DEVICE,
                    dtype=torch.float32
                )
            )
        )

        print("\nTemperature:", temperature)
        print("Attention max:", attention.max().item())
        print("Attention min:", attention.min().item())
        print("Attention std:", attention.std().item())
        print(
            "Normalized entropy:",
            normalized_entropy.mean().item()
        )

