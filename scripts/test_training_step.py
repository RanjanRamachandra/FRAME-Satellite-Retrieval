
import torch
import torch.nn as nn

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResAutoencoder


def main():

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Training device:", device)

    # Load actual Himawari observations
    train_loader, test_loader = create_dataloaders(
        batch_size=2,
        image_size=200
    )

    print("Training observations:", len(train_loader.dataset))
    print("Testing observations:", len(test_loader.dataset))

    # Initialize autoencoder
    model = SEResAutoencoder(
        embedding_dim=512
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()

    model.train()

    # Perform exactly one training batch
    images, filenames = next(iter(train_loader))
    images = images.to(device)

    optimizer.zero_grad()

    reconstruction, embedding = model(images)

    loss = criterion(reconstruction, images)

    loss.backward()

    optimizer.step()

    # Synchronize MPS before reporting results
    if device.type == "mps":
        torch.mps.synchronize()

    print("\nActual satellite training batch")
    print("Files:", filenames)
    print("Input shape:", images.shape)
    print("Embedding shape:", embedding.shape)
    print("Reconstruction shape:", reconstruction.shape)
    print("MSE loss:", loss.item())

    gradients_ok = all(
        p.grad is None or torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )

    print("Gradients finite:", gradients_ok)

    assert torch.isfinite(loss).item()
    assert gradients_ok
    assert embedding.shape == (2, 512)
    assert reconstruction.shape == images.shape

    print("\nREAL DATA TRAINING STEP PASSED")


if __name__ == "__main__":
    main()

