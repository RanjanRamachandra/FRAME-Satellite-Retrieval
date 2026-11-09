
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        batch, channels, _, _ = x.shape

        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)

        return x * weights


class SEResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = nn.GroupNorm(8, out_channels)

        self.se = SqueezeExcitation(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)

        out = F.relu(out + identity)

        return out


class SEResEncoder(nn.Module):

    def __init__(self, embedding_dim=512):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 32,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = SEResidualBlock(32, 32, stride=1)
        self.layer2 = SEResidualBlock(32, 64, stride=2)
        self.layer3 = SEResidualBlock(64, 128, stride=2)
        self.layer4 = SEResidualBlock(128, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Linear(256, embedding_dim)

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x).flatten(1)

        embedding = self.embedding(x)

        return embedding


class SEResDecoder(nn.Module):

    def __init__(self, embedding_dim=512):
        super().__init__()

        self.fc = nn.Linear(
            embedding_dim,
            256 * 13 * 13
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(
            32, 3,
            kernel_size=3,
            padding=1
        )

    def forward(self, embedding):

        batch = embedding.shape[0]

        x = self.fc(embedding)
        x = x.view(batch, 256, 13, 13)

        x = F.interpolate(x, size=(25, 25), mode="bilinear",
                          align_corners=False)
        x = self.block1(x)

        x = F.interpolate(x, size=(50, 50), mode="bilinear",
                          align_corners=False)
        x = self.block2(x)

        x = F.interpolate(x, size=(100, 100), mode="bilinear",
                          align_corners=False)
        x = self.block3(x)

        x = F.interpolate(x, size=(200, 200), mode="bilinear",
                          align_corners=False)

        reconstruction = self.output(x)

        return reconstruction


class SEResAutoencoder(nn.Module):

    def __init__(self, embedding_dim=512):
        super().__init__()

        self.encoder = SEResEncoder(embedding_dim)
        self.decoder = SEResDecoder(embedding_dim)

    def forward(self, x):

        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)

        return reconstruction, embedding


if __name__ == "__main__":

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    model = SEResAutoencoder(
        embedding_dim=512
    ).to(device)

    # Two sample satellite images
    sample = torch.randn(
        2, 3, 200, 200,
        device=device
    )

    reconstruction, embedding = model(sample)

    print("Input shape:", sample.shape)
    print("Reconstruction shape:", reconstruction.shape)
    print("Embedding shape:", embedding.shape)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("Total parameters:", f"{total_params:,}")

    # Check that gradients can flow through the model
    loss = F.mse_loss(reconstruction, sample)

    loss.backward()

    print("Test reconstruction loss:", loss.item())
    print("Backward pass: successful")

    assert reconstruction.shape == sample.shape
    assert embedding.shape == (2, 512)
    assert torch.isfinite(reconstruction).all()

    print("\nSE-RES AUTOENCODER TEST PASSED")

