
import torch
import torch.nn as nn
import torch.nn.functional as F

from .se_res_autoencoder import SEResidualBlock


class SpatialAttentionPoolingV4(nn.Module):

    def __init__(self, channels=256, embedding_dim=512):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.projection = nn.Linear(channels, embedding_dim)

    def forward(self, feature_map, mask):

        batch, channels, height, width = feature_map.shape

        # Resize input mask to feature-map resolution
        visible_mask = F.interpolate(
            mask,
            size=(height, width),
            mode="nearest"
        )

        scores = self.attention(feature_map)

        # Suppress attention over masked regions
        scores = scores.masked_fill(
            visible_mask < 0.5,
            -1e4
        )

        weights = torch.softmax(
            scores.flatten(2),
            dim=2
        ).view(batch, 1, height, width)

        pooled = (
            feature_map * weights
        ).sum(dim=(2, 3))

        embedding = self.projection(pooled)

        return embedding, weights


class SpatialSEResEncoderV4(nn.Module):

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

        self.spatial_pool = SpatialAttentionPoolingV4(
            channels=256,
            embedding_dim=embedding_dim
        )

    def forward(self, x, mask, return_attention=False):

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feature_map = self.layer4(x)

        embedding, attention = self.spatial_pool(
            feature_map,
            mask
        )

        if return_attention:
            return embedding, attention

        return embedding

