import torch
import torch.nn as nn

from .se_res_autoencoder import SEResEncoder


class MaskEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.encoder = SEResEncoder(
            embedding_dim=embedding_dim
        )

    def forward(self, mask):
        mask = mask.repeat(1, 3, 1, 1)
        return self.encoder(mask)


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.network(x)


class FRAMEModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.mask_encoder = MaskEncoder(embedding_dim)
        self.projection_head = ProjectionHead(embedding_dim)

    def forward(self, original_embedding, mask):
        mask_embedding = self.mask_encoder(mask)

        combined = original_embedding * mask_embedding

        predicted_embedding = self.projection_head(combined)

        return predicted_embedding
