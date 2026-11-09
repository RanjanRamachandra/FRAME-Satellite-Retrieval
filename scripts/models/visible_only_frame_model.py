import torch
import torch.nn as nn

from .se_res_autoencoder import SEResEncoder
from .frame_model import MaskEncoder


class VisibleOnlyFRAMEModel(nn.Module):

    def __init__(self, embedding_dim=512):
        super().__init__()

        self.visible_encoder = SEResEncoder(
            embedding_dim=embedding_dim
        )

        self.mask_encoder = MaskEncoder(
            embedding_dim=embedding_dim
        )

        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, image, mask):

        visible_image = image * mask

        visible_embedding = self.visible_encoder(
            visible_image
        )

        mask_embedding = self.mask_encoder(mask)

        combined = torch.cat(
            [visible_embedding, mask_embedding],
            dim=1
        )

        return self.projection_head(combined)
