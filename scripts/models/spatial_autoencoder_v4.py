
import torch.nn as nn

from .spatial_encoder_v4 import SpatialSEResEncoderV4
from .se_res_autoencoder import SEResDecoder


class SpatialSEResAutoencoderV4(nn.Module):

    def __init__(self, embedding_dim=512):

        super().__init__()

        self.encoder = SpatialSEResEncoderV4(
            embedding_dim=embedding_dim
        )

        self.decoder = SEResDecoder(
            embedding_dim=embedding_dim
        )

    def forward(self, x, mask, return_attention=False):

        if return_attention:

            embedding, attention = self.encoder(
                x,
                mask,
                return_attention=True
            )

        else:

            embedding = self.encoder(x, mask)

        reconstruction = self.decoder(embedding)

        if return_attention:

            return reconstruction, embedding, attention

        return reconstruction, embedding

