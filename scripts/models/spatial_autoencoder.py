import torch.nn as nn

from .spatial_encoder import SpatialSEResEncoder
from .se_res_autoencoder import SEResDecoder


class SpatialSEResAutoencoder(nn.Module):

    def __init__(self, embedding_dim=512):

        super().__init__()

        self.encoder = SpatialSEResEncoder(
            embedding_dim=embedding_dim
        )

        self.decoder = SEResDecoder(
            embedding_dim=embedding_dim
        )

    def forward(self, x, return_attention=False):

        if return_attention:

            embedding, attention = self.encoder(
                x,
                return_attention=True
            )

        else:

            embedding = self.encoder(x)

        reconstruction = self.decoder(embedding)

        if return_attention:
            return reconstruction, embedding, attention

        return reconstruction, embedding
