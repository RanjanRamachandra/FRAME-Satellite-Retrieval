
import torch
import torch.nn.functional as F

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResEncoder
from models.spatial_autoencoder_v4 import SpatialSEResAutoencoderV4

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Use exactly the same 11 test observations for both models.
_, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

# Load original encoder.
original_checkpoint = torch.load(
    "checkpoints/best_masked_encoder.pth",
    map_location="cpu"
)

original_model = SEResEncoder(embedding_dim=512)
original_model.load_state_dict(
    original_checkpoint["encoder_state_dict"]
)
original_model.to(DEVICE)
original_model.eval()

# Load V5 encoder.
v5_checkpoint = torch.load(
    "checkpoints/best_spatial_teacher_v5.pth",
    map_location="cpu"
)

v5_model = SpatialSEResAutoencoderV4().to(DEVICE)
v5_model.load_state_dict(
    v5_checkpoint["model_state_dict"]
)
v5_model.eval()

original_embeddings = []
v5_embeddings = []
filenames_list = []

with torch.no_grad():

    for images, filenames in test_loader:

        images = images.to(DEVICE)

        original_embedding = original_model(images)

        full_mask = torch.ones(
            images.shape[0],
            1,
            images.shape[2],
            images.shape[3],
            device=DEVICE
        )

        v5_embedding = v5_model.encoder(
            images,
            full_mask
        )

        original_embeddings.append(
            original_embedding.cpu()
        )

        v5_embeddings.append(
            v5_embedding.cpu()
        )

        filenames_list.extend(filenames)


def analyze_embeddings(name, embeddings):

    embeddings = torch.cat(embeddings, dim=0)

    normalized = F.normalize(
        embeddings, dim=1
    )

    similarity_matrix = normalized @ normalized.T

    n = similarity_matrix.shape[0]

    different_mask = ~torch.eye(
        n, dtype=torch.bool
    )

    similarities = similarity_matrix[different_mask]

    pairwise_distances = torch.pdist(
        normalized, p=2
    )

    print("\n", name)
    print("=" * 45)

    print("Embedding shape:", embeddings.shape)

    print(
        "Raw embedding standard deviation:",
        embeddings.std().item()
    )

    print(
        "Normalized embedding dimension std:",
        normalized.std(dim=0).mean().item()
    )

    print(
        "Mean different-image cosine:",
        similarities.mean().item()
    )

    print(
        "Minimum different-image cosine:",
        similarities.min().item()
    )

    print(
        "Maximum different-image cosine:",
        similarities.max().item()
    )

    print(
        "Cosine standard deviation:",
        similarities.std().item()
    )

    print(
        "Mean normalized Euclidean distance:",
        pairwise_distances.mean().item()
    )

    print(
        "Minimum normalized Euclidean distance:",
        pairwise_distances.min().item()
    )

    print(
        "Maximum normalized Euclidean distance:",
        pairwise_distances.max().item()
    )


print("\nORIGINAL vs V5 EMBEDDING COMPARISON")
print("====================================")

print("Test observations:", len(filenames_list))

analyze_embeddings(
    "ORIGINAL ENCODER",
    original_embeddings
)

analyze_embeddings(
    "V5 SPATIAL ENCODER",
    v5_embeddings
)

