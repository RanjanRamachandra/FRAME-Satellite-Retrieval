import torch
import torch.nn.functional as F
from itertools import combinations

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResEncoder

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINTS = {
    "original": "checkpoints/original_encoder.pth",
    "masked": "checkpoints/best_masked_encoder.pth"
}


def get_embeddings(checkpoint_path, loader):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    encoder = SEResEncoder(embedding_dim=512)
    encoder.load_state_dict(
        checkpoint["encoder_state_dict"]
    )
    encoder = encoder.to(DEVICE)
    encoder.eval()

    embeddings = []
    filenames = []

    with torch.no_grad():
        for images, names in loader:
            images = images.to(DEVICE)
            output = encoder(images)

            embeddings.append(output.cpu())
            filenames.extend(names)

    return torch.cat(embeddings), filenames


def analyze(name, embeddings):
    normalized = F.normalize(embeddings, dim=1)

    similarities = []

    for i, j in combinations(range(len(normalized)), 2):
        similarities.append(
            torch.dot(normalized[i], normalized[j]).item()
        )

    similarities = torch.tensor(similarities)

    print(f"\n{name}")
    print("Images:", len(embeddings))
    print("Embedding norm mean:",
          embeddings.norm(dim=1).mean().item())
    print("Embedding std:",
          embeddings.std(dim=0).mean().item())
    print("Different-image cosine mean:",
          similarities.mean().item())
    print("Minimum:",
          similarities.min().item())
    print("Maximum:",
          similarities.max().item())
    print("Cosine standard deviation:",
          similarities.std().item())


train_loader, test_loader = create_dataloaders(
    batch_size=2,
    image_size=200
)

for name, checkpoint_path in CHECKPOINTS.items():

    print("\n" + "=" * 55)
    print("CHECKPOINT:", name)
    print("=" * 55)

    train_embeddings, _ = get_embeddings(
        checkpoint_path, train_loader
    )

    test_embeddings, _ = get_embeddings(
        checkpoint_path, test_loader
    )

    analyze("TRAIN EMBEDDINGS", train_embeddings)
    analyze("DEVELOPMENT EMBEDDINGS", test_embeddings)
