import torch
import torch.nn.functional as F
import pandas as pd

from models.se_res_autoencoder import SEResEncoder
from satellite_dataset import create_dataloaders

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINTS = {
    "original": "checkpoints/original_encoder.pth",
    "masked": "checkpoints/best_masked_encoder.pth"
}


def load_encoder(path):
    checkpoint = torch.load(path, map_location="cpu")

    encoder = SEResEncoder(embedding_dim=512)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = encoder.to(DEVICE)
    encoder.eval()

    return encoder


def encode_dataset(encoder, loader):
    embeddings = []
    filenames = []

    with torch.no_grad():
        for images, names in loader:
            images = images.to(DEVICE)
            output = encoder(images)

            embeddings.append(
                F.normalize(output, dim=1).cpu()
            )
            filenames.extend(names)

    return torch.cat(embeddings), filenames


train_loader, test_loader = create_dataloaders(
    batch_size=2,
    image_size=200
)

all_results = {}

for name, path in CHECKPOINTS.items():

    print("\n" + "=" * 55)
    print("ENCODER:", name)
    print("=" * 55)

    encoder = load_encoder(path)

    database, database_names = encode_dataset(
        encoder, train_loader
    )

    queries, query_names = encode_dataset(
        encoder, test_loader
    )

    similarity = queries @ database.T

    rows = []

    for i, query_name in enumerate(query_names):

        ranking = torch.argsort(
            similarity[i],
            descending=True
        )

        top5 = [
            database_names[j]
            for j in ranking[:5].tolist()
        ]

        rows.append({
            "query": query_name,
            "top1": top5[0],
            "top5": " | ".join(top5)
        })

        print("\nQuery:", query_name)
        print("Top 1:", top5[0])
        print("Top 5:", top5)

    all_results[name] = pd.DataFrame(rows)

    output = (
        f"results/encoder_retrieval_{name}.csv"
    )

    all_results[name].to_csv(output, index=False)

    print("\nSaved:", output)

print("\nRETRIEVAL COMPARISON COMPLETE")
