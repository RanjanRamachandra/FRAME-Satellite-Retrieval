import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from models.se_res_autoencoder import SEResEncoder
from satellite_dataset import create_dataloaders

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINT = "checkpoints/best_masked_encoder.pth"


def load_encoder():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    encoder = SEResEncoder(embedding_dim=512)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = encoder.to(DEVICE)
    encoder.eval()

    return encoder


def load_images(loader):
    images = {}
    for batch, filenames in loader:
        for image, filename in zip(batch, filenames):
            images[filename] = image
    return images


train_loader, test_loader = create_dataloaders(
    batch_size=2,
    image_size=200
)

encoder = load_encoder()

database_images = load_images(train_loader)
query_images = load_images(test_loader)

database_names = list(database_images.keys())
query_names = list(query_images.keys())

database_tensor = torch.stack(
    [database_images[name] for name in database_names]
).to(DEVICE)

query_tensor = torch.stack(
    [query_images[name] for name in query_names]
).to(DEVICE)

with torch.no_grad():
    database_embeddings = F.normalize(
        encoder(database_tensor), dim=1
    )
    query_embeddings = F.normalize(
        encoder(query_tensor), dim=1
    )

similarity = query_embeddings @ database_embeddings.T

for i, query_name in enumerate(query_names):

    ranking = torch.argsort(
        similarity[i], descending=True
    )[:5].cpu().tolist()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    query = query_images[query_name].numpy()
    query_rgb = np.transpose(query, (1, 2, 0))

    # Display B13, B08, B07 as RGB channels
    query_rgb = query_rgb[:, :, [2, 1, 0]]
    query_rgb = np.clip((query_rgb + 5) / 10, 0, 1)

    axes[0].imshow(query_rgb)
    axes[0].set_title("QUERY\n" + query_name)

    for rank, db_index in enumerate(ranking, start=1):

        filename = database_names[db_index]
        image = database_images[filename].numpy()

        rgb = np.transpose(image, (1, 2, 0))
        rgb = rgb[:, :, [2, 1, 0]]
        rgb = np.clip((rgb + 5) / 10, 0, 1)

        axes[rank].imshow(rgb)
        score = similarity[i, db_index].item()

        axes[rank].set_title(
            f"Top {rank}: {filename}\nCosine: {score:.4f}"
        )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output = (
        "results/masked_retrieval_"
        + query_name.replace(".npz", ".png")
    )

    plt.savefig(output, dpi=130)
    plt.close()

    print("Saved:", output)

print("\nVISUALIZATION COMPLETE")
