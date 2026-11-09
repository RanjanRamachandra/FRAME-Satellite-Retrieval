import torch
import torch.nn.functional as F
from models.se_res_autoencoder import SEResEncoder
from satellite_dataset import create_dataloaders

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_loader, test_loader = create_dataloaders(
    batch_size=2, image_size=200
)

def load_images(loader):
    images = {}
    for batch, filenames in loader:
        for image, filename in zip(batch, filenames):
            images[filename] = image
    return images

checkpoint = torch.load(
    "checkpoints/best_masked_encoder.pth",
    map_location="cpu"
)

encoder = SEResEncoder(embedding_dim=512)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
encoder = encoder.to(DEVICE).eval()

database = load_images(train_loader)
queries = load_images(test_loader)

database_names = list(database.keys())
query_names = list(queries.keys())

database_tensor = torch.stack(
    [database[n] for n in database_names]
).to(DEVICE)

query_tensor = torch.stack(
    [queries[n] for n in query_names]
).to(DEVICE)

with torch.no_grad():
    db = F.normalize(encoder(database_tensor), dim=1)
    q = F.normalize(encoder(query_tensor), dim=1)
    similarity = q @ db.T

print("SIMILARITY DIAGNOSTICS")
print("Shape:", tuple(similarity.shape))
print("Minimum:", similarity.min().item())
print("Maximum:", similarity.max().item())
print("Mean:", similarity.mean().item())
print("Std:", similarity.std().item())

print("\nFIRST QUERY: EXACT TOP-5 SCORES")
values, indices = torch.topk(similarity[0], 5)

for rank, (score, index) in enumerate(zip(values, indices), 1):
    print(rank, database_names[index.item()], f"{score.item():.8f}")

print("\nQUERY EMBEDDING DIVERSITY")
print("Mean pairwise cosine:",
      (q @ q.T)[~torch.eye(len(q), dtype=torch.bool, device=DEVICE)].mean().item())

print("\nDATABASE EMBEDDING DIVERSITY")
print("Mean pairwise cosine:",
      (db @ db.T)[~torch.eye(len(db), dtype=torch.bool, device=DEVICE)].mean().item())
