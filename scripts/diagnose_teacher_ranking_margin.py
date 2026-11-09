import torch
import torch.nn.functional as F
import pandas as pd

from models.se_res_autoencoder import SEResEncoder
from satellite_dataset import create_dataloaders

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

teacher = SEResEncoder(embedding_dim=512).to(DEVICE)

ckpt = torch.load(
    "checkpoints/best_masked_encoder.pth",
    map_location=DEVICE
)
teacher.load_state_dict(ckpt["encoder_state_dict"])
teacher.eval()

train_loader, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

database = []

with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(DEVICE)
        database.append(
            F.normalize(teacher(images), dim=1)
        )

database = torch.cat(database, dim=0)

rows = []

with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        images = images.to(DEVICE)

        torch.manual_seed(1000 + i)

        mask = torch.ones(
            (1, 1, 200, 200), device=DEVICE
        )

        side = torch.randint(40, 101, (1,)).item()
        top = torch.randint(0, 201-side, (1,)).item()
        left = torch.randint(0, 201-side, (1,)).item()

        mask[:, :, top:top+side, left:left+side] = 0

        query = F.normalize(
            teacher(images * mask), dim=1
        )

        scores = (query @ database.T)[0]
        top2 = torch.topk(scores, 2)

        rows.append({
            "query_index": i + 1,
            "top1_score": top2.values[0].item(),
            "top2_score": top2.values[1].item(),
            "margin": (
                top2.values[0] - top2.values[1]
            ).item()
        })

df = pd.DataFrame(rows)

print("\nTEACHER TOP-1 / TOP-2 MARGINS")
print(df.to_string(index=False))

print("\nMean margin:", df["margin"].mean())
print("Minimum margin:", df["margin"].min())

df.to_csv(
    "results/teacher_ranking_margins.csv",
    index=False
)
