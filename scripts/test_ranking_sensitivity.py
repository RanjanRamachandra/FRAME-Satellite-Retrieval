import torch
import torch.nn.functional as F
import pandas as pd

from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel
from satellite_dataset import create_dataloaders

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

ALPHAS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]

def load_encoder(path):
    model = SEResEncoder(embedding_dim=512).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model.eval()
    return model

original = load_encoder("checkpoints/original_encoder.pth")
teacher = load_encoder("checkpoints/best_masked_encoder.pth")

frame = FRAMEModel(512).to(DEVICE)
ckpt = torch.load(
    "checkpoints/best_frame_model.pth",
    map_location=DEVICE
)
frame.load_state_dict(ckpt["frame_model_state_dict"])
frame.eval()

train_loader, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

database = []

with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(DEVICE)
        emb = F.normalize(teacher(images), dim=1)
        database.append(emb)

database = torch.cat(database, dim=0)

results = []

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

        original_emb = original(images)
        teacher_emb = F.normalize(
            teacher(images * mask), dim=1
        )
        frame_emb = F.normalize(
            frame(original_emb, mask), dim=1
        )

        teacher_scores = teacher_emb @ database.T
        teacher_top5 = torch.topk(
            teacher_scores, 5, dim=1
        ).indices[0].tolist()

        for alpha in ALPHAS:
            query = F.normalize(
                (1-alpha) * teacher_emb + alpha * frame_emb,
                dim=1
            )

            scores = query @ database.T
            top5 = torch.topk(scores, 5, dim=1).indices[0].tolist()

            results.append({
                "query_index": i + 1,
                "alpha": alpha,
                "top1_agreement": int(
                    top5[0] == teacher_top5[0]
                ),
                "top5_overlap": len(
                    set(top5) & set(teacher_top5)
                )
            })

df = pd.DataFrame(results)
df.to_csv(
    "results/ranking_sensitivity.csv",
    index=False
)

summary = df.groupby("alpha").agg(
    top1_agreement=("top1_agreement", "mean"),
    average_top5_overlap=("top5_overlap", "mean")
)

print("\nRANKING SENSITIVITY RESULTS")
print(summary.to_string())

print("\nSaved: results/ranking_sensitivity.csv")
