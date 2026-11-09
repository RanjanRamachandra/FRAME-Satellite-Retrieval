import os
import torch
import torch.nn.functional as F
import pandas as pd

from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel
from models.visible_frame_model import VisibleFRAMEModel
from models.visible_only_frame_model import VisibleOnlyFRAMEModel
from satellite_dataset import create_dataloaders

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EMBEDDING_DIM = 512
BATCH_SIZE = 1

os.makedirs("results", exist_ok=True)

# Load checkpoints
teacher_ckpt = torch.load(
    "checkpoints/best_masked_encoder.pth",
    map_location=DEVICE
)

teacher = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
teacher.load_state_dict(teacher_ckpt["encoder_state_dict"])
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

# Load original encoder for FRAME and VisibleFRAME
original_ckpt = torch.load(
    "checkpoints/original_encoder.pth",
    map_location=DEVICE
)

original = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
original.load_state_dict(original_ckpt["encoder_state_dict"])
original.eval()

for p in original.parameters():
    p.requires_grad = False

frame_ckpt = torch.load(
    "checkpoints/best_frame_model.pth",
    map_location=DEVICE
)

frame = FRAMEModel(EMBEDDING_DIM).to(DEVICE)
frame.load_state_dict(frame_ckpt["frame_model_state_dict"])
frame.eval()

visible_ckpt = torch.load(
    "checkpoints/best_visible_frame_model.pth",
    map_location=DEVICE
)

visible = VisibleFRAMEModel(EMBEDDING_DIM).to(DEVICE)
visible.load_state_dict(
    visible_ckpt["visible_frame_model_state_dict"]
)
visible.eval()

visible_only_ckpt = torch.load(
    "checkpoints/best_visible_only_frame_model.pth",
    map_location=DEVICE
)

visible_only = VisibleOnlyFRAMEModel(EMBEDDING_DIM).to(DEVICE)
visible_only.load_state_dict(
    visible_only_ckpt["visible_only_frame_model_state_dict"]
)
visible_only.eval()

# Load data
train_loader, test_loader = create_dataloaders(
    batch_size=BATCH_SIZE,
    image_size=200
)

# Build historical database using full-image Teacher embeddings
database_embeddings = []
database_names = []

with torch.no_grad():
    for images, filenames in train_loader:
        images = images.to(DEVICE)
        embeddings = F.normalize(teacher(images), dim=1)
        database_embeddings.append(embeddings)
        database_names.extend(filenames)

database_embeddings = torch.cat(database_embeddings, dim=0)

# Deterministic query masks
def create_mask(index, height=200, width=200):

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1000 + index)

    side = torch.randint(
        40, 101, (1,), generator=generator
    ).item()

    top = torch.randint(
        0, height - side + 1, (1,), generator=generator
    ).item()

    left = torch.randint(
        0, width - side + 1, (1,), generator=generator
    ).item()

    mask = torch.ones(1, 1, height, width)
    mask[:, :, top:top + side, left:left + side] = 0

    return mask.to(DEVICE)


results = []
summary = []

with torch.no_grad():

    for index, (images, filenames) in enumerate(test_loader):

        images = images.to(DEVICE)
        filename = filenames[0]

        mask = create_mask(index)
        masked_images = images * mask

        teacher_embedding = teacher(masked_images)

        original_embedding = original(images)

        frame_embedding = frame(
            original_embedding, mask
        )

        visible_embedding = visible(
            original_embedding, images, mask
        )

        visible_only_embedding = visible_only(
            images, mask
        )

        query_embeddings = {
            "Teacher": teacher_embedding,
            "FRAME": frame_embedding,
            "VisibleFRAME": visible_embedding,
            "VisibleOnlyFRAME": visible_only_embedding
        }

        rankings = {}

        for method, embedding in query_embeddings.items():

            normalized = F.normalize(embedding, dim=1)
            scores = normalized @ database_embeddings.T

            top_indices = torch.argsort(
                scores[0], descending=True
            )[:5]

            rankings[method] = [
                database_names[i] for i in top_indices.tolist()
            ]

            for rank, db_index in enumerate(top_indices.tolist(), 1):
                results.append({
                    "query": filename,
                    "method": method,
                    "rank": rank,
                    "candidate": database_names[db_index],
                    "similarity": scores[0, db_index].item()
                })

        teacher_top5 = set(rankings["Teacher"])
        teacher_top1 = rankings["Teacher"][0]

        for method in ["FRAME", "VisibleFRAME", "VisibleOnlyFRAME"]:

            candidate_top5 = set(rankings[method])

            overlap = len(teacher_top5 & candidate_top5)

            summary.append({
                "query": filename,
                "method": method,
                "teacher_top5_overlap": overlap,
                "teacher_top1_agreement": int(
                    rankings[method][0] == teacher_top1
                ),
                "teacher_embedding_cosine": F.cosine_similarity(
                    F.normalize(query_embeddings[method], dim=1),
                    F.normalize(teacher_embedding, dim=1),
                    dim=1
                ).item()
            })

pd.DataFrame(results).to_csv(
    "results/visible_only_frame_retrieval_results.csv",
    index=False
)

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "results/visible_only_frame_retrieval_summary.csv",
    index=False
)

print("\nVISIBLE-ONLY FRAME RETRIEVAL SUMMARY")
print(summary_df.groupby("method").agg(
    mean_teacher_cosine=("teacher_embedding_cosine", "mean"),
    mean_top5_overlap=("teacher_top5_overlap", "mean"),
    top1_agreement=("teacher_top1_agreement", "mean")
).to_string())

print("\nPer-query results:")
print(summary_df.to_string(index=False))

print("\nEvaluation complete.")
