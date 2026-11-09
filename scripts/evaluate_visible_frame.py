import os
import torch
import torch.nn.functional as F
import pandas as pd

from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel
from models.visible_frame_model import VisibleFRAMEModel
from satellite_dataset import create_dataloaders

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EMBEDDING_DIM = 512
TOP_K = 5

print("Evaluation device:", DEVICE)

# Load checkpoints
original_ckpt = torch.load(
    "checkpoints/original_encoder.pth", map_location=DEVICE
)
teacher_ckpt = torch.load(
    "checkpoints/best_masked_encoder.pth", map_location=DEVICE
)
frame_ckpt = torch.load(
    "checkpoints/best_frame_model.pth", map_location=DEVICE
)
visible_ckpt = torch.load(
    "checkpoints/best_visible_frame_model.pth", map_location=DEVICE
)

original_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
teacher_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
frame_model = FRAMEModel(EMBEDDING_DIM).to(DEVICE)
visible_model = VisibleFRAMEModel(EMBEDDING_DIM).to(DEVICE)

original_encoder.load_state_dict(original_ckpt["encoder_state_dict"])
teacher_encoder.load_state_dict(teacher_ckpt["encoder_state_dict"])
frame_model.load_state_dict(frame_ckpt["frame_model_state_dict"])
visible_model.load_state_dict(
    visible_ckpt["visible_frame_model_state_dict"]
)

for model in [original_encoder, teacher_encoder, frame_model, visible_model]:
    model.eval()

print("All checkpoints loaded successfully.")

def create_mask(height, width, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    mask = torch.ones(1, 1, height, width)

    side = torch.randint(40, 101, (1,), generator=generator).item()
    top = torch.randint(0, height - side + 1, (1,), generator=generator).item()
    left = torch.randint(0, width - side + 1, (1,), generator=generator).item()

    mask[:, :, top:top + side, left:left + side] = 0
    return mask.to(DEVICE)

# Load datasets
train_loader, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

print("Database observations:", len(train_loader.dataset))
print("Query observations:", len(test_loader.dataset))

# Historical database using the frozen Teacher encoder
database_embeddings = []
database_filenames = []

with torch.no_grad():
    for images, filenames in train_loader:
        images = images.to(DEVICE)
        embedding = teacher_encoder(images)
        database_embeddings.append(embedding)
        database_filenames.extend(list(filenames))

database_embeddings = F.normalize(
    torch.cat(database_embeddings, dim=0), p=2, dim=1
)

# Evaluate all methods
results = []
summary = []

with torch.no_grad():
    for index, (images, filenames) in enumerate(test_loader):
        images = images.to(DEVICE)
        filename = filenames[0]

        mask = create_mask(
            images.shape[2],
            images.shape[3],
            seed=1000 + index
        )

        masked_images = images * mask

        original_embedding = original_encoder(images)
        teacher_embedding = teacher_encoder(masked_images)

        frame_embedding = frame_model(
            original_embedding, mask
        )

        visible_embedding = visible_model(
            original_embedding, images, mask
        )

        query_embeddings = {
            "Teacher": teacher_embedding,
            "FRAME": frame_embedding,
            "VisibleFRAME": visible_embedding
        }

        rankings = {}

        for method, embedding in query_embeddings.items():
            query = F.normalize(embedding, p=2, dim=1)
            scores = (query @ database_embeddings.T).squeeze(0)
            top = torch.topk(scores, TOP_K)
            rankings[method] = [
                database_filenames[i.item()] for i in top.indices
            ]

            for rank in range(TOP_K):
                db_index = top.indices[rank].item()
                results.append({
                    "query": filename,
                    "method": method,
                    "rank": rank + 1,
                    "retrieved_image": database_filenames[db_index],
                    "similarity": top.values[rank].item()
                })

        teacher_set = set(rankings["Teacher"])
        frame_set = set(rankings["FRAME"])
        visible_set = set(rankings["VisibleFRAME"])

        teacher_cosine = F.cosine_similarity(
            visible_embedding, teacher_embedding, dim=1
        ).item()

        summary.append({
            "query": filename,
            "visible_teacher_cosine": teacher_cosine,
            "visible_teacher_top5_overlap": len(
                visible_set & teacher_set
            ),
            "visible_frame_top5_overlap": len(
                visible_set & frame_set
            ),
            "visible_teacher_top1_agreement": int(
                rankings["VisibleFRAME"][0] == rankings["Teacher"][0]
            ),
            "visible_frame_top1_agreement": int(
                rankings["VisibleFRAME"][0] == rankings["FRAME"][0]
            )
        })

        print("\nQuery:", filename)
        print("Teacher Top-1:", rankings["Teacher"][0])
        print("FRAME Top-1:", rankings["FRAME"][0])
        print("VisibleFRAME Top-1:", rankings["VisibleFRAME"][0])
        print("VisibleFRAME-Teacher cosine:", round(teacher_cosine, 6))

os.makedirs("results", exist_ok=True)

pd.DataFrame(results).to_csv(
    "results/visible_frame_retrieval_results.csv", index=False
)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    "results/visible_frame_retrieval_summary.csv", index=False
)

print("\n====================================")
print("VISIBLEFRAME EVALUATION COMPLETE")
print("====================================")
print("Queries evaluated:", len(summary_df))
print(
    "Mean VisibleFRAME-Teacher cosine:",
    round(summary_df["visible_teacher_cosine"].mean(), 6)
)
print(
    "Mean VisibleFRAME-Teacher Top-5 overlap:",
    round(summary_df["visible_teacher_top5_overlap"].mean(), 3),
    "/ 5"
)
print(
    "Mean VisibleFRAME-FRAME Top-5 overlap:",
    round(summary_df["visible_frame_top5_overlap"].mean(), 3),
    "/ 5"
)
print(
    "VisibleFRAME-Teacher Top-1 agreement:",
    f'{summary_df["visible_teacher_top1_agreement"].mean() * 100:.2f}%'
)
print(
    "VisibleFRAME-FRAME Top-1 agreement:",
    f'{summary_df["visible_frame_top1_agreement"].mean() * 100:.2f}%'
)
print("Results: results/visible_frame_retrieval_results.csv")
print("Summary: results/visible_frame_retrieval_summary.csv")
