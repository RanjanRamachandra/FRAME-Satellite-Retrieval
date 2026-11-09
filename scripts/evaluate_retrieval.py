
import os
import torch
import torch.nn.functional as F
import pandas as pd

from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel
from satellite_dataset import create_dataloaders


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

EMBEDDING_DIM = 512
TOP_K = 5

print("Evaluation device:", DEVICE)


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

original_checkpoint = torch.load(
    "checkpoints/original_encoder.pth",
    map_location=DEVICE
)

masked_checkpoint = torch.load(
    "checkpoints/best_masked_encoder.pth",
    map_location=DEVICE
)

frame_checkpoint = torch.load(
    "checkpoints/best_frame_model.pth",
    map_location=DEVICE
)

original_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
original_encoder.load_state_dict(
    original_checkpoint["encoder_state_dict"]
)

masked_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
masked_encoder.load_state_dict(
    masked_checkpoint["encoder_state_dict"]
)

frame_model = FRAMEModel(EMBEDDING_DIM).to(DEVICE)
frame_model.load_state_dict(
    frame_checkpoint["frame_model_state_dict"]
)

original_encoder.eval()
masked_encoder.eval()
frame_model.eval()

print("All three checkpoints loaded successfully.")


# --------------------------------------------------
# Generate reproducible square masks
# --------------------------------------------------

def create_mask(height, width, seed):

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    mask = torch.ones(1, 1, height, width)

    side = torch.randint(
        40, 101, (1,), generator=generator
    ).item()

    top = torch.randint(
        0, height - side + 1, (1,), generator=generator
    ).item()

    left = torch.randint(
        0, width - side + 1, (1,), generator=generator
    ).item()

    mask[:, :, top:top + side, left:left + side] = 0

    return mask.to(DEVICE)


# --------------------------------------------------
# Load training and test observations
# --------------------------------------------------

train_loader, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

print("Database observations:", len(train_loader.dataset))
print("Query observations:", len(test_loader.dataset))


# --------------------------------------------------
# Create historical image database
# --------------------------------------------------

database_embeddings = []
database_filenames = []

with torch.no_grad():

    for images, filenames in train_loader:

        images = images.to(DEVICE)

        embeddings = masked_encoder(images)

        database_embeddings.append(embeddings)
        database_filenames.extend(list(filenames))

database_embeddings = torch.cat(
    database_embeddings, dim=0
)

database_embeddings = F.normalize(
    database_embeddings, p=2, dim=1
)

print("Database embedding shape:", database_embeddings.shape)


# --------------------------------------------------
# Evaluate queries
# --------------------------------------------------

results = []

frame_teacher_similarities = []
top1_agreements = []

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

        teacher_embedding = masked_encoder(masked_images)

        frame_embedding = frame_model(
            original_embedding,
            mask
        )

        # Compare FRAME prediction with teacher
        teacher_similarity = F.cosine_similarity(
            frame_embedding,
            teacher_embedding,
            dim=1
        ).item()

        frame_teacher_similarities.append(
            teacher_similarity
        )

        # Normalize query embeddings
        teacher_query = F.normalize(
            teacher_embedding, p=2, dim=1
        )

        frame_query = F.normalize(
            frame_embedding, p=2, dim=1
        )

        # Retrieve from historical image database
        teacher_scores = (
            teacher_query @ database_embeddings.T
        ).squeeze(0)

        frame_scores = (
            frame_query @ database_embeddings.T
        ).squeeze(0)

        teacher_top = torch.topk(
            teacher_scores, TOP_K
        )

        frame_top = torch.topk(
            frame_scores, TOP_K
        )

        teacher_top1 = teacher_top.indices[0].item()
        frame_top1 = frame_top.indices[0].item()

        agreement = teacher_top1 == frame_top1

        top1_agreements.append(int(agreement))

        print("\nQuery:", filename)
        print("Teacher top-1:", database_filenames[teacher_top1])
        print("FRAME top-1:", database_filenames[frame_top1])
        print("Top-1 agreement:", agreement)
        print("FRAME-teacher cosine similarity:", round(teacher_similarity, 6))

        for rank in range(TOP_K):

            db_index = frame_top.indices[rank].item()

            results.append({
                "query": filename,
                "method": "FRAME",
                "rank": rank + 1,
                "retrieved_image": database_filenames[db_index],
                "similarity": frame_top.values[rank].item()
            })

            db_index = teacher_top.indices[rank].item()

            results.append({
                "query": filename,
                "method": "Teacher",
                "rank": rank + 1,
                "retrieved_image": database_filenames[db_index],
                "similarity": teacher_top.values[rank].item()
            })


# --------------------------------------------------
# Save results
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

output_file = "results/retrieval_results.csv"

pd.DataFrame(results).to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

average_similarity = sum(
    frame_teacher_similarities
) / len(frame_teacher_similarities)

agreement_rate = sum(
    top1_agreements
) / len(top1_agreements)

print("\n====================================")
print("RETRIEVAL EVALUATION COMPLETE")
print("====================================")

print("Queries evaluated:", len(test_loader.dataset))

print(
    "Average FRAME-teacher cosine similarity:",
    round(average_similarity, 6)
)

print(
    "Top-1 retrieval agreement:",
    f"{agreement_rate * 100:.2f}%"
)

print("Results saved:", output_file)

