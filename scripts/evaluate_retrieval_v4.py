
import os
import torch
import torch.nn.functional as F
import pandas as pd

from models.se_res_autoencoder import SEResEncoder
from models.spatial_autoencoder_v4 import SpatialSEResAutoencoderV4
from satellite_dataset import create_dataloaders


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

EMBEDDING_DIM = 512
TOP_K = 5

print("V4 Retrieval Evaluation")
print("Device:", DEVICE)


# --------------------------------------------------
# Load original encoder
# --------------------------------------------------

original_checkpoint = torch.load(
    "checkpoints/original_encoder.pth",
    map_location=DEVICE
)

original_encoder = SEResEncoder(
    EMBEDDING_DIM
).to(DEVICE)

original_encoder.load_state_dict(
    original_checkpoint["encoder_state_dict"]
)


# --------------------------------------------------
# Load trained V4 model
# --------------------------------------------------

v4_checkpoint = torch.load(
    "checkpoints/best_spatial_teacher_v4.pth",
    map_location=DEVICE
)

v4_model = SpatialSEResAutoencoderV4(
    EMBEDDING_DIM
).to(DEVICE)

v4_model.load_state_dict(
    v4_checkpoint["model_state_dict"]
)

original_encoder.eval()
v4_model.eval()

print("Original and V4 checkpoints loaded.")


# --------------------------------------------------
# Create reproducible masks
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
# Load chronological database and query observations
# --------------------------------------------------

train_loader, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

print("Database observations:", len(train_loader.dataset))
print("Query observations:", len(test_loader.dataset))


# --------------------------------------------------
# Generate separate embedding databases
# --------------------------------------------------

original_database = []
v4_database = []
database_filenames = []

with torch.no_grad():

    for images, filenames in train_loader:

        images = images.to(DEVICE)

        full_mask = torch.ones(
            images.size(0), 1,
            images.size(2), images.size(3),
            device=DEVICE
        )

        original_embedding = original_encoder(images)

        _, v4_embedding = v4_model(
            images, full_mask
        )

        original_database.append(original_embedding)
        v4_database.append(v4_embedding)

        database_filenames.extend(list(filenames))


original_database = F.normalize(
    torch.cat(original_database, dim=0),
    p=2, dim=1
)

v4_database = F.normalize(
    torch.cat(v4_database, dim=0),
    p=2, dim=1
)

print("Original database:", original_database.shape)
print("V4 database:", v4_database.shape)


# --------------------------------------------------
# Evaluate queries
# --------------------------------------------------

results = []

top1_original_v4 = []
top1_v4_full_masked = []

top5_original_v4 = []
top5_v4_full_masked = []

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

        # Original full-image query
        original_query = F.normalize(
            original_encoder(images),
            p=2, dim=1
        )

        # V4 full-image query
        full_mask = torch.ones_like(mask)

        _, v4_full_query = v4_model(
            images, full_mask
        )

        v4_full_query = F.normalize(
            v4_full_query, p=2, dim=1
        )

        # V4 masked-image query
        _, v4_masked_query = v4_model(
            masked_images, mask
        )

        v4_masked_query = F.normalize(
            v4_masked_query, p=2, dim=1
        )

        # Retrieval in each model's own embedding space
        original_scores = (
            original_query @ original_database.T
        ).squeeze(0)

        v4_full_scores = (
            v4_full_query @ v4_database.T
        ).squeeze(0)

        v4_masked_scores = (
            v4_masked_query @ v4_database.T
        ).squeeze(0)

        original_top = torch.topk(
            original_scores, TOP_K
        )

        v4_full_top = torch.topk(
            v4_full_scores, TOP_K
        )

        v4_masked_top = torch.topk(
            v4_masked_scores, TOP_K
        )

        original_names = [
            database_filenames[i]
            for i in original_top.indices.tolist()
        ]

        v4_full_names = [
            database_filenames[i]
            for i in v4_full_top.indices.tolist()
        ]

        v4_masked_names = [
            database_filenames[i]
            for i in v4_masked_top.indices.tolist()
        ]

        # Top-1 agreement
        original_v4_agreement = (
            original_names[0] == v4_masked_names[0]
        )

        full_masked_agreement = (
            v4_full_names[0] == v4_masked_names[0]
        )

        top1_original_v4.append(
            int(original_v4_agreement)
        )

        top1_v4_full_masked.append(
            int(full_masked_agreement)
        )

        # Top-5 overlap
        overlap_original_v4 = (
            len(set(original_names) & set(v4_masked_names)) / TOP_K
        )

        overlap_full_masked = (
            len(set(v4_full_names) & set(v4_masked_names)) / TOP_K
        )

        top5_original_v4.append(overlap_original_v4)
        top5_v4_full_masked.append(overlap_full_masked)

        print("\nQuery:", filename)

        print("Original top-1:", original_names[0])
        print("V4 full top-1:", v4_full_names[0])
        print("V4 masked top-1:", v4_masked_names[0])

        print("Original vs V4 masked top-1:", original_v4_agreement)
        print("V4 full vs masked top-1:", full_masked_agreement)

        print(
            "Original vs V4 masked top-5 overlap:",
            round(overlap_original_v4 * 100, 2), "%"
        )

        print(
            "V4 full vs masked top-5 overlap:",
            round(overlap_full_masked * 100, 2), "%"
        )

        # Save all retrieval rankings
        methods = [
            ("Original", original_top),
            ("V4_Full", v4_full_top),
            ("V4_Masked", v4_masked_top)
        ]

        for method, ranking in methods:

            for rank in range(TOP_K):

                db_index = ranking.indices[rank].item()

                results.append({
                    "query": filename,
                    "method": method,
                    "rank": rank + 1,
                    "retrieved_image": database_filenames[db_index],
                    "similarity": ranking.values[rank].item()
                })


# --------------------------------------------------
# Save results
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

pd.DataFrame(results).to_csv(
    "results/retrieval_results_v4.csv",
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

summary = {
    "metric": [
        "Original vs V4 masked Top-1 agreement",
        "V4 full vs V4 masked Top-1 agreement",
        "Original vs V4 masked Top-5 overlap",
        "V4 full vs V4 masked Top-5 overlap"
    ],
    "value_percent": [
        sum(top1_original_v4) / len(top1_original_v4) * 100,
        sum(top1_v4_full_masked) / len(top1_v4_full_masked) * 100,
        sum(top5_original_v4) / len(top5_original_v4) * 100,
        sum(top5_v4_full_masked) / len(top5_v4_full_masked) * 100
    ]
}

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "results/retrieval_summary_v4.csv",
    index=False
)

print("\n================================")
print("V4 RETRIEVAL EVALUATION COMPLETE")
print("================================")

print(summary_df.to_string(index=False))

print("\nDetailed results:")
print("results/retrieval_results_v4.csv")

print("\nSummary:")
print("results/retrieval_summary_v4.csv")

