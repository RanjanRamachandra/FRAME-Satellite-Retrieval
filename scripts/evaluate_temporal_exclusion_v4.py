
import os
from datetime import datetime

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
EXCLUSION_HOURS = 12

print("V4 TEMPORAL EXCLUSION EVALUATION")
print("Device:", DEVICE)
print("Exclusion window:", EXCLUSION_HOURS, "hours")


def get_timestamp(filename):
    parts = os.path.basename(filename).split("_")

    date_part = parts[-2]
    time_part = parts[-1].replace(".npz", "")

    return datetime.strptime(
        date_part + "_" + time_part,
        "%Y%m%d_%H%M"
    )


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


# Load original encoder
original_checkpoint = torch.load(
    "checkpoints/original_encoder.pth",
    map_location=DEVICE
)

original_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)

original_encoder.load_state_dict(
    original_checkpoint["encoder_state_dict"]
)


# Load V4 model
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


# Load dataset
train_loader, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)

print("Database observations:", len(train_loader.dataset))
print("Query observations:", len(test_loader.dataset))


# Build separate embedding databases
original_database = []
v4_database = []
database_filenames = []

with torch.no_grad():

    for images, filenames in train_loader:

        images = images.to(DEVICE)

        full_mask = torch.ones(
            images.size(0),
            1,
            images.size(2),
            images.size(3),
            device=DEVICE
        )

        original_embedding = original_encoder(images)

        _, v4_embedding = v4_model(
            images,
            full_mask
        )

        original_database.append(original_embedding)
        v4_database.append(v4_embedding)

        database_filenames.extend(list(filenames))


original_database = F.normalize(
    torch.cat(original_database, dim=0),
    p=2,
    dim=1
)

v4_database = F.normalize(
    torch.cat(v4_database, dim=0),
    p=2,
    dim=1
)


print("Original database:", original_database.shape)
print("V4 database:", v4_database.shape)


results = []

top1_original_v4 = []
top1_v4_full_masked = []

top5_original_v4 = []
top5_v4_full_masked = []

eligible_counts = []


# Evaluate queries
with torch.no_grad():

    for index, (images, filenames) in enumerate(test_loader):

        images = images.to(DEVICE)
        query_filename = filenames[0]

        query_time = get_timestamp(query_filename)

        # Exclude database images within 12 hours
        eligible_indices = []

        for db_index, db_filename in enumerate(database_filenames):

            db_time = get_timestamp(db_filename)

            time_difference = abs(
                (query_time - db_time).total_seconds()
            ) / 3600

            if time_difference > EXCLUSION_HOURS:
                eligible_indices.append(db_index)

        eligible_counts.append(len(eligible_indices))

        if len(eligible_indices) < TOP_K:
            print(
                "\nSkipping query:",
                query_filename,
                "- insufficient eligible database images:",
                len(eligible_indices)
            )
            continue

        eligible_tensor = torch.tensor(
            eligible_indices,
            dtype=torch.long,
            device=DEVICE
        )

        # Create deterministic mask
        mask = create_mask(
            images.shape[2],
            images.shape[3],
            seed=1000 + index
        )

        masked_images = images * mask

        # Generate embeddings
        original_query = F.normalize(
            original_encoder(images),
            p=2,
            dim=1
        )

        full_mask = torch.ones_like(mask)

        _, v4_full_query = v4_model(
            images,
            full_mask
        )

        v4_full_query = F.normalize(
            v4_full_query,
            p=2,
            dim=1
        )

        _, v4_masked_query = v4_model(
            masked_images,
            mask
        )

        v4_masked_query = F.normalize(
            v4_masked_query,
            p=2,
            dim=1
        )

        # Restrict all databases to the same eligible images
        original_eligible = original_database.index_select(
            0, eligible_tensor
        )

        v4_eligible = v4_database.index_select(
            0, eligible_tensor
        )

        # Calculate similarities
        original_scores = (
            original_query @ original_eligible.T
        ).squeeze(0)

        v4_full_scores = (
            v4_full_query @ v4_eligible.T
        ).squeeze(0)

        v4_masked_scores = (
            v4_masked_query @ v4_eligible.T
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

        # Convert local ranking positions to database indices
        original_names = [
            database_filenames[eligible_indices[i]]
            for i in original_top.indices.tolist()
        ]

        v4_full_names = [
            database_filenames[eligible_indices[i]]
            for i in v4_full_top.indices.tolist()
        ]

        v4_masked_names = [
            database_filenames[eligible_indices[i]]
            for i in v4_masked_top.indices.tolist()
        ]

        # Agreement metrics
        original_agreement = (
            original_names[0] == v4_masked_names[0]
        )

        full_masked_agreement = (
            v4_full_names[0] == v4_masked_names[0]
        )

        overlap_original = (
            len(set(original_names) & set(v4_masked_names)) / TOP_K
        )

        overlap_full = (
            len(set(v4_full_names) & set(v4_masked_names)) / TOP_K
        )

        top1_original_v4.append(int(original_agreement))
        top1_v4_full_masked.append(int(full_masked_agreement))

        top5_original_v4.append(overlap_original)
        top5_v4_full_masked.append(overlap_full)

        print("\nQuery:", query_filename)
        print("Eligible database images:", len(eligible_indices))

        print("Original Top-1:", original_names[0])
        print("V4 Full Top-1:", v4_full_names[0])
        print("V4 Masked Top-1:", v4_masked_names[0])

        print("Original vs V4 Masked Top-1:", original_agreement)
        print("V4 Full vs Masked Top-1:", full_masked_agreement)

        print(
            "Original vs V4 Masked Top-5 overlap:",
            round(overlap_original * 100, 2), "%"
        )

        print(
            "V4 Full vs Masked Top-5 overlap:",
            round(overlap_full * 100, 2), "%"
        )

        methods = [
            ("Original", original_top, original_names),
            ("V4_Full", v4_full_top, v4_full_names),
            ("V4_Masked", v4_masked_top, v4_masked_names)
        ]

        for method, ranking, names in methods:

            for rank in range(TOP_K):

                results.append({
                    "query": query_filename,
                    "method": method,
                    "rank": rank + 1,
                    "retrieved_image": names[rank],
                    "similarity": ranking.values[rank].item(),
                    "eligible_database_count": len(eligible_indices)
                })


# Save results
os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/retrieval_temporal_exclusion_v4.csv",
    index=False
)


if not top1_original_v4:
    raise RuntimeError(
        "No queries had enough eligible database images."
    )


summary = pd.DataFrame({
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
})

summary.to_csv(
    "results/retrieval_temporal_summary_v4.csv",
    index=False
)

print("\n======================================")
print("TEMPORAL EXCLUSION EVALUATION COMPLETE")
print("======================================")

print(summary.to_string(index=False))

print(
    "\nAverage eligible database images:",
    round(sum(eligible_counts) / len(eligible_counts), 2)
)

print("\nDetailed results:")
print("results/retrieval_temporal_exclusion_v4.csv")

print("\nSummary:")
print("results/retrieval_temporal_summary_v4.csv")

