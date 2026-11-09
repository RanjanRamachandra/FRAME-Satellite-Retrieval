import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

from expanded_satellite_dataset import ExpandedHimawariDataset
from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CHECKPOINT_DIR = "checkpoints/expanded_2022"
OUTPUT_DIR = "results/expanded_2022"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 200
EMBEDDING_DIM = 512
EXPECTED_TRAIN = 112
EXPECTED_TEST = 56

print("EXPANDED FINAL TEST EVALUATION")
print("Device:", DEVICE)


# --------------------------------------------------
# Load train and final-test datasets only
# --------------------------------------------------

train = ExpandedHimawariDataset(
    split="train",
    image_size=IMAGE_SIZE
)

test = ExpandedHimawariDataset(
    split="test",
    image_size=IMAGE_SIZE
)

assert len(train) == EXPECTED_TRAIN
assert len(test) == EXPECTED_TEST

database_images = {}
for image, filename in train:
    database_images[filename] = image.numpy()

test_images = []
test_names = []

for image, filename in test:
    test_images.append(image.numpy())
    test_names.append(filename)

print("Training database:", len(database_images))
print("Independent test queries:", len(test_images))
print("Validation dataset: not loaded")


# --------------------------------------------------
# Load best checkpoints
# --------------------------------------------------

auto_checkpoint = torch.load(
    os.path.join(CHECKPOINT_DIR, "best_autoencoder.pth"),
    map_location="cpu",
    weights_only=False
)

teacher_checkpoint = torch.load(
    os.path.join(CHECKPOINT_DIR, "best_masked_encoder.pth"),
    map_location="cpu",
    weights_only=False
)

frame_checkpoint = torch.load(
    os.path.join(CHECKPOINT_DIR, "best_frame_model.pth"),
    map_location="cpu",
    weights_only=False
)

autoencoder = SEResEncoder(EMBEDDING_DIM)

auto_state = {
    key.replace("encoder.", "", 1): value
    for key, value in auto_checkpoint["model_state_dict"].items()
    if key.startswith("encoder.")
}

autoencoder.load_state_dict(auto_state)

teacher = SEResEncoder(EMBEDDING_DIM)
teacher.load_state_dict(
    teacher_checkpoint["encoder_state_dict"]
)

frame_model = FRAMEModel(EMBEDDING_DIM)
frame_model.load_state_dict(
    frame_checkpoint["frame_model_state_dict"]
)

autoencoder = autoencoder.to(DEVICE).eval()
teacher = teacher.to(DEVICE).eval()
frame_model = frame_model.to(DEVICE).eval()

print("Autoencoder checkpoint loaded.")
print("Masked teacher checkpoint loaded.")
print("FRAME checkpoint loaded.")


# --------------------------------------------------
# Extract normalized embeddings
# --------------------------------------------------

def get_embedding(model, image):
    tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        embedding = model(tensor)

    return F.normalize(embedding, p=2, dim=1)


print("\nBuilding training database embeddings...")

teacher_database = {}
auto_database = {}

with torch.inference_mode():
    for filename, image in database_images.items():
        tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        teacher_database[filename] = F.normalize(
            teacher(tensor), p=2, dim=1
        )

        auto_database[filename] = F.normalize(
            autoencoder(tensor), p=2, dim=1
        )

print("Teacher database embeddings:", len(teacher_database))
print("Autoencoder database embeddings:", len(auto_database))


# --------------------------------------------------
# Deterministic query masks
# --------------------------------------------------

def create_mask(query_index):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1000 + query_index)

    mask = torch.ones(1, 1, IMAGE_SIZE, IMAGE_SIZE)

    side = torch.randint(
        40, 101, (1,), generator=generator
    ).item()

    top = torch.randint(
        0, IMAGE_SIZE - side + 1, (1,), generator=generator
    ).item()

    left = torch.randint(
        0, IMAGE_SIZE - side + 1, (1,), generator=generator
    ).item()

    mask[:, :, top:top + side, left:left + side] = 0

    return mask, side


# --------------------------------------------------
# Ranking helper
# --------------------------------------------------

ranking_rows = []


def save_rankings(query_name, method, scores, score_name):
    ranked = sorted(
        scores,
        key=lambda item: (
            -item[1] if score_name == "similarity" else item[1],
            item[0]
        )
    )

    for rank, (database_name, score) in enumerate(ranked, start=1):
        ranking_rows.append({
            "query": query_name,
            "method": method,
            "rank": rank,
            "retrieved_image": database_name,
            "score": float(score)
        })

    return [item[0] for item in ranked]


# --------------------------------------------------
# Final-test retrieval
# --------------------------------------------------

query_metrics = []

for query_index, (query, query_name) in enumerate(
    zip(test_images, test_names)
):

    mask, mask_side = create_mask(query_index)

    visible = mask[0, 0].numpy().astype(bool)

    masked_query = query * mask[0].numpy()

    query_tensor = torch.from_numpy(query).unsqueeze(0).to(DEVICE)
    masked_tensor = torch.from_numpy(masked_query).unsqueeze(0).to(DEVICE)
    mask_device = mask.to(DEVICE)

    with torch.inference_mode():

        original_embedding = autoencoder(query_tensor)

        teacher_query_embedding = F.normalize(
            teacher(masked_tensor), p=2, dim=1
        )

        frame_query_embedding = F.normalize(
            frame_model(original_embedding, mask_device),
            p=2,
            dim=1
        )

        auto_query_embedding = F.normalize(
            original_embedding, p=2, dim=1
        )

    # Teacher and FRAME use the same teacher database.
    teacher_scores = []

    for db_name, db_embedding in teacher_database.items():
        score = torch.sum(
            teacher_query_embedding * db_embedding
        ).item()
        teacher_scores.append((db_name, score))

    frame_scores = []

    for db_name, db_embedding in teacher_database.items():
        score = torch.sum(
            frame_query_embedding * db_embedding
        ).item()
        frame_scores.append((db_name, score))

    # Autoencoder uses its own embedding database.
    auto_scores = []

    for db_name, db_embedding in auto_database.items():
        score = torch.sum(
            auto_query_embedding * db_embedding
        ).item()
        auto_scores.append((db_name, score))

    # Visible-pixel MSE baseline.
    pixel_scores = []

    for db_name, db_image in database_images.items():
        mse = np.mean(
            (query[:, visible] - db_image[:, visible]) ** 2
        )
        pixel_scores.append((db_name, float(mse)))

    # Visible-region SSIM baseline.
    valid = binary_erosion(
        visible,
        structure=np.ones((7, 7), dtype=bool),
        border_value=0
    )

    if valid.sum() == 0:
        raise ValueError("No valid visible SSIM locations.")

    ssim_scores = []

    for db_name, db_image in database_images.items():
        channel_scores = []

        for channel in range(3):
            _, similarity_map = structural_similarity(
                query[channel],
                db_image[channel],
                data_range=10.0,
                win_size=7,
                full=True
            )

            channel_scores.append(
                similarity_map[valid].mean()
            )

        ssim_scores.append(
            (db_name, float(np.mean(channel_scores)))
        )

    # Save complete rankings.
    teacher_top = save_rankings(
        query_name, "Teacher", teacher_scores, "similarity"
    )

    frame_top = save_rankings(
        query_name, "FRAME", frame_scores, "similarity"
    )

    auto_top = save_rankings(
        query_name, "Autoencoder", auto_scores, "similarity"
    )

    pixel_top = save_rankings(
        query_name, "Pixel", pixel_scores, "distance"
    )

    ssim_top = save_rankings(
        query_name, "SSIM", ssim_scores, "similarity"
    )

    top_lists = {
        "Teacher": teacher_top,
        "FRAME": frame_top,
        "Autoencoder": auto_top,
        "Pixel": pixel_top,
        "SSIM": ssim_top
    }

    for method, top_list in top_lists.items():
        query_metrics.append({
            "query": query_name,
            "method": method,
            "top1": top_list[0],
            "top5": "|".join(top_list[:5]),
            "mask_size": mask_side
        })

    print(
        f"Processed {query_index + 1}/{len(test_images)}: "
        f"{query_name}"
    )


# --------------------------------------------------
# Save full rankings and per-query results
# --------------------------------------------------

rankings = pd.DataFrame(ranking_rows)

rankings.to_csv(
    os.path.join(OUTPUT_DIR, "final_test_rankings.csv"),
    index=False
)

metrics = pd.DataFrame(query_metrics)

metrics.to_csv(
    os.path.join(OUTPUT_DIR, "final_test_per_query.csv"),
    index=False
)


# --------------------------------------------------
# Method agreement summary
# --------------------------------------------------

methods = [
    "Teacher",
    "FRAME",
    "Autoencoder",
    "Pixel",
    "SSIM"
]

summary_rows = []

for method in methods:
    method_df = metrics[metrics["method"] == method]

    summary_rows.append({
        "method": method,
        "queries": len(method_df),
        "unique_top1_matches": method_df["top1"].nunique()
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "final_test_method_summary.csv"),
    index=False
)


# --------------------------------------------------
# Pairwise top-1 agreement and top-5 overlap
# --------------------------------------------------

comparison_rows = []

for query_name in test_names:

    query_df = metrics[metrics["query"] == query_name]

    method_data = {}

    for method in methods:
        row = query_df[query_df["method"] == method].iloc[0]

        method_data[method] = {
            "top1": row["top1"],
            "top5": set(row["top5"].split("|"))
        }

    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1:]:

            comparison_rows.append({
                "query": query_name,
                "method_a": method_a,
                "method_b": method_b,
                "top1_agreement": (
                    method_data[method_a]["top1"]
                    == method_data[method_b]["top1"]
                ),
                "top5_overlap": len(
                    method_data[method_a]["top5"]
                    & method_data[method_b]["top5"]
                )
            })

comparisons = pd.DataFrame(comparison_rows)

comparisons.to_csv(
    os.path.join(OUTPUT_DIR, "final_test_pairwise_comparison.csv"),
    index=False
)


# --------------------------------------------------
# Print final report
# --------------------------------------------------

print("\nFINAL TEST METHOD SUMMARY")
print(summary.to_string(index=False))

print("\nPAIRWISE METHOD COMPARISON")

pairwise_summary = comparisons.groupby(
    ["method_a", "method_b"]
).agg(
    top1_agreement=("top1_agreement", "mean"),
    average_top5_overlap=("top5_overlap", "mean")
).reset_index()

print(pairwise_summary.to_string(index=False))

print("\nOUTPUT FILES")
print(os.path.join(OUTPUT_DIR, "final_test_rankings.csv"))
print(os.path.join(OUTPUT_DIR, "final_test_per_query.csv"))
print(os.path.join(OUTPUT_DIR, "final_test_method_summary.csv"))
print(os.path.join(OUTPUT_DIR, "final_test_pairwise_comparison.csv"))

print("\nEXPANDED FINAL TEST EVALUATION COMPLETE")
