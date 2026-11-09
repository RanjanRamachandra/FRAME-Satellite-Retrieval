import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from datetime import datetime
from skimage.metrics import structural_similarity
from scipy.ndimage import binary_erosion

from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel
from satellite_dataset import HimawariDataset

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

EMBEDDING_DIM = 512
EXCLUSION_HOURS = 12
TOP_K = 5

print("Device:", DEVICE)

def get_time(filename):
    value = filename.replace(
        "himawari_asia_", ""
    ).replace(".npz", "")
    return datetime.strptime(value, "%Y%m%d_%H%M")


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

    mask[:, :, top:top+side, left:left+side] = 0

    return mask.to(DEVICE)


# Load checkpoints
original_ckpt = torch.load(
    "checkpoints/original_encoder.pth",
    map_location=DEVICE
)

masked_ckpt = torch.load(
    "checkpoints/best_masked_encoder.pth",
    map_location=DEVICE
)

frame_ckpt = torch.load(
    "checkpoints/best_frame_model.pth",
    map_location=DEVICE
)

original_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
original_encoder.load_state_dict(
    original_ckpt["encoder_state_dict"]
)

masked_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
masked_encoder.load_state_dict(
    masked_ckpt["encoder_state_dict"]
)

frame_model = FRAMEModel(EMBEDDING_DIM).to(DEVICE)
frame_model.load_state_dict(
    frame_ckpt["frame_model_state_dict"]
)

original_encoder.eval()
masked_encoder.eval()
frame_model.eval()

print("All checkpoints loaded.")

# Chronological datasets
train = HimawariDataset(
    split="train", image_size=200
)

dev = HimawariDataset(
    split="test", image_size=200
)

database = [
    (image.numpy(), filename)
    for image, filename in train
]

query_data = list(dev)

print("Database:", len(database))
print("Queries:", len(query_data))

# Precompute full-image database embeddings
db_images = torch.stack([
    torch.from_numpy(image)
    for image, _ in database
]).to(DEVICE)

db_names = [filename for _, filename in database]
db_times = [get_time(name) for name in db_names]

with torch.no_grad():
    db_embeddings = F.normalize(
        masked_encoder(db_images), p=2, dim=1
    )

all_results = []
summary_rows = []

for query_index, (query_tensor, query_name) in enumerate(query_data):

    query = query_tensor.numpy()
    query_time = get_time(query_name)

    eligible_indices = [
        i for i, db_time in enumerate(db_times)
        if abs((query_time - db_time).total_seconds()) / 3600
        > EXCLUSION_HOURS
    ]

    if len(eligible_indices) < TOP_K:
        raise ValueError(
            f"Not enough eligible candidates for {query_name}"
        )

    eligible_names = [
        db_names[i] for i in eligible_indices
    ]

    eligible_images = [
        database[i][0] for i in eligible_indices
    ]

    # Exact evaluation mask
    mask = create_mask(
        200, 200, seed=1000 + query_index
    )

    mask_np = mask.cpu().numpy()[0, 0]
    visible = mask_np.astype(bool)

    # SSIM valid region: complete 7x7 windows only
    valid = binary_erosion(
        visible,
        structure=np.ones((7, 7), dtype=bool),
        border_value=0
    )

    # Pixel MSE on visible pixels
    pixel_scores = []

    # SSIM on visible pixels
    ssim_scores = []

    for db_image in eligible_images:

        difference = query[:, visible] - db_image[:, visible]
        mse = float(np.mean(difference ** 2))
        pixel_scores.append(mse)

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
                float(similarity_map[valid].mean())
            )

        ssim_scores.append(float(np.mean(channel_scores)))

    # Learned query embeddings
    images = query_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        masked_image = images * mask

        original_embedding = original_encoder(images)

        teacher_embedding = masked_encoder(masked_image)

        frame_embedding = frame_model(
            original_embedding, mask
        )

        teacher_query = F.normalize(
            teacher_embedding, p=2, dim=1
        )

        frame_query = F.normalize(
            frame_embedding, p=2, dim=1
        )

        teacher_scores = (
            teacher_query @ db_embeddings[eligible_indices].T
        ).squeeze(0).cpu().numpy()

        frame_scores = (
            frame_query @ db_embeddings[eligible_indices].T
        ).squeeze(0).cpu().numpy()

    score_sets = {
        "Pixel": (pixel_scores, False),
        "SSIM": (ssim_scores, True),
        "Teacher": (teacher_scores, True),
        "FRAME": (frame_scores, True)
    }

    print(
        f"\nQuery {query_index+1}/11: {query_name}"
    )
    print("Eligible database candidates:", len(eligible_indices))

    query_top1 = {}

    for method, (scores, descending) in score_sets.items():

        order = np.argsort(scores)

        if descending:
            order = order[::-1]

        query_top1[method] = eligible_names[order[0]]

        for rank, candidate_index in enumerate(
            order[:TOP_K], start=1
        ):
            all_results.append({
                "query": query_name,
                "method": method,
                "rank": rank,
                "retrieved_image": eligible_names[candidate_index],
                "score": float(scores[candidate_index]),
                "eligible_candidates": len(eligible_indices)
            })

        print(
            f"{method:8s} Top-1: {query_top1[method]}"
        )

    summary_rows.append({
        "query": query_name,
        **{
            method + "_top1": image
            for method, image in query_top1.items()
        }
    })

# Save complete rankings
os.makedirs("results", exist_ok=True)

pd.DataFrame(all_results).to_csv(
    "results/temporal_exclusion_12h_rankings.csv",
    index=False
)

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    "results/temporal_exclusion_12h_top1.csv",
    index=False
)

print("\nTOP-1 AGREEMENT WITH EACH OTHER METHOD")

for method in ["Pixel", "SSIM", "Teacher", "FRAME"]:
    matches = (
        summary[method + "_top1"] ==
        summary["Pixel_top1"]
    ).sum()

    print(
        f"{method} vs Pixel: {matches}/11 "
        f"({matches/11:.2%})"
    )

print("\nAVERAGE TOP-5 OVERLAP")

results_df = pd.DataFrame(all_results)

for method_a, method_b in [
    ("Pixel", "SSIM"),
    ("Pixel", "Teacher"),
    ("Pixel", "FRAME"),
    ("SSIM", "Teacher"),
    ("SSIM", "FRAME"),
    ("Teacher", "FRAME")
]:
    overlaps = []

    for query_name in summary["query"]:
        a = set(
            results_df[
                (results_df["query"] == query_name) &
                (results_df["method"] == method_a)
            ]["retrieved_image"]
        )

        b = set(
            results_df[
                (results_df["query"] == query_name) &
                (results_df["method"] == method_b)
            ]["retrieved_image"]
        )

        overlaps.append(len(a & b))

    print(
        f"{method_a} vs {method_b}: "
        f"{np.mean(overlaps):.2f}/5"
    )

print("\nSaved:")
print("results/temporal_exclusion_12h_rankings.csv")
print("results/temporal_exclusion_12h_top1.csv")
print("\nTEMPORAL EXCLUSION EVALUATION COMPLETE")
