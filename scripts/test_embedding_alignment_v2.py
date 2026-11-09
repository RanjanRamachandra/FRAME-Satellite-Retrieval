import os
import csv
import random
import torch
import torch.nn.functional as F

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

EMBEDDING_DIM = 512

print("Using device:", DEVICE)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

_, test_loader = create_dataloaders(batch_size=2, image_size=200)

print("Test observations:", len(test_loader.dataset))


# --------------------------------------------------
# Load original encoder
# --------------------------------------------------

original_checkpoint = torch.load(
    "checkpoints/original_encoder.pth",
    map_location="cpu"
)

original_encoder = SEResEncoder(embedding_dim=EMBEDDING_DIM)

original_encoder.load_state_dict(
    original_checkpoint["encoder_state_dict"]
)

original_encoder.to(DEVICE)
original_encoder.eval()


# --------------------------------------------------
# Load masked teacher encoder
# --------------------------------------------------

teacher_checkpoint = torch.load(
    "checkpoints/best_masked_encoder_aligned.pth",
    map_location="cpu"
)

masked_encoder = SEResEncoder(embedding_dim=EMBEDDING_DIM)

masked_encoder.load_state_dict(
    teacher_checkpoint["encoder_state_dict"]
)

masked_encoder.to(DEVICE)
masked_encoder.eval()


# --------------------------------------------------
# Load FRAME model
# --------------------------------------------------

frame_checkpoint = torch.load(
    "checkpoints/best_frame_model_v2.pth",
    map_location="cpu"
)

frame_model = FRAMEModel(embedding_dim=EMBEDDING_DIM)

frame_model.load_state_dict(
    frame_checkpoint["frame_model_state_dict"]
)

frame_model.to(DEVICE)
frame_model.eval()


# --------------------------------------------------
# Generate random square masks
# --------------------------------------------------

def create_masks(batch_size, height, width):

    masks = torch.ones(batch_size, 1, height, width)

    for i in range(batch_size):

        side = random.randint(40, 100)

        top = random.randint(0, height - side)
        left = random.randint(0, width - side)

        masks[i, :, top:top + side, left:left + side] = 0

    return masks


# --------------------------------------------------
# Evaluate embedding alignment
# --------------------------------------------------

results = []

with torch.no_grad():

    for images, filenames in test_loader:

        images = images.to(DEVICE)

        batch_size, channels, height, width = images.shape

        masks = create_masks(
            batch_size, height, width
        ).to(DEVICE)

        masked_images = images * masks

        # Original encoder on full image
        original_embeddings = original_encoder(images)

        # Teacher on full image
        teacher_full_embeddings = masked_encoder(images)

        # Teacher on masked image
        teacher_masked_embeddings = masked_encoder(masked_images)
    
        similarity_teacher_full_masked = F.cosine_similarity(
            teacher_full_embeddings,
            teacher_masked_embeddings,
            dim=1
        )

        # FRAME prediction
        frame_embeddings = frame_model(
            original_embeddings,
            masks
        )

        # Calculate cosine similarities
        similarity_original_teacher_full = F.cosine_similarity(
            original_embeddings,
            teacher_full_embeddings,
            dim=1
        )

        similarity_original_teacher_masked = F.cosine_similarity(
            original_embeddings,
            teacher_masked_embeddings,
            dim=1
        )

        similarity_frame_teacher = F.cosine_similarity(
            frame_embeddings,
            teacher_masked_embeddings,
            dim=1
        )

        for i in range(batch_size):

            results.append({
                "filename": filenames[i],

                "original_vs_teacher_full":
                    similarity_original_teacher_full[i].item(),

                "original_vs_teacher_masked":
                    similarity_original_teacher_masked[i].item(),

                "frame_vs_teacher_masked":
                    similarity_frame_teacher[i].item(),
                "teacher_full_vs_masked":
                    similarity_teacher_full_masked[i].item()
            })


# --------------------------------------------------
# Save results
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

output_file = "results/embedding_alignment_v2.csv"

with open(output_file, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\n======================================")
print("EMBEDDING ALIGNMENT VALIDATION")
print("======================================")

metrics = [
    "original_vs_teacher_full",
    "original_vs_teacher_masked",
    "frame_vs_teacher_masked",
    "teacher_full_vs_masked",
]

for metric in metrics:

    values = [row[metric] for row in results]

    average = sum(values) / len(values)

    print(
        metric,
        ":",
        round(average, 6)
    )

print("\nObservations evaluated:", len(results))
print("Results saved:", output_file)

print("\nEMBEDDING ALIGNMENT TEST COMPLETE")
