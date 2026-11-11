"""
Mask ablation experiments for FRAME satellite image retrieval.

Investigates how mask shape and size affect FRAME retrieval behaviour.

ABLATIONS
---------
1. Mask shape:
   - Square (baseline)
   - Circle
   - Rectangle (2:1 aspect ratio)

2. Mask size:
   - Small  : side/diameter 20-40 px  (10-20% of image width)
   - Medium : side/diameter 40-80 px  (20-40%) — baseline range
   - Large  : side/diameter 80-120 px (40-60%)

Each combination trains a separate FRAME model using the same frozen
autoencoder and teacher encoders as the baseline experiment.

USAGE
-----
python scripts/train_ablation_masks.py

OUTPUT
------
checkpoints/ablation_masks/
    frame_<shape>_<size>.pth    — Best checkpoint per configuration

results/ablation_masks/
    ablation_training_history.csv  — Loss curves per configuration
    ablation_summary.csv           — Best validation loss per configuration

INTERPRETATION NOTE
-------------------
Lower FRAME validation loss means better imitation of the teacher embedding.
It does NOT directly mean better retrieval quality.
Run evaluate_ablation_results.py after this script to compare retrieval behaviour.
"""

import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from expanded_satellite_dataset import ExpandedHimawariDataset
from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

AUTO_CHECKPOINT = "checkpoints/expanded_2022/best_autoencoder.pth"
TEACHER_CHECKPOINT = "checkpoints/expanded_2022/best_masked_encoder.pth"
OUT_CHECKPOINTS = Path("checkpoints/ablation_masks")
OUT_RESULTS = Path("results/ablation_masks")

OUT_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
OUT_RESULTS.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 200
EMBEDDING_DIM = 512
BATCH_SIZE = 2
LR = 0.001
MAX_EPOCHS = 50
PATIENCE = 8
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Ablation grid
MASK_SHAPES = ["square", "circle", "rectangle"]
MASK_SIZES = {
    "small":  (20, 40),   # (min_side, max_side) in pixels
    "medium": (40, 80),   # baseline range
    "large":  (80, 120),
}

print("FRAME MASK ABLATION EXPERIMENTS")
print(f"Device: {DEVICE}")
print(f"Shapes: {MASK_SHAPES}")
print(f"Sizes : {list(MASK_SIZES.keys())}")
print(f"Total configurations: {len(MASK_SHAPES) * len(MASK_SIZES)}")
print()

# --------------------------------------------------
# Load frozen encoders
# --------------------------------------------------

auto_checkpoint = torch.load(
    AUTO_CHECKPOINT, map_location="cpu", weights_only=False
)
teacher_checkpoint = torch.load(
    TEACHER_CHECKPOINT, map_location="cpu", weights_only=False
)

auto_encoder = SEResEncoder(EMBEDDING_DIM)
auto_state = {
    key.replace("encoder.", "", 1): value
    for key, value in auto_checkpoint["model_state_dict"].items()
    if key.startswith("encoder.")
}
auto_encoder.load_state_dict(auto_state)
auto_encoder = auto_encoder.to(DEVICE).eval()
for p in auto_encoder.parameters():
    p.requires_grad = False

teacher_encoder = SEResEncoder(EMBEDDING_DIM)
teacher_encoder.load_state_dict(teacher_checkpoint["encoder_state_dict"])
teacher_encoder = teacher_encoder.to(DEVICE).eval()
for p in teacher_encoder.parameters():
    p.requires_grad = False

print("Frozen encoders loaded.")

# --------------------------------------------------
# Mask generation functions
# --------------------------------------------------

def make_square_mask(image_size, min_side, max_side, generator=None):
    """Binary mask with a square hidden region (0 = hidden)."""
    mask = torch.ones(1, 1, image_size, image_size)
    if generator is None:
        side = random.randint(min_side, max_side)
        top  = random.randint(0, image_size - side)
        left = random.randint(0, image_size - side)
    else:
        side = torch.randint(min_side, max_side + 1, (1,), generator=generator).item()
        top  = torch.randint(0, image_size - side + 1, (1,), generator=generator).item()
        left = torch.randint(0, image_size - side + 1, (1,), generator=generator).item()
    mask[:, :, top:top + side, left:left + side] = 0
    return mask


def make_circle_mask(image_size, min_side, max_side):
    """Binary mask with a circular hidden region (0 = hidden)."""
    mask = torch.ones(1, 1, image_size, image_size)
    radius = random.randint(min_side // 2, max_side // 2)
    cy = random.randint(radius, image_size - radius)
    cx = random.randint(radius, image_size - radius)
    y_grid = torch.arange(image_size).unsqueeze(1).float()
    x_grid = torch.arange(image_size).unsqueeze(0).float()
    dist = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2).sqrt()
    mask[0, 0][dist <= radius] = 0
    return mask


def make_rectangle_mask(image_size, min_side, max_side):
    """Binary mask with a 2:1 rectangle hidden region (0 = hidden)."""
    mask = torch.ones(1, 1, image_size, image_size)
    short = random.randint(min_side, max_side)
    long  = min(short * 2, image_size)
    # Randomly orient: tall or wide
    if random.random() < 0.5:
        h, w = long, short
    else:
        h, w = short, long
    top  = random.randint(0, max(1, image_size - h))
    left = random.randint(0, max(1, image_size - w))
    mask[:, :, top:top + h, left:left + w] = 0
    return mask


MASK_FUNCTIONS = {
    "square":    make_square_mask,
    "circle":    make_circle_mask,
    "rectangle": make_rectangle_mask,
}


def apply_mask(image_tensor, mask):
    """Zero out the masked region in the image tensor."""
    return image_tensor * mask.to(image_tensor.device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = ExpandedHimawariDataset(split="train", image_size=IMAGE_SIZE)
val_dataset   = ExpandedHimawariDataset(split="val",   image_size=IMAGE_SIZE)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
print()

# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_frame_ablation(shape, size_name, min_side, max_side):
    label = f"{shape}_{size_name}"
    print(f"Training: {label} | side range [{min_side}, {max_side}]")

    model = FRAMEModel(EMBEDDING_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    mask_fn = MASK_FUNCTIONS[shape]
    best_val_loss = float("inf")
    patience_count = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        # --- Train ---
        model.train()
        train_losses = []

        for images, _ in train_loader:
            images = images.to(DEVICE)

            masks = torch.cat([
                mask_fn(IMAGE_SIZE, min_side, max_side)
                for _ in range(len(images))
            ]).to(DEVICE)

            masked_images = apply_mask(images, masks)

            with torch.no_grad():
                z_full    = auto_encoder(images)
                z_teacher = teacher_encoder(masked_images)
                z_teacher = F.normalize(z_teacher, p=2, dim=1)

            z_pred = model(z_full, masks)
            z_pred = F.normalize(z_pred, p=2, dim=1)

            loss = (1 - (z_pred * z_teacher).sum(dim=1)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)

                masks = torch.cat([
                    mask_fn(IMAGE_SIZE, min_side, max_side)
                    for _ in range(len(images))
                ]).to(DEVICE)

                masked_images = apply_mask(images, masks)

                z_full    = auto_encoder(images)
                z_teacher = F.normalize(teacher_encoder(masked_images), p=2, dim=1)
                z_pred    = F.normalize(model(z_full, masks), p=2, dim=1)

                loss = (1 - (z_pred * z_teacher).sum(dim=1)).mean()
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step(val_loss)

        history.append({
            "configuration": label,
            "epoch": epoch,
            "train_loss": round(train_loss, 8),
            "val_loss": round(val_loss, 8),
        })

        print(
            f"  Epoch {epoch:3d} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(
                {
                    "configuration": label,
                    "shape": shape,
                    "size_name": size_name,
                    "min_side": min_side,
                    "max_side": max_side,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "frame_model_state_dict": model.state_dict(),
                },
                OUT_CHECKPOINTS / f"frame_{label}.pth"
            )
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  Best val loss: {best_val_loss:.8f}")
    print()
    return history, best_val_loss


# --------------------------------------------------
# Run all ablations
# --------------------------------------------------

all_history = []
summary_rows = []

for shape in MASK_SHAPES:
    for size_name, (min_side, max_side) in MASK_SIZES.items():
        history, best_val = train_frame_ablation(
            shape, size_name, min_side, max_side
        )
        all_history.extend(history)
        summary_rows.append({
            "configuration": f"{shape}_{size_name}",
            "shape": shape,
            "size": size_name,
            "min_side_px": min_side,
            "max_side_px": max_side,
            "best_val_loss": round(best_val, 8),
        })

# --------------------------------------------------
# Save results
# --------------------------------------------------

history_df = pd.DataFrame(all_history)
history_df.to_csv(OUT_RESULTS / "ablation_training_history.csv", index=False)

summary_df = pd.DataFrame(summary_rows).sort_values("best_val_loss")
summary_df.to_csv(OUT_RESULTS / "ablation_summary.csv", index=False)

print("ABLATION SUMMARY (sorted by best val loss)")
print("=" * 55)
print(summary_df.to_string(index=False))

print("\nOUTPUT FILES")
print(f"  {OUT_CHECKPOINTS}")
print(f"  {OUT_RESULTS / 'ablation_training_history.csv'}")
print(f"  {OUT_RESULTS / 'ablation_summary.csv'}")

print("\nNEXT STEP")
print("  Run: python scripts/evaluate_ablation_results.py")
print("  to compare retrieval behaviour across configurations.")
print()
print("REMINDER")
print("  Lower val loss = better teacher imitation.")
print("  It does NOT directly mean better retrieval quality.")
print("  Retrieval comparison requires evaluate_ablation_results.py")
