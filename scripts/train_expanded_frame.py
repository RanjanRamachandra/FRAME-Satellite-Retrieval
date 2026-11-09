import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel
from expanded_satellite_dataset import ExpandedHimawariDataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
EMBEDDING_DIM = 512
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 8

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINT_DIR = "checkpoints/expanded_2022"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

print("EXPANDED FRAME TRAINING")
print("Device:", DEVICE)
print("Seed:", SEED)


# --------------------------------------------------
# Load expanded full-image encoder
# --------------------------------------------------

autoencoder_checkpoint = torch.load(
    os.path.join(CHECKPOINT_DIR, "best_autoencoder.pth"),
    map_location=DEVICE
)

original_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)

autoencoder_encoder_state = {
    key[len("encoder."):]: value
    for key, value in
    autoencoder_checkpoint["model_state_dict"].items()
    if key.startswith("encoder.")
}

original_encoder.load_state_dict(
    autoencoder_encoder_state,
    strict=True
)


# --------------------------------------------------
# Load expanded masked teacher encoder
# --------------------------------------------------

masked_checkpoint = torch.load(
    os.path.join(CHECKPOINT_DIR, "best_masked_encoder.pth"),
    map_location=DEVICE
)

masked_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)

masked_encoder.load_state_dict(
    masked_checkpoint["encoder_state_dict"],
    strict=True
)


# --------------------------------------------------
# Freeze pretrained encoders
# --------------------------------------------------

original_encoder.eval()
masked_encoder.eval()

for parameter in original_encoder.parameters():
    parameter.requires_grad = False

for parameter in masked_encoder.parameters():
    parameter.requires_grad = False

print("Expanded full-image encoder loaded and frozen.")
print("Expanded masked teacher loaded and frozen.")


# --------------------------------------------------
# Load FRAME model
# --------------------------------------------------

frame_model = FRAMEModel(EMBEDDING_DIM).to(DEVICE)

optimizer = torch.optim.Adam(
    frame_model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# --------------------------------------------------
# Load ONLY training and validation datasets
# --------------------------------------------------

train_dataset = ExpandedHimawariDataset(
    split="train",
    image_size=200
)

validation_dataset = ExpandedHimawariDataset(
    split="validation",
    image_size=200
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training observations:", len(train_dataset))
print("Validation observations:", len(validation_dataset))
print("Final test dataset: NOT LOADED")


# --------------------------------------------------
# Generate random square masks
# --------------------------------------------------

def random_square_mask(batch_size, height, width, device):

    mask = torch.ones(
        batch_size,
        1,
        height,
        width,
        device=device
    )

    for i in range(batch_size):

        side = torch.randint(40, 101, (1,)).item()

        top = torch.randint(
            0, height - side + 1, (1,)
        ).item()

        left = torch.randint(
            0, width - side + 1, (1,)
        ).item()

        mask[
            i, :, top:top + side, left:left + side
        ] = 0

    return mask


# --------------------------------------------------
# Training and validation
# --------------------------------------------------

def run_epoch(loader, training=True):

    if training:
        frame_model.train()
    else:
        frame_model.eval()

    original_encoder.eval()
    masked_encoder.eval()

    total_loss = 0.0
    total_samples = 0

    for images, filenames in loader:

        images = images.to(DEVICE)

        batch_size = images.size(0)
        height = images.size(2)
        width = images.size(3)

        mask = random_square_mask(
            batch_size,
            height,
            width,
            DEVICE
        )

        masked_images = images * mask

        with torch.no_grad():

            original_embedding = original_encoder(images)

            target_embedding = masked_encoder(masked_images)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            predicted_embedding = frame_model(
                original_embedding,
                mask
            )

            loss = 1 - F.cosine_similarity(
                predicted_embedding,
                target_embedding,
                dim=1
            ).mean()

            if training:
                loss.backward()
                optimizer.step()

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite FRAME loss detected.")

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# --------------------------------------------------
# Main training loop
# --------------------------------------------------

best_validation_loss = float("inf")
epochs_without_improvement = 0
history = []

best_checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "best_frame_model.pth"
)

last_checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "last_frame_model.pth"
)

history_path = os.path.join(
    CHECKPOINT_DIR,
    "frame_training_history.csv"
)

for epoch in range(1, EPOCHS + 1):

    train_loss = run_epoch(
        train_loader,
        training=True
    )

    validation_loss = run_epoch(
        validation_loader,
        training=False
    )

    scheduler.step(validation_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Validation Loss: {validation_loss:.6f} "
        f"LR: {current_lr:.6f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "learning_rate": current_lr
    })

    checkpoint = {
        "epoch": epoch,
        "frame_model_state_dict": frame_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "embedding_dim": EMBEDDING_DIM,
        "seed": SEED,
        "training_observations": len(train_dataset),
        "validation_observations": len(validation_dataset)
    }

    torch.save(checkpoint, last_checkpoint_path)

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss
        epochs_without_improvement = 0

        torch.save(checkpoint, best_checkpoint_path)

        print("  Best expanded FRAME checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("Early stopping triggered.")
        break


# --------------------------------------------------
# Save final results
# --------------------------------------------------

pd.DataFrame(history).to_csv(
    history_path,
    index=False
)

print("\n====================================")
print("EXPANDED FRAME TRAINING COMPLETE")
print("====================================")
print("Best validation loss:", best_validation_loss)
print("Best checkpoint:", best_checkpoint_path)
print("Last checkpoint:", last_checkpoint_path)
print("Training history:", history_path)
print("Final test dataset was not loaded or used.")
