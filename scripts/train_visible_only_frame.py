import os
import random
import torch
import torch.nn.functional as F
import pandas as pd

from models.se_res_autoencoder import SEResEncoder
from models.visible_only_frame_model import VisibleOnlyFRAMEModel
from satellite_dataset import create_dataloaders

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

EMBEDDING_DIM = 512
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 20261110

os.makedirs("checkpoints", exist_ok=True)

torch.manual_seed(SEED)
random.seed(SEED)

print("Visible-only training device:", DEVICE)

# Load frozen original encoder
original_checkpoint = torch.load(
    "checkpoints/original_encoder.pth",
    map_location=DEVICE
)

original_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
original_encoder.load_state_dict(
    original_checkpoint["encoder_state_dict"]
)
original_encoder.eval()

for p in original_encoder.parameters():
    p.requires_grad = False

# Load frozen Teacher encoder
teacher_checkpoint = torch.load(
    "checkpoints/best_masked_encoder.pth",
    map_location=DEVICE
)

teacher_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
teacher_encoder.load_state_dict(
    teacher_checkpoint["encoder_state_dict"]
)
teacher_encoder.eval()

for p in teacher_encoder.parameters():
    p.requires_grad = False

# Initialize model
model = VisibleOnlyFRAMEModel(EMBEDDING_DIM).to(DEVICE)

# Initialize visible encoder from original encoder
model.visible_encoder.load_state_dict(
    original_encoder.state_dict()
)

# Reuse trained FRAME mask encoder
frame_checkpoint = torch.load(
    "checkpoints/best_frame_model.pth",
    map_location=DEVICE
)

frame_state = frame_checkpoint["frame_model_state_dict"]

mask_state = {
    key.replace("mask_encoder.", "", 1): value
    for key, value in frame_state.items()
    if key.startswith("mask_encoder.")
}

model.mask_encoder.load_state_dict(mask_state)

for p in model.mask_encoder.parameters():
    p.requires_grad = False

print("Original encoder loaded and frozen.")
print("Teacher encoder loaded and frozen.")
print("Mask encoder loaded from FRAME and frozen.")

optimizer = torch.optim.Adam(
    [
        {"params": model.visible_encoder.parameters()},
        {"params": model.projection_head.parameters()}
    ],
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

train_loader, test_loader = create_dataloaders(
    batch_size=BATCH_SIZE,
    image_size=200
)

print("Training observations:", len(train_loader.dataset))
print("Validation observations:", len(test_loader.dataset))


def random_square_mask(batch_size, height, width, device):

    masks = torch.ones(
        batch_size, 1, height, width, device=device
    )

    for i in range(batch_size):

        side = torch.randint(40, 101, (1,)).item()
        top = torch.randint(0, height - side + 1, (1,)).item()
        left = torch.randint(0, width - side + 1, (1,)).item()

        masks[i, :, top:top + side, left:left + side] = 0

    return masks


# Fixed validation masks
validation_masks = {}

for filename in test_loader.dataset.split_df["filename"].tolist():

    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        SEED + sum(ord(c) for c in filename)
    )

    side = torch.randint(
        40, 101, (1,), generator=generator
    ).item()

    top = torch.randint(
        0, 200 - side + 1, (1,), generator=generator
    ).item()

    left = torch.randint(
        0, 200 - side + 1, (1,), generator=generator
    ).item()

    mask = torch.ones(1, 1, 200, 200)
    mask[:, :, top:top + side, left:left + side] = 0

    validation_masks[filename] = mask


def run_epoch(loader, training=True):

    model.train(training)

    original_encoder.eval()
    teacher_encoder.eval()
    model.mask_encoder.eval()

    total_loss = 0.0
    total_samples = 0

    for images, filenames in loader:

        images = images.to(DEVICE)
        batch_size = images.size(0)

        if training:

            mask = random_square_mask(
                batch_size,
                images.size(2),
                images.size(3),
                DEVICE
            )

        else:

            mask = torch.cat(
                [validation_masks[name] for name in filenames],
                dim=0
            ).to(DEVICE)

        masked_images = images * mask

        with torch.no_grad():

            target_embedding = teacher_encoder(
                masked_images
            )

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            predicted_embedding = model(
                images, mask
            )

            loss = 1 - F.cosine_similarity(
                predicted_embedding,
                target_embedding,
                dim=1
            ).mean()

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# Training loop
best_validation_loss = float("inf")
patience = 8
epochs_without_improvement = 0
history = []

for epoch in range(1, EPOCHS + 1):

    train_loss = run_epoch(train_loader, training=True)
    validation_loss = run_epoch(test_loader, training=False)

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

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch,
            "visible_only_frame_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "embedding_dim": EMBEDDING_DIM
        }, "checkpoints/best_visible_only_frame_model.pth")

        print("Best VisibleOnlyFRAME checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

pd.DataFrame(history).to_csv(
    "checkpoints/visible_only_frame_training_history.csv",
    index=False
)

print("\nVISIBLE-ONLY FRAME TRAINING COMPLETE")
print("Best validation loss:", best_validation_loss)
print("Checkpoint: checkpoints/best_visible_only_frame_model.pth")
print("History: checkpoints/visible_only_frame_training_history.csv")
