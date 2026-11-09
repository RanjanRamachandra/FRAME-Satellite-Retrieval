import os
import torch
import torch.nn.functional as F
import pandas as pd

from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel
from satellite_dataset import create_dataloaders


# Configuration
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

EMBEDDING_DIM = 512
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 0.001

os.makedirs("checkpoints", exist_ok=True)
print("FRAME V2 training device:", DEVICE)


# Load original encoder
original_checkpoint = torch.load(
    "checkpoints/original_encoder.pth",
    map_location=DEVICE
)

original_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
original_encoder.load_state_dict(
    original_checkpoint["encoder_state_dict"]
)


# Load V2 aligned masked teacher
teacher_checkpoint = torch.load(
    "checkpoints/best_masked_encoder_aligned.pth",
    map_location=DEVICE
)

masked_encoder = SEResEncoder(EMBEDDING_DIM).to(DEVICE)
masked_encoder.load_state_dict(
    teacher_checkpoint["encoder_state_dict"]
)


# Freeze both encoders
original_encoder.eval()
masked_encoder.eval()

for parameter in original_encoder.parameters():
    parameter.requires_grad = False

for parameter in masked_encoder.parameters():
    parameter.requires_grad = False

print("Original encoder loaded and frozen.")
print("V2 aligned teacher loaded and frozen.")


# Create a fresh FRAME model
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


# Random square masks
def random_square_mask(batch_size, height, width, device):

    mask = torch.ones(
        batch_size, 1, height, width,
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

        mask[i, :, top:top + side, left:left + side] = 0

    return mask


# Load dataset
train_loader, test_loader = create_dataloaders(
    batch_size=BATCH_SIZE,
    image_size=200
)

print("Training observations:", len(train_loader.dataset))
print("Validation observations:", len(test_loader.dataset))


# Training and validation
def run_epoch(loader, training=True):

    if training:
        frame_model.train()
    else:
        frame_model.eval()

    total_loss = 0.0
    total_samples = 0

    for images, filenames in loader:

        images = images.to(DEVICE)

        batch_size = images.size(0)
        height = images.size(2)
        width = images.size(3)

        mask = random_square_mask(
            batch_size, height, width, DEVICE
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

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# Main training loop
best_validation_loss = float("inf")
patience = 8
epochs_without_improvement = 0
history = []

for epoch in range(1, EPOCHS + 1):

    train_loss = run_epoch(train_loader, training=True)

    validation_loss = run_epoch(
        test_loader,
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

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch,
            "frame_model_state_dict": frame_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "embedding_dim": EMBEDDING_DIM
        }, "checkpoints/best_frame_model_v2.pth")

    else:
        epochs_without_improvement += 1

    torch.save({
        "epoch": epoch,
        "frame_model_state_dict": frame_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "embedding_dim": EMBEDDING_DIM
    }, "checkpoints/last_frame_model_v2.pth")

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break


# Save training history
history_df = pd.DataFrame(history)

history_df.to_csv(
    "checkpoints/frame_training_history_v2.csv",
    index=False
)

print("\nFRAME V2 TRAINING COMPLETE")
print("Best validation loss:", best_validation_loss)
print("Best checkpoint: checkpoints/best_frame_model_v2.pth")
print("Last checkpoint: checkpoints/last_frame_model_v2.pth")
print("Training history: checkpoints/frame_training_history_v2.csv")
