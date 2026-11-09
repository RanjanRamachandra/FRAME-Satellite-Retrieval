
import csv
from pathlib import Path

import torch
import torch.nn as nn

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResAutoencoder


# -----------------------------
# Configuration
# -----------------------------

EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 0.001
PATIENCE = 7

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_autoencoder.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_autoencoder.pth"
HISTORY_PATH = CHECKPOINT_DIR / "training_history.csv"


def train_one_epoch(model, loader, optimizer, criterion):

    model.train()
    total_loss = 0.0
    total_images = 0

    for images, filenames in loader:

        images = images.to(DEVICE)

        optimizer.zero_grad()

        reconstruction, embedding = model(images)

        loss = criterion(reconstruction, images)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_images += batch_size

    return total_loss / total_images


def validate(model, loader, criterion):

    model.eval()
    total_loss = 0.0
    total_images = 0

    with torch.no_grad():

        for images, filenames in loader:

            images = images.to(DEVICE)

            reconstruction, embedding = model(images)

            loss = criterion(reconstruction, images)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_images += batch_size

    return total_loss / total_images


def save_checkpoint(model, optimizer, epoch, val_loss, path):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": val_loss,
        "embedding_dim": 512
    }

    torch.save(checkpoint, path)


def main():

    print("====================================")
    print("SE-RES AUTOENCODER TRAINING")
    print("====================================")

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    train_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=200
    )

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(test_loader.dataset))

    model = SEResAutoencoder(
        embedding_dim=512
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss = validate(
            model,
            test_loader,
            criterion
        )

        if DEVICE.type == "mps":
            torch.mps.synchronize()

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append([
            epoch,
            train_loss,
            val_loss,
            current_lr
        ])

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.6f}"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            LAST_MODEL_PATH
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                BEST_MODEL_PATH
            )

            print("  Best model checkpoint saved.")

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print("\nEarly stopping triggered.")
            break

    with open(HISTORY_PATH, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "training_loss",
            "validation_loss",
            "learning_rate"
        ])

        writer.writerows(history)

    print("\n====================================")
    print("TRAINING COMPLETE")
    print("====================================")

    print("Best validation loss:", best_val_loss)
    print("Best model:", BEST_MODEL_PATH)
    print("Last model:", LAST_MODEL_PATH)
    print("Training history:", HISTORY_PATH)


if __name__ == "__main__":
    main()

