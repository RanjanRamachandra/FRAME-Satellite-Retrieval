import csv
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from expanded_satellite_dataset import create_expanded_dataloaders
from models.se_res_autoencoder import SEResAutoencoder


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = ROOT / "checkpoints" / "expanded_2022"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = CHECKPOINT_DIR / "best_masked_encoder.pth"
LAST_PATH = CHECKPOINT_DIR / "last_masked_encoder.pth"
HISTORY_PATH = CHECKPOINT_DIR / "masked_encoder_training_history.csv"

EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 0.001
PATIENCE = 7
SEED = 42

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def random_square_mask(batch_size, height, width, device):
    masks = torch.ones(
        batch_size, 1, height, width,
        device=device
    )

    for i in range(batch_size):
        side = torch.randint(40, 101, (1,)).item()
        top = torch.randint(0, height - side + 1, (1,)).item()
        left = torch.randint(0, width - side + 1, (1,)).item()

        masks[i, :, top:top + side, left:left + side] = 0

    return masks


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_images = 0

    for images, filenames in loader:
        images = images.to(DEVICE)

        mask = random_square_mask(
            images.size(0),
            images.size(2),
            images.size(3),
            DEVICE
        )

        masked_images = images * mask

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            reconstruction, embedding = model(masked_images)

            # Reconstruct the complete original image.
            loss = criterion(reconstruction, images)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

    return total_loss / total_images


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    encoder_state = {
        key: value.detach().cpu()
        for key, value in model.encoder.state_dict().items()
    }

    checkpoint = {
        "encoder_state_dict": encoder_state,
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": val_loss,
        "embedding_dim": 512,
        "seed": SEED,
        "training_observations": 112,
        "validation_observations": 56
    }

    torch.save(checkpoint, path)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if DEVICE.type == "mps":
        torch.mps.manual_seed(SEED)

    print("====================================")
    print("EXPANDED MASKED TEACHER TRAINING")
    print("====================================")
    print("Device:", DEVICE)
    print("Seed:", SEED)
    print("Starting from random weights")
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    loaders = create_expanded_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=200
    )

    train_loader = loaders["train"]
    validation_loader = loaders["validation"]

    print("Training observations:", len(train_loader.dataset))
    print("Validation observations:", len(validation_loader.dataset))
    print("Final test observations: reserved, not loaded")

    # Independent teacher: do not load autoencoder weights.
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
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss = run_epoch(
            model,
            validation_loader,
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
            LAST_PATH
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                BEST_PATH
            )

            print("Best masked teacher checkpoint saved.")
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
    print("EXPANDED MASKED TEACHER COMPLETE")
    print("====================================")
    print("Best validation loss:", best_val_loss)
    print("Best checkpoint:", BEST_PATH)
    print("Last checkpoint:", LAST_PATH)
    print("Training history:", HISTORY_PATH)


if __name__ == "__main__":
    main()
