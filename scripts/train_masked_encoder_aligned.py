import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResAutoencoder
from models.se_res_autoencoder import SEResEncoder


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"

ORIGINAL_PATH = CHECKPOINT_DIR / "original_encoder.pth"

BEST_PATH = CHECKPOINT_DIR / "best_masked_encoder_aligned.pth"
LAST_PATH = CHECKPOINT_DIR / "last_masked_encoder_aligned.pth"

EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 0.001
ALIGNMENT_WEIGHT = 0.1
PATIENCE = 7

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

        top = torch.randint(
            0, height - side + 1, (1,)
        ).item()

        left = torch.randint(
            0, width - side + 1, (1,)
        ).item()

        masks[i, :, top:top + side, left:left + side] = 0

    return masks


def run_epoch(
    model,
    original_encoder,
    loader,
    reconstruction_criterion,
    optimizer=None
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    original_encoder.eval()

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_alignment_loss = 0.0
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

            # Frozen original encoder provides reference embeddings
            with torch.no_grad():
                reference_embeddings = original_encoder(images)

            # Masked teacher processes the masked image
            reconstruction, teacher_embeddings = model(masked_images)

            # Reconstruct the complete original image
            reconstruction_loss = reconstruction_criterion(
                reconstruction, images
            )

            # Preserve alignment with original embedding space
            alignment_loss = (
                1.0 - F.cosine_similarity(
                    teacher_embeddings,
                    reference_embeddings,
                    dim=1
                ).mean()
            )

            loss = (
                reconstruction_loss
                + ALIGNMENT_WEIGHT * alignment_loss
            )

            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_reconstruction_loss += (
            reconstruction_loss.item() * batch_size
        )
        total_alignment_loss += (
            alignment_loss.item() * batch_size
        )

        total_images += batch_size

    return (
        total_loss / total_images,
        total_reconstruction_loss / total_images,
        total_alignment_loss / total_images
    )


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
        "alignment_weight": ALIGNMENT_WEIGHT
    }

    torch.save(checkpoint, path)


def main():
    print("========================================")
    print("ALIGNED MASKED ENCODER TRAINING - V2")
    print("========================================")

    print("Device:", DEVICE)
    print("Alignment weight:", ALIGNMENT_WEIGHT)

    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(
            f"Original encoder checkpoint not found: {ORIGINAL_PATH}"
        )

    train_loader, val_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=200
    )

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    # Load frozen original encoder
    original_encoder = SEResEncoder(
        embedding_dim=512
    ).to(DEVICE)

    original_checkpoint = torch.load(
        ORIGINAL_PATH,
        map_location=DEVICE
    )

    original_encoder.load_state_dict(
        original_checkpoint["encoder_state_dict"]
    )

    original_encoder.eval()

    for parameter in original_encoder.parameters():
        parameter.requires_grad = False

    # Initialize masked teacher from original encoder weights
    model = SEResAutoencoder(
        embedding_dim=512
    ).to(DEVICE)

    model.encoder.load_state_dict(
        original_encoder.state_dict()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    reconstruction_criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):

        train_metrics = run_epoch(
            model,
            original_encoder,
            train_loader,
            reconstruction_criterion,
            optimizer
        )

        val_metrics = run_epoch(
            model,
            original_encoder,
            val_loader,
            reconstruction_criterion
        )

        train_loss, train_reconstruction, train_alignment = train_metrics

        val_loss, val_reconstruction, val_alignment = val_metrics

        if DEVICE.type == "mps":
            torch.mps.synchronize()

        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch [{epoch:02d}/{EPOCHS}]"
        )

        print(
            f"Train Total: {train_loss:.6f} | "
            f"Reconstruction: {train_reconstruction:.6f} | "
            f"Alignment: {train_alignment:.6f}"
        )

        print(
            f"Val Total: {val_loss:.6f} | "
            f"Reconstruction: {val_reconstruction:.6f} | "
            f"Alignment: {val_alignment:.6f} | "
            f"LR: {lr:.6f}"
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

            print("Best aligned teacher checkpoint saved.")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("\n========================================")
    print("ALIGNED MASKED ENCODER TRAINING COMPLETE")
    print("========================================")
    print("Best validation loss:", best_val_loss)
    print("Best checkpoint:", BEST_PATH)
    print("Last checkpoint:", LAST_PATH)


if __name__ == "__main__":
    main()

