
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResEncoder
from models.spatial_autoencoder_v4 import SpatialSEResAutoencoderV4

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"

INITIAL_PATH = CHECKPOINT_DIR / "spatial_teacher_v5_initial.pth"
ORIGINAL_PATH = CHECKPOINT_DIR / "original_encoder.pth"

BEST_PATH = CHECKPOINT_DIR / "best_spatial_teacher_v5.pth"
LAST_PATH = CHECKPOINT_DIR / "last_spatial_teacher_v5.pth"

EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 0.0001
ALIGNMENT_WEIGHT = 0.1
PATIENCE = 7

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def random_square_mask(batch, height, width, device):

    mask = torch.ones(
        batch, 1, height, width,
        device=device
    )

    for i in range(batch):

        side = torch.randint(40, 101, (1,)).item()

        top = torch.randint(
            0, height - side + 1, (1,)
        ).item()

        left = torch.randint(
            0, width - side + 1, (1,)
        ).item()

        mask[i, :, top:top + side, left:left + side] = 0

    return mask


def run_epoch(model, original_encoder, loader,
              criterion, optimizer=None):

    training = optimizer is not None

    model.train() if training else model.eval()
    original_encoder.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_alignment = 0.0
    batches = 0

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

            with torch.no_grad():
                reference = original_encoder(images)

            reconstruction, embedding, attention = model(
                masked_images,
                mask,
                return_attention=True
            )

            reconstruction_loss = criterion(
                reconstruction,
                images
            )

            alignment_loss = (
                1 - F.cosine_similarity(
                    embedding,
                    reference,
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

        total_loss += loss.item()
        total_reconstruction += reconstruction_loss.item()
        total_alignment += alignment_loss.item()

        batches += 1

    return (
        total_loss / batches,
        total_reconstruction / batches,
        total_alignment / batches
    )


def save_checkpoint(path, model, optimizer, epoch, loss):

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "validation_loss": loss,
            "embedding_dim": 512,
            "architecture": "SpatialSEResAutoencoderV4"
        },
        path
    )


def main():

    print("\nV5 REGION-AWARE TEACHER TRAINING")
    print("================================")
    print("Device:", DEVICE)

    train_loader, val_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=200
    )

    print("Training observations:", len(train_loader.dataset))
    print("Validation observations:", len(val_loader.dataset))

    original_checkpoint = torch.load(
        ORIGINAL_PATH,
        map_location=DEVICE
    )

    original_encoder = SEResEncoder(
        embedding_dim=512
    ).to(DEVICE)

    original_encoder.load_state_dict(
        original_checkpoint["encoder_state_dict"]
    )

    original_encoder.eval()

    for parameter in original_encoder.parameters():
        parameter.requires_grad = False

    initial_checkpoint = torch.load(
        INITIAL_PATH,
        map_location=DEVICE
    )

    model = SpatialSEResAutoencoderV4(
        embedding_dim=512
    ).to(DEVICE)

    model.load_state_dict(
        initial_checkpoint["model_state_dict"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        train_metrics = run_epoch(
            model,
            original_encoder,
            train_loader,
            criterion,
            optimizer
        )

        val_metrics = run_epoch(
            model,
            original_encoder,
            val_loader,
            criterion
        )

        train_loss, train_rec, train_align = train_metrics
        val_loss, val_rec, val_align = val_metrics

        scheduler.step(val_loss)

        print(f"\nEpoch [{epoch:02d}/{EPOCHS}]")

        print(
            f"Train Total: {train_loss:.6f} | "
            f"Reconstruction: {train_rec:.6f} | "
            f"Alignment: {train_align:.6f}"
        )

        print(
            f"Val Total: {val_loss:.6f} | "
            f"Reconstruction: {val_rec:.6f} | "
            f"Alignment: {val_align:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        save_checkpoint(
            LAST_PATH,
            model,
            optimizer,
            epoch,
            val_loss
        )

        if val_loss < best_loss:

            best_loss = val_loss
            patience_counter = 0

            save_checkpoint(
                BEST_PATH,
                model,
                optimizer,
                epoch,
                val_loss
            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")
            break

    print("\nV5 TRAINING COMPLETE")
    print("Best validation loss:", best_loss)
    print("Best checkpoint:", BEST_PATH)
    print("Last checkpoint:", LAST_PATH)


if __name__ == "__main__":
    main()

