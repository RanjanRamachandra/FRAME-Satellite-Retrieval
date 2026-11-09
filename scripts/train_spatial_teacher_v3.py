import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from satellite_dataset import create_dataloaders
from models.se_res_autoencoder import SEResEncoder
from models.spatial_autoencoder import SpatialSEResAutoencoder


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"

ORIGINAL_PATH = CHECKPOINT_DIR / "original_encoder.pth"
V2_PATH = CHECKPOINT_DIR / "best_masked_encoder_aligned.pth"
V3_INIT_PATH = CHECKPOINT_DIR / "spatial_teacher_v3_initial.pth"

BEST_PATH = CHECKPOINT_DIR / "best_spatial_teacher_v3.pth"
LAST_PATH = CHECKPOINT_DIR / "last_spatial_teacher_v3.pth"

EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 0.0003
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
    criterion,
    optimizer=None
):

    training = optimizer is not None

    model.train() if training else model.eval()
    original_encoder.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_alignment = 0.0
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

            with torch.no_grad():
                reference = original_encoder(images)

            reconstruction, embedding = model(masked_images)

            reconstruction_loss = criterion(
                reconstruction, images
            )

            alignment_loss = (
                1.0 - F.cosine_similarity(
                    embedding, reference, dim=1
                ).mean()
            )

            loss = (
                reconstruction_loss
                + ALIGNMENT_WEIGHT * alignment_loss
            )

            if training:
                loss.backward()
                optimizer.step()

        batch = images.size(0)

        total_loss += loss.item() * batch
        total_reconstruction += reconstruction_loss.item() * batch
        total_alignment += alignment_loss.item() * batch
        total_images += batch

    return (
        total_loss / total_images,
        total_reconstruction / total_images,
        total_alignment / total_images
    )


def save_checkpoint(model, optimizer, epoch, val_loss, path):

    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "encoder_state_dict": {
            key: value.detach().cpu()
            for key, value in model.encoder.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": val_loss,
        "embedding_dim": 512,
        "alignment_weight": ALIGNMENT_WEIGHT,
        "architecture": "SpatialSEResAutoencoderV3"
    }

    torch.save(checkpoint, path)


def main():

    print("========================================")
    print("SPATIAL MASKED TEACHER TRAINING - V3")
    print("========================================")

    print("Device:", DEVICE)

    train_loader, val_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=200
    )

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

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

    model = SpatialSEResAutoencoder(
        embedding_dim=512
    ).to(DEVICE)

    # Load initialized spatial encoder.
    init_checkpoint = torch.load(
        V3_INIT_PATH,
        map_location="cpu"
    )

    model.encoder.load_state_dict(
        init_checkpoint["encoder_state_dict"]
    )

    # Transfer the trained V2 decoder.
    v2_checkpoint = torch.load(
        V2_PATH,
        map_location="cpu"
    )

    v2_model_weights = v2_checkpoint["model_state_dict"]

    decoder_weights = {
        key.replace("decoder.", "", 1): value
        for key, value in v2_model_weights.items()
        if key.startswith("decoder.")
    }

    model.decoder.load_state_dict(decoder_weights)

    print("V3 spatial encoder initialized.")
    print("V2 decoder transferred successfully.")

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

        if DEVICE.type == "mps":
            torch.mps.synchronize()

        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]

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
            f"LR: {lr:.6f}"
        )

        save_checkpoint(
            model, optimizer, epoch, val_loss, LAST_PATH
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            save_checkpoint(
                model, optimizer, epoch, val_loss, BEST_PATH
            )

            print("Best V3 checkpoint saved.")

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print("Early stopping triggered.")
            break

    print("\nV3 TEACHER TRAINING COMPLETE")
    print("Best validation loss:", best_val_loss)
    print("Best checkpoint:", BEST_PATH)
    print("Last checkpoint:", LAST_PATH)


if __name__ == "__main__":
    main()
