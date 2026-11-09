
import torch
import torch.nn.functional as F
import pandas as pd

from models.spatial_autoencoder_v4 import SpatialSEResAutoencoderV4
from satellite_dataset import create_dataloaders

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("V4 ATTENTION DIAGNOSTICS")
print("Device:", DEVICE)

checkpoint = torch.load(
    "checkpoints/best_spatial_teacher_v4.pth",
    map_location=DEVICE
)

model = SpatialSEResAutoencoderV4(512).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

train_loader, test_loader = create_dataloaders(
    batch_size=1,
    image_size=200
)


def create_mask(height, width, side, top, left):
    mask = torch.ones(1, 1, height, width)

    mask[:, :, top:top + side, left:left + side] = 0

    return mask.to(DEVICE)


def attention_metrics(attention, mask):
    attention = attention.flatten()
    visible = F.interpolate(
        mask,
        size=(13, 13),
        mode="nearest"
    ).flatten() > 0.5

    visible_weights = attention[visible]

    entropy = -(
        visible_weights *
        torch.log(visible_weights.clamp_min(1e-12))
    ).sum()

    normalized_entropy = entropy / torch.log(
        torch.tensor(
            float(len(visible_weights)),
            device=DEVICE
        )
    )

    return {
        "visible_cells": visible.sum().item(),
        "hidden_cells": (~visible).sum().item(),
        "attention_sum": attention.sum().item(),
        "hidden_attention": attention[~visible].sum().item(),
        "max_attention": attention.max().item(),
        "min_visible_attention": visible_weights.min().item(),
        "max_visible_attention": visible_weights.max().item(),
        "entropy": entropy.item(),
        "normalized_entropy": normalized_entropy.item()
    }


records = []

with torch.no_grad():

    for index, (images, filenames) in enumerate(test_loader):

        if index >= 3:
            break

        images = images.to(DEVICE)

        height, width = images.shape[2:]

        masks = {
            "full": torch.ones(
                1, 1, height, width,
                device=DEVICE
            ),
            "top_left": create_mask(
                height, width, 80, 0, 0
            ),
            "center": create_mask(
                height, width, 80, 60, 60
            ),
            "bottom_right": create_mask(
                height, width, 80, 120, 120
            )
        }

        for mask_name, mask in masks.items():

            masked_image = images * mask

            reconstruction, embedding, attention = model(
                masked_image,
                mask,
                return_attention=True
            )

            metrics = attention_metrics(
                attention,
                mask
            )

            metrics["query"] = filenames[0]
            metrics["mask"] = mask_name
            metrics["embedding_norm"] = embedding.norm().item()

            records.append(metrics)

            print("\nQuery:", filenames[0])
            print("Mask:", mask_name)

            for key, value in metrics.items():

                if key not in ["query", "mask"]:
                    print(
                        f"{key}: {value:.6f}"
                        if isinstance(value, float)
                        else f"{key}: {value}"
                    )


df = pd.DataFrame(records)

df.to_csv(
    "results/attention_diagnostics_v4.csv",
    index=False
)

print("\n================================")
print("ATTENTION DIAGNOSTICS COMPLETE")
print("================================")

print(df.to_string(index=False))

print("\nSaved:")
print("results/attention_diagnostics_v4.csv")

