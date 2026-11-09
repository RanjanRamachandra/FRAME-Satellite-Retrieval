
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data/processed/satellite_images/7days"
IMAGE_DIR = DATA_DIR
SPLIT_FILE = DATA_DIR / "dataset_split.csv"
STATS_FILE = DATA_DIR / "normalization_stats.json"

CHANNELS = ["B07", "B08", "B13"]


class HimawariDataset(Dataset):

    def __init__(self, split="train", image_size=200):

        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        self.image_size = image_size

        self.split_df = pd.read_csv(SPLIT_FILE)
        self.split_df = self.split_df[
            self.split_df["split"] == split
        ].reset_index(drop=True)

        with open(STATS_FILE, "r") as f:
            self.stats = json.load(f)

        if len(self.split_df) == 0:
            raise ValueError(f"No images found for split: {split}")

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, index):

        row = self.split_df.iloc[index]
        filename = str(row["filename"])

        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with np.load(image_path) as data:

            channels = []

            for channel in CHANNELS:

                image = data[channel].astype(np.float32)

                mean = float(self.stats[channel]["mean_kelvin"])
                std = float(self.stats[channel]["std_kelvin"])

                if not np.isfinite(std) or std <= 0:
                    raise ValueError(
                        f"Invalid standard deviation for {channel}: {std}"
                    )

                valid = np.isfinite(image)

                normalized = np.zeros_like(image, dtype=np.float32)

                normalized[valid] = (
                    image[valid] - mean
                ) / std

                normalized = np.clip(normalized, -5.0, 5.0)

                channels.append(normalized)

        image = np.stack(channels, axis=0)

        tensor = torch.from_numpy(image)

        tensor = tensor.unsqueeze(0)

        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        )

        tensor = tensor.squeeze(0)

        return tensor, filename


def create_dataloaders(batch_size=4, image_size=200):

    train_dataset = HimawariDataset(
        split="train",
        image_size=image_size
    )

    test_dataset = HimawariDataset(
        split="test",
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


if __name__ == "__main__":

    print("Loading Himawari dataset...")

    train_loader, test_loader = create_dataloaders(
        batch_size=2,
        image_size=200
    )

    print("Training images:", len(train_loader.dataset))
    print("Testing images:", len(test_loader.dataset))

    images, filenames = next(iter(train_loader))

    print("\nBatch validation")
    print("Tensor shape:", images.shape)
    print("Data type:", images.dtype)
    print("Minimum:", images.min().item())
    print("Maximum:", images.max().item())
    print("Mean:", images.mean().item())
    print("Standard deviation:", images.std().item())
    print("Filenames:", filenames)

    assert images.shape == (2, 3, 200, 200)
    assert torch.isfinite(images).all()

    print("\nDATA LOADER TEST PASSED")

