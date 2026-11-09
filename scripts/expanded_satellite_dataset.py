import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_FILE = (
    PROJECT_ROOT / "data/expanded_2022/metadata/expanded_dataset_manifest.csv"
)

STATS_FILE = (
    PROJECT_ROOT / "data/expanded_2022/metadata/normalization_stats.json"
)

CHANNELS = ["B07", "B08", "B13"]
EXPECTED_SHAPE = (1000, 1000)


class ExpandedHimawariDataset(Dataset):

    def __init__(self, split="train", image_size=200):

        if split not in ("train", "validation", "test"):
            raise ValueError("Invalid split")

        self.image_size = image_size

        self.split_df = pd.read_csv(MANIFEST_FILE)
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

        image_path = PROJECT_ROOT / row["npz_path"]

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with np.load(image_path) as data:

            channels = []

            for channel in CHANNELS:

                image = data[channel].astype(np.float32)

                if image.shape != EXPECTED_SHAPE:
                    raise ValueError(
                        f"Incorrect shape in {image_path}: {image.shape}"
                    )

                mean = float(self.stats[channel]["mean_kelvin"])
                std = float(self.stats[channel]["std_kelvin"])

                if not np.isfinite(std) or std <= 0:
                    raise ValueError(f"Invalid std for {channel}")

                valid = np.isfinite(image)

                normalized = np.zeros_like(image, dtype=np.float32)

                normalized[valid] = (
                    image[valid] - mean
                ) / std

                normalized = np.clip(normalized, -5.0, 5.0)

                channels.append(normalized)

        image = np.stack(channels, axis=0)

        tensor = torch.from_numpy(image).unsqueeze(0)

        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        )

        tensor = tensor.squeeze(0)

        filename = str(row["observation"])

        return tensor, filename


def create_expanded_dataloaders(batch_size=2, image_size=200):

    loaders = {}

    for split in ("train", "validation", "test"):

        dataset = ExpandedHimawariDataset(
            split=split,
            image_size=image_size
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0
        )

    return loaders


if __name__ == "__main__":

    print("Loading expanded Himawari dataset...")

    loaders = create_expanded_dataloaders(
        batch_size=2,
        image_size=200
    )

    for split, loader in loaders.items():

        images, filenames = next(iter(loader))

        print(f"\n{split.upper()}")
        print("Observations:", len(loader.dataset))
        print("Batch shape:", images.shape)
        print("Data type:", images.dtype)
        print("Minimum:", images.min().item())
        print("Maximum:", images.max().item())
        print("All finite:", torch.isfinite(images).all().item())
        print("Filenames:", filenames)

        assert images.shape == (2, 3, 200, 200)
        assert torch.isfinite(images).all()

    print("\nEXPANDED DATASET LOADER TEST PASSED")
