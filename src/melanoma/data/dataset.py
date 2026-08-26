"""Torch-Dataset und Augmentierungen."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2


class MelanomaDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform, with_label: bool = True) -> None:
        self.paths = frame["image_path"].tolist()
        self.labels = frame["label"].tolist() if with_label else None
        self.image_ids = frame["image_id"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            image = self.transform(img.convert("RGB"))
        if self.labels is None:
            return image, self.image_ids[idx]
        return image, torch.tensor(float(self.labels[idx]))


def build_transforms(image_size: int, mean, std, train: bool):
    """Dermatoskopische Bilder haben keine kanonische Orientierung - Flips und
    Rotationen in alle Richtungen sind daher unbedenklich und wirksam."""
    if train:
        return v2.Compose(
            [
                v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomApply([v2.RandomRotation(30)], p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ]
        )
    return v2.Compose(
        [
            v2.Resize(int(image_size * 1.15)),
            v2.CenterCrop(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def load_split(splits_csv: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(splits_csv)
    subset = df[df["split"] == split].reset_index(drop=True)
    if subset.empty:
        raise SystemExit(f"Split '{split}' ist leer in {splits_csv}")
    return subset


def make_balanced_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    """Oversampled die Minderheitsklasse, sodass ein Batch im Mittel ausgeglichen ist."""
    counts = frame["label"].value_counts().to_dict()
    weights = frame["label"].map(lambda y: 1.0 / counts[y]).astype(float).tolist()
    return WeightedRandomSampler(weights, num_samples=len(frame), replacement=True)
