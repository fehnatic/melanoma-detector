"""Torch-Dataset und Augmentierungen."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF


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


def _pad_to_square(img):
    """Polstert ein Bild mit gespiegelten Randpixeln auf ein Quadrat auf, statt
    etwas abzuschneiden. Fuer Fotos mit beliebigem Seitenverhaeltnis (normale
    Handyfotos) wichtig: ISIC/PAD-UFES sind kuratiert und nahezu quadratisch,
    ein frei aufgenommenes Foto aber nicht - CenterCrop wuerde dabei im
    schlimmsten Fall die Laesion abschneiden, wenn sie nicht exakt mittig ist.
    """
    w, h = img.size if hasattr(img, "size") else img.shape[-1::-1][:2]
    diff = abs(w - h)
    if diff == 0:
        return img
    pad1, pad2 = diff // 2, diff - diff // 2
    padding = (0, pad1, 0, pad2) if w > h else (pad1, 0, pad2, 0)
    try:
        return TF.pad(img, padding, padding_mode="reflect")
    except RuntimeError:
        # reflect verlangt Padding < Kantenlaenge - bei extremen Seitenverhaeltnissen
        # (z.B. Screenshot-Streifen) auf Randwiederholung ausweichen, die dafuer keine
        # Grenze kennt.
        return TF.pad(img, padding, padding_mode="edge")


def build_preview_crop(image_size: int):
    """Reine PIL->PIL-Variante des Eval-Zuschnitts (Resize+CenterCrop), ohne
    Tensor-Konvertierung/Normalisierung. Damit laesst sich anzeigen, was das
    Modell tatsaechlich zu sehen bekommt - z.B. als Grundlage fuer die
    Grad-CAM-Heatmap in explain.py. Wichtig: exakt dieselben Schritte wie im
    Eval-Zweig von build_transforms(train=False, pad_to_square=False), sonst
    passen Heatmap und angezeigtes Bild nicht zusammen."""
    return v2.Compose([v2.Resize(int(image_size * 1.15)), v2.CenterCrop(image_size)])


def build_transforms(image_size: int, mean, std, train: bool, pad_to_square: bool = False):
    """Dermatoskopische Bilder haben keine kanonische Orientierung - Flips und
    Rotationen in alle Richtungen sind daher unbedenklich und wirksam.

    `pad_to_square` betrifft nur den Eval-Zweig: statt Resize+CenterCrop (schneidet
    bei nicht-quadratischen Bildern Inhalt weg) wird das Bild vollstaendig erhalten
    und auf ein Quadrat aufgepolstert. Siehe `_pad_to_square`. Gemessen kostet das
    aber AUROC (siehe README) - Standard bleibt daher Resize+CenterCrop.
    """
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
    if pad_to_square:
        return v2.Compose(
            [
                v2.Lambda(_pad_to_square),
                v2.Resize((image_size, image_size)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )
    return v2.Compose(
        [
            *build_preview_crop(image_size).transforms,
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
