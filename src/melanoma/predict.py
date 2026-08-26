"""Inferenz fuer ein einzelnes Bild - genutzt von der CLI und der Gradio-App."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from melanoma.calibrate import load_calibrator
from melanoma.config import load_config, resolve_paths
from melanoma.data.dataset import build_transforms
from melanoma.models import build_model, pick_device


class Predictor:
    """Laedt Checkpoint und Kalibrator einmal und haelt sie im Speicher."""

    def __init__(self, run_dir: Path, device: str = "auto", tta: bool = True) -> None:
        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Kein Modell unter {ckpt_path}. Erst trainieren: python -m melanoma.train"
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.device = pick_device(device)
        self.model = build_model(ckpt["backbone"], pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)

        self.image_size = ckpt["image_size"]
        self.transform = build_transforms(self.image_size, ckpt["mean"], ckpt["std"], train=False)
        self.tta = tta
        self.meta = {
            "backbone": ckpt["backbone"],
            "epoch": ckpt.get("epoch"),
            "val_auroc": ckpt.get("val_auroc"),
        }

        self.calibrator = load_calibrator(run_dir)
        self.reference_prevalence = self.calibrator.prevalence if self.calibrator else None

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        # Test-Time-Augmentation: dermatoskopische Bilder sind flip-invariant,
        # das Mitteln ueber die vier Spiegelungen glaettet den Wert spuerbar.
        batch = [tensor]
        if self.tta:
            batch += [
                torch.flip(tensor, dims=[3]),
                torch.flip(tensor, dims=[2]),
                torch.flip(tensor, dims=[2, 3]),
            ]
        logits = self.model(torch.cat(batch)).squeeze(1).float().cpu().numpy()
        raw = float(np.mean(1.0 / (1.0 + np.exp(-logits))))

        calibrated = float(self.calibrator(np.array([raw]))[0]) if self.calibrator else raw
        return {
            "prob_malignant": calibrated,
            "prob_benign": 1.0 - calibrated,
            "prob_raw": raw,
            "reference_prevalence": self.reference_prevalence,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wahrscheinlichkeit fuer ein Bild ausgeben")
    parser.add_argument("image", nargs="+", help="Pfad(e) zu Bilddatei(en)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default="baseline")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    predictor = Predictor(
        paths.artifacts / args.run_name,
        device=cfg["train"]["device"],
        tta=cfg["inference"]["tta"],
    )

    for path in args.image:
        with Image.open(path) as img:
            result = predictor.predict(img)
        print(
            f"{path}: maligne {result['prob_malignant']:.1%} | "
            f"benigne {result['prob_benign']:.1%}"
        )
    print("\nHinweis: Kein Medizinprodukt. Ersetzt keine aerztliche Abklaerung.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
