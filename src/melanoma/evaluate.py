"""Auswertung auf dem Testsplit - inklusive Aufschluesselung nach Quelle."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from melanoma.calibrate import load_calibrator
from melanoma.config import load_config, resolve_paths
from melanoma.data.dataset import MelanomaDataset, build_transforms, load_split
from melanoma.models import build_model, pick_device


def operating_points(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Schwellwerte, die klinisch relevant sind: hohe Sensitivitaet zuerst."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    points = {}
    for target in (0.90, 0.95, 0.98):
        idx = int(np.argmax(tpr >= target))
        if tpr[idx] < target:
            continue
        points[f"sens_{int(target * 100)}"] = {
            "threshold": float(thresholds[idx]),
            "sensitivity": float(tpr[idx]),
            "specificity": float(1 - fpr[idx]),
        }
    return points


@torch.no_grad()
def predict_loader(model, loader, device) -> np.ndarray:
    from tqdm import tqdm

    model.eval()
    out = []
    for images, _ in tqdm(loader, desc="test"):
        logits = model(images.to(device)).squeeze(1).float().cpu().numpy()
        out.append(1.0 / (1.0 + np.exp(-logits)))
    return np.concatenate(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Modell auf dem Testsplit auswerten")
    parser.add_argument("--config", default=None)
    parser.add_argument("--splits-name", default="default")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    run_dir = paths.artifacts / args.run_name

    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} fehlt - erst trainieren.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    device = pick_device(cfg["train"]["device"])
    model = build_model(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    df = load_split(paths.processed / f"splits_{args.splits_name}.csv", args.split)
    transform = build_transforms(ckpt["image_size"], ckpt["mean"], ckpt["std"], train=False)
    loader = DataLoader(
        MelanomaDataset(df, transform),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    raw = predict_loader(model, loader, device)

    calibrator = load_calibrator(run_dir)
    if calibrator is None:
        print("Warnung: kein Kalibrator gefunden - rohe Wahrscheinlichkeiten.")
        probs = raw
    else:
        probs = calibrator(raw)

    labels = df["label"].to_numpy()
    preds = (probs >= cfg["inference"]["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    results = {
        "split": args.split,
        "n": int(len(labels)),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "at_threshold": {
            "threshold": cfg["inference"]["threshold"],
            "sensitivity": float(tp / max(1, tp + fn)),
            "specificity": float(tn / max(1, tn + fp)),
            "precision": float(tp / max(1, tp + fp)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "operating_points": operating_points(labels, probs),
        "by_source": {},
    }

    for source, group in df.groupby("source"):
        mask = df["source"].to_numpy() == source
        y, p = labels[mask], probs[mask]
        results["by_source"][source] = {
            "n": int(len(y)),
            "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        }

    out = paths.reports / f"metrics_{args.run_name}_{args.split}.json"
    paths.reports.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
