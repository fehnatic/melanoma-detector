"""Cross-Domain-Auswertung: ein bereits trainiertes Modell auf einem strukturell
anderen Datensatz laufen lassen - reine Inferenz, kein Training, kein Split.

Aktuell verfuegbar: PAD-UFES-20 (klinische Handyfotos statt Dermoskopie,
siehe melanoma.data.download.download_pad_ufes20). Damit laesst sich pruefen,
wie gut ein auf Dermoskopie trainiertes Modell mit normalen Handyfotos
zurechtkommt - relevant, bevor man Zeit in ein eigenes Training auf
Handyfoto-Daten steckt.

Aufruf:
    python -m melanoma.eval_external --dataset pad_ufes_20 --run-name isic_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from melanoma.calibrate import load_calibrator
from melanoma.config import load_config, resolve_paths
from melanoma.data.dataset import MelanomaDataset, build_transforms
from melanoma.evaluate import operating_points, predict_loader
from melanoma.models import build_model, pick_device


def build_pad_ufes20_manifest(raw_dir: Path) -> pd.DataFrame:
    root = raw_dir / "pad_ufes_20"
    gt_path = root / "ground_truth.csv"
    if not gt_path.exists():
        raise SystemExit(
            f"{gt_path} fehlt - erst 'python -m melanoma.data.download --sources pad_ufes_20' laufen lassen."
        )

    df = pd.read_csv(gt_path)
    df["image_path"] = df["image_name"].map(lambda n: str(root / "images" / f"{n}.jpg"))
    # Gleiche Semantik wie das ISIC-2020-Label (target): Melanom vs. Rest, nicht
    # "irgendein Malignom" - BCC/SCC zaehlen hier bewusst als 0.
    df["label"] = df["diagnosis"].str.contains("melanoma", case=False, na=False).astype(int)
    df["source"] = "pad_ufes_20"
    df["image_id"] = "pad_ufes_20/" + df["image_name"]
    return df


DATASETS = {
    "pad_ufes_20": build_pad_ufes20_manifest,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Modell auf einem externen Datensatz auswerten")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--dataset", default="pad_ufes_20", choices=list(DATASETS))
    parser.add_argument(
        "--pad-to-square",
        action="store_true",
        help="Bild vollstaendig erhalten (Padding) statt Resize+CenterCrop - siehe dataset.build_transforms",
    )
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

    df = DATASETS[args.dataset](paths.raw)
    print(f"{args.dataset}: {len(df)} Bilder ({df['label'].mean():.3%} maligne)")

    transform = build_transforms(
        ckpt["image_size"], ckpt["mean"], ckpt["std"], train=False, pad_to_square=args.pad_to_square
    )
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
        "dataset": args.dataset,
        "run_name": args.run_name,
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
    }

    suffix = "_padded" if args.pad_to_square else ""
    out = paths.reports / f"metrics_{args.run_name}_{args.dataset}{suffix}.json"
    paths.reports.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
