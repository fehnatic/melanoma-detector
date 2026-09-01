"""Trainingsschleife.

Aufruf:  python -m melanoma.train
Optionen: --epochs, --batch-size, --backbone, --image-size, --limit (Smoke-Test)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from melanoma.config import load_config, resolve_paths
from melanoma.data.dataset import (
    MelanomaDataset,
    build_transforms,
    load_split,
    make_balanced_sampler,
)
from melanoma.models import ModelEma, build_model, pick_device


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    logits_all, labels_all = [], []
    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1).float().cpu()
        logits_all.append(logits)
        labels_all.append(labels)

    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Bei nur einer vertretenen Klasse sind AUROC/AP undefiniert.
    single_class = len(np.unique(labels)) < 2
    return {
        "auroc": float("nan") if single_class else float(roc_auc_score(labels, probs)),
        "ap": float("nan") if single_class else float(average_precision_score(labels, probs)),
        "loss": float(
            nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(logits), torch.from_numpy(labels)
            )
        ),
        "probs": probs,
        "labels": labels,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Melanom-Klassifikator trainieren")
    parser.add_argument("--config", default=None)
    parser.add_argument("--splits-name", default="default")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--limit", type=int, default=None, help="nur N Trainingsbilder (Smoke-Test)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    paths.mkdirs()

    t = dict(cfg["train"])
    d = dict(cfg["data"])
    backbone = args.backbone or t["backbone"]
    epochs = args.epochs or t["epochs"]
    batch_size = args.batch_size or t["batch_size"]
    image_size = args.image_size or d["image_size"]
    lr = args.lr or t["lr"]

    device = pick_device(args.device or t["device"])
    print(f"Device: {device} | Backbone: {backbone} | {image_size}px | bs={batch_size}")

    splits_csv = paths.processed / f"splits_{args.splits_name}.csv"
    if not splits_csv.exists():
        raise SystemExit(f"{splits_csv} fehlt - erst 'make data' laufen lassen.")

    train_df = load_split(splits_csv, "train")
    val_df = load_split(splits_csv, "val")
    if args.limit:
        train_df = train_df.sample(n=min(args.limit, len(train_df)), random_state=0).reset_index(
            drop=True
        )
        val_df = val_df.sample(
            n=min(max(args.limit // 4, 2), len(val_df)), random_state=0
        ).reset_index(drop=True)
    print(
        f"Train: {len(train_df)} ({train_df.label.mean():.3%} maligne) | "
        f"Val: {len(val_df)} ({val_df.label.mean():.3%} maligne)"
    )

    train_ds = MelanomaDataset(train_df, build_transforms(image_size, d["mean"], d["std"], True))
    val_ds = MelanomaDataset(val_df, build_transforms(image_size, d["mean"], d["std"], False))

    sampler = make_balanced_sampler(train_df) if t["imbalance"] == "sampler" else None
    num_workers = t["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    model = build_model(backbone, pretrained=True).to(device)
    ema = ModelEma(model, decay=t["ema_decay"]) if t["ema_decay"] else None

    pos_weight = None
    if t["imbalance"] == "pos_weight":
        n_pos = max(1, int(train_df["label"].sum()))
        pos_weight = torch.tensor([(len(train_df) - n_pos) / n_pos], device=device)
        print(f"pos_weight = {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=t["weight_decay"])
    total_steps = epochs * len(train_loader)
    warmup_steps = int(t["warmup_epochs"] * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, total_steps, warmup_steps)
    )

    run_dir = paths.artifacts / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_auroc, best_epoch, history = -1.0, -1, []
    smoothing = t["label_smoothing"]

    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        started = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if smoothing:
                labels = labels * (1 - 2 * smoothing) + smoothing

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images).squeeze(1), labels)
            loss.backward()
            if t["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()
            scheduler.step()
            if ema:
                ema.update(model)

            running += loss.item() * images.size(0)
            seen += images.size(0)
            pbar.set_postfix(loss=f"{running / seen:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        eval_model = ema.module.to(device) if ema else model
        metrics = evaluate(eval_model, val_loader, device)
        entry = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, seen),
            "val_loss": metrics["loss"],
            "val_auroc": metrics["auroc"],
            "val_ap": metrics["ap"],
            "seconds": round(time.time() - started, 1),
        }
        history.append(entry)
        print(
            f"  epoch {epoch + 1}: train_loss={entry['train_loss']:.4f} "
            f"val_loss={entry['val_loss']:.4f} AUROC={entry['val_auroc']:.4f} "
            f"AP={entry['val_ap']:.4f} ({entry['seconds']}s)"
        )

        if metrics["auroc"] > best_auroc:
            best_auroc, best_epoch = metrics["auroc"], epoch + 1
            torch.save(
                {
                    "state_dict": eval_model.state_dict(),
                    "backbone": backbone,
                    "image_size": image_size,
                    "mean": d["mean"],
                    "std": d["std"],
                    "epoch": epoch + 1,
                    "val_auroc": best_auroc,
                    "train_prevalence": float(train_df["label"].mean()),
                },
                run_dir / "best.pt",
            )
            np.savez(
                run_dir / "val_predictions.npz",
                probs=metrics["probs"],
                labels=metrics["labels"],
            )
            print(f"  -> neues bestes Modell (AUROC {best_auroc:.4f})")

        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if t["early_stopping_patience"] and epoch + 1 - best_epoch >= t["early_stopping_patience"]:
            print(f"Early stopping: seit {t['early_stopping_patience']} Epochen keine Besserung.")
            break

    print(f"\nBestes Modell: Epoche {best_epoch}, AUROC {best_auroc:.4f} -> {run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
