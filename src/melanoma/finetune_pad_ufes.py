"""Fine-Tuning eines bestehenden Checkpoints auf PAD-UFES-20 (Handyfotos).

Nur 52 Melanom-Faelle insgesamt in PAD-UFES-20 - ein einzelner Train/Val/Test-
Split wie bei ISIC waere kaum belastbar (ein paar Positive im Val, hohe
Varianz). Stattdessen gruppen-stratifizierte K-Fold-Kreuzvalidierung nach
patient_id: pro Fold wird der Basis-Checkpoint kurz mit niedriger Lernrate
weitertrainiert, und die Out-of-Fold-Vorhersagen aller Folds zusammen ergeben
eine einzige AUROC ueber alle 2298 Bilder - ohne dass ein Bild je gleichzeitig
in Training und Auswertung war.

Aufruf:
    python -m melanoma.finetune_pad_ufes --base-run isic_baseline --run-name pad_ufes_finetune
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from melanoma.config import load_config, resolve_paths
from melanoma.data.dataset import MelanomaDataset, build_transforms, make_balanced_sampler
from melanoma.eval_external import build_pad_ufes20_manifest
from melanoma.evaluate import operating_points
from melanoma.models import ModelEma, build_model, pick_device
from melanoma.train import cosine_with_warmup
from melanoma.train import evaluate as run_eval


def finetune_fold(fold, train_df, val_df, base_ckpt, cfg, device, epochs, lr):
    d, t = dict(cfg["data"]), dict(cfg["train"])
    image_size = base_ckpt["image_size"]

    model = build_model(base_ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(base_ckpt["state_dict"])
    ema = ModelEma(model, decay=t["ema_decay"]) if t["ema_decay"] else None

    train_loader = DataLoader(
        MelanomaDataset(train_df, build_transforms(image_size, d["mean"], d["std"], True)),
        batch_size=t["batch_size"],
        sampler=make_balanced_sampler(train_df),
        num_workers=t["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        MelanomaDataset(val_df, build_transforms(image_size, d["mean"], d["std"], False)),
        batch_size=t["batch_size"],
        shuffle=False,
        num_workers=t["num_workers"],
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=t["weight_decay"])
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, total_steps, warmup_steps)
    )

    best_auroc, best_state, best_probs, best_labels = -1.0, None, None, None
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images).squeeze(1), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()
            scheduler.step()
            if ema:
                ema.update(model)

        eval_model = ema.module.to(device) if ema else model
        metrics = run_eval(eval_model, val_loader, device)
        auroc_str = f"{metrics['auroc']:.4f}" if not np.isnan(metrics["auroc"]) else "nan"
        print(f"  fold {fold + 1} epoch {epoch + 1}/{epochs}: val_auroc={auroc_str}")

        # Bei nan (Val-Fold ohne beide Klassen) trotzdem den letzten Stand behalten,
        # damit der Fold am Ende ueberhaupt Out-of-Fold-Vorhersagen liefert.
        if best_state is None or (not np.isnan(metrics["auroc"]) and metrics["auroc"] > best_auroc):
            best_auroc = metrics["auroc"]
            best_state = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
            best_probs, best_labels = metrics["probs"], metrics["labels"]

    return best_probs, best_labels, best_auroc, best_state


def build_final_split(df, val_fraction: float, seed: int):
    """Kleiner gruppen-stratifizierter Holdout - dient nur der Kalibrierung, nicht
    der Epochenwahl (die ist bereits per Kreuzvalidierung entschieden)."""
    n_splits = max(2, round(1.0 / val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, y=df["label"], groups=df["patient_id"]))
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)


def train_final(df, base_ckpt, cfg, device, epochs, lr, val_fraction, seed):
    """Ein einzelnes Produktionsmodell auf (fast) allen Daten.

    Anders als bei der Kreuzvalidierung wird hier NICHT die Epoche mit der
    besten Val-AUROC behalten - das waere bei nur ~10 positiven Val-Beispielen
    zu verrauscht. Stattdessen wird die per CV bereits validierte Epochenzahl
    fest durchtrainiert und die letzte Epoche gespeichert; der kleine Holdout
    dient nur dazu, ueberhaupt Vorhersagen fuer die Kalibrierung zu haben.
    """
    d, t = dict(cfg["data"]), dict(cfg["train"])
    train_df, val_df = build_final_split(df, val_fraction, seed)
    print(
        f"Finales Modell: {len(train_df)} train / {len(val_df)} val "
        f"(Val nur fuer Kalibrierung, {int(val_df['label'].sum())} maligne)"
    )

    image_size = base_ckpt["image_size"]
    model = build_model(base_ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(base_ckpt["state_dict"])
    ema = ModelEma(model, decay=t["ema_decay"]) if t["ema_decay"] else None

    train_loader = DataLoader(
        MelanomaDataset(train_df, build_transforms(image_size, d["mean"], d["std"], True)),
        batch_size=t["batch_size"],
        sampler=make_balanced_sampler(train_df),
        num_workers=t["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        MelanomaDataset(val_df, build_transforms(image_size, d["mean"], d["std"], False)),
        batch_size=t["batch_size"],
        shuffle=False,
        num_workers=t["num_workers"],
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=t["weight_decay"])
    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, total_steps, warmup_steps)
    )

    metrics = None
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images).squeeze(1), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()
            scheduler.step()
            if ema:
                ema.update(model)

        eval_model = ema.module.to(device) if ema else model
        metrics = run_eval(eval_model, val_loader, device)
        auroc_str = f"{metrics['auroc']:.4f}" if not np.isnan(metrics["auroc"]) else "nan"
        print(f"  epoch {epoch + 1}/{epochs}: val_auroc={auroc_str}")

    return eval_model, metrics, float(train_df["label"].mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-Tuning auf PAD-UFES-20, ausgewertet per Gruppen-CV")
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-run", default="isic_baseline", help="Checkpoint, der fine-getuned wird")
    parser.add_argument("--run-name", default="pad_ufes_finetune")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--final",
        action="store_true",
        help="kein Fold-CV, sondern ein einzelnes Produktionsmodell (best.pt) auf fast allen Daten",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15, help="nur mit --final: Holdout fuer Kalibrierung")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)

    base_ckpt_path = paths.artifacts / args.base_run / "best.pt"
    if not base_ckpt_path.exists():
        raise SystemExit(f"{base_ckpt_path} fehlt - erst trainieren.")
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)

    device = pick_device(cfg["train"]["device"])
    df = build_pad_ufes20_manifest(paths.raw).reset_index(drop=True)
    print(
        f"pad_ufes_20: {len(df)} Bilder ({df['label'].mean():.3%} maligne), "
        f"{df['patient_id'].nunique()} Patienten, Basis-Checkpoint: {args.base_run}"
    )

    if args.final:
        run_dir = paths.artifacts / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        model, metrics, train_prevalence = train_final(
            df, base_ckpt, cfg, device, args.epochs, args.lr, args.val_fraction, args.seed
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "backbone": base_ckpt["backbone"],
                "image_size": base_ckpt["image_size"],
                "mean": cfg["data"]["mean"],
                "std": cfg["data"]["std"],
                "epoch": args.epochs,
                "val_auroc": metrics["auroc"],
                "train_prevalence": train_prevalence,
                "finetuned_from": args.base_run,
            },
            run_dir / "best.pt",
        )
        np.savez(run_dir / "val_predictions.npz", probs=metrics["probs"], labels=metrics["labels"])
        print(f"\n-> {run_dir / 'best.pt'}")
        print("Naechster Schritt fuer kalibrierte Wahrscheinlichkeiten:")
        print(f"  python -m melanoma.calibrate --run-name {args.run_name}")
        return 0

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros(len(df))
    oof_labels = df["label"].to_numpy().astype(float)
    fold_reports = []

    run_dir = paths.artifacts / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(df, y=df["label"], groups=df["patient_id"])):
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        print(
            f"\n== Fold {fold + 1}/{args.folds}: {len(train_df)} train / {len(val_df)} val "
            f"({int(val_df['label'].sum())} maligne im Val) =="
        )

        probs, labels, fold_auroc, state = finetune_fold(
            fold, train_df, val_df, base_ckpt, cfg, device, args.epochs, args.lr
        )
        oof_probs[val_idx] = probs
        fold_reports.append({"fold": fold, "best_val_auroc": fold_auroc, "n_val": len(val_df)})
        torch.save(
            {
                "state_dict": state,
                "backbone": base_ckpt["backbone"],
                "image_size": base_ckpt["image_size"],
                "mean": cfg["data"]["mean"],
                "std": cfg["data"]["std"],
                "fold": fold,
                "val_auroc": fold_auroc,
            },
            run_dir / f"fold_{fold}.pt",
        )

    preds = (oof_probs >= cfg["inference"]["threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(oof_labels, preds, labels=[0, 1]).ravel()

    results = {
        "run_name": args.run_name,
        "base_run": args.base_run,
        "folds": args.folds,
        "epochs_per_fold": args.epochs,
        "lr": args.lr,
        "n": int(len(oof_labels)),
        "prevalence": float(oof_labels.mean()),
        "oof_auroc": float(roc_auc_score(oof_labels, oof_probs)),
        "oof_average_precision": float(average_precision_score(oof_labels, oof_probs)),
        "at_threshold": {
            "threshold": cfg["inference"]["threshold"],
            "sensitivity": float(tp / max(1, tp + fn)),
            "specificity": float(tn / max(1, tn + fp)),
            "precision": float(tp / max(1, tp + fp)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "operating_points": operating_points(oof_labels, oof_probs),
        "per_fold": fold_reports,
    }

    out = paths.reports / f"metrics_{args.run_name}_oof.json"
    paths.reports.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("\n" + json.dumps(results, indent=2))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
