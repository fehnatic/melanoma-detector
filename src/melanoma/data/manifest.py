"""Baut aus den drei Rohquellen ein einheitliches Manifest.

Ergebnis: data/interim/manifest_raw.csv mit den Spalten

    image_path   absoluter Pfad zum Bild
    source       isic2020 | mittal | javed
    label        0 = benigne, 1 = maligne
    patient_id   nur ISIC 2020 hat echte Patienten-IDs. Fuer die anderen Quellen
                 bekommt jedes Bild seine eigene Gruppe, damit der spaetere
                 gruppenweise Split nicht faelschlich Bilder zusammenzieht.
    orig_split   urspruengliche Zuordnung der Quelle (train/test/unknown)
    image_id     eindeutiger Schluessel "<source>/<dateiname ohne Endung>"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# Ordnernamen, wie sie in den beiden Ordner-Datensaetzen vorkommen.
BENIGN_DIRS = {"benign", "nevus", "not_melanoma", "notmelanoma"}
MALIGNANT_DIRS = {"malignant", "melanoma"}


def _iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def build_isic2020(raw_dir: Path) -> pd.DataFrame:
    """ISIC 2020: Labels, Patienten- und Laesions-IDs kommen aus der Ground-Truth-CSV."""
    root = raw_dir / "isic2020"
    csv_path = root / "ISIC_2020_Training_GroundTruth_v2.csv"
    if not csv_path.exists():
        print(f"[manifest] isic2020 uebersprungen (kein {csv_path})")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Die ZIP-Struktur ist nicht garantiert, deshalb einmal alle JPEGs indexieren
    # und ueber den Dateinamen zuordnen.
    index = {path.stem: path for path in (root / "images").rglob("*.jpg")}
    df["image_path"] = df["image_name"].map(lambda n: str(index[n]) if n in index else None)

    missing = int(df["image_path"].isna().sum())
    if missing:
        print(f"[manifest] isic2020: {missing} in der CSV gelistete Bilder fehlen lokal -> ignoriert")
    df = df.dropna(subset=["image_path"]).copy()

    out = pd.DataFrame(
        {
            "image_path": df["image_path"],
            "source": "isic2020",
            "label": df["target"].astype(int),
            "patient_id": "isic2020:" + df["patient_id"].astype(str),
            "orig_split": "train",
            "image_id": "isic2020/" + df["image_name"].astype(str),
        }
    )
    print(f"[manifest] isic2020: {len(out)} Bilder ({int(out.label.sum())} maligne)")
    return out


def build_folder_dataset(raw_dir: Path, source: str) -> pd.DataFrame:
    """Ordner-Datensaetze: Label steckt im Namen des Elternordners."""
    root = raw_dir / source
    if not root.exists():
        print(f"[manifest] {source} uebersprungen (kein {root})")
        return pd.DataFrame()

    rows = []
    unknown_dirs: set[str] = set()
    for path in _iter_images(root):
        parent = path.parent.name.lower().replace(" ", "_").replace("-", "_")
        if parent in BENIGN_DIRS:
            label = 0
        elif parent in MALIGNANT_DIRS:
            label = 1
        else:
            unknown_dirs.add(str(path.parent.relative_to(root)))
            continue

        grandparent = path.parent.parent.name.lower()
        orig_split = grandparent if grandparent in {"train", "test", "val", "valid"} else "unknown"
        image_id = f"{source}/{path.parent.parent.name}_{parent}_{path.stem}"
        rows.append(
            {
                "image_path": str(path),
                "source": source,
                "label": label,
                "patient_id": f"{source}:{image_id}",
                "orig_split": orig_split,
                "image_id": image_id,
            }
        )

    if unknown_dirs:
        preview = ", ".join(sorted(unknown_dirs)[:5])
        print(
            f"[manifest] {source}: {len(unknown_dirs)} Ordner ohne erkennbares Label "
            f"ignoriert (z.B. {preview}). Falls das echte Klassenordner sind, "
            "BENIGN_DIRS/MALIGNANT_DIRS in manifest.py ergaenzen."
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        print(f"[manifest] {source}: {len(out)} Bilder ({int(out.label.sum())} maligne)")
    return out


def build_manifest(raw_dir: Path) -> pd.DataFrame:
    frames = [
        build_isic2020(raw_dir),
        build_folder_dataset(raw_dir, "mittal"),
        build_folder_dataset(raw_dir, "javed"),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit(
            "Keine Bilder gefunden. Erst 'python -m melanoma.data.download' laufen lassen."
        )
    manifest = pd.concat(frames, ignore_index=True)

    dupe_ids = manifest["image_id"].duplicated().sum()
    if dupe_ids:
        raise SystemExit(f"{dupe_ids} doppelte image_id - das darf nicht passieren.")
    return manifest


def main(argv: list[str] | None = None) -> int:
    import argparse

    from melanoma.config import load_config, resolve_paths

    parser = argparse.ArgumentParser(description="Einheitliches Manifest bauen")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    paths.mkdirs()

    manifest = build_manifest(paths.raw)
    out = paths.interim / "manifest_raw.csv"
    manifest.to_csv(out, index=False)

    print("\n--- Zusammenfassung ---")
    summary = manifest.groupby("source")["label"].agg(["count", "sum", "mean"])
    summary.columns = ["bilder", "maligne", "anteil_maligne"]
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nGesamt: {len(manifest)} Bilder -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
