"""Download der Rohdaten.

Zwei verschiedene Bezugswege, aus gutem Grund:

* Der SIIM-ISIC-Datensatz wird NICHT ueber Kaggle geladen, sondern direkt vom
  ISIC-Archiv - der Originalquelle. Vorteile: kein Kaggle-Account noetig, kein
  Beitritt zur Competition, und die Ground-Truth-Datei ist dort in einer
  neueren Fassung (v2) verfuegbar, die zusaetzlich `lesion_id` enthaelt. Ausserdem
  liefert das Archiv eine offizielle Liste bekannter Duplikate mit.
  Lizenz: CC-BY-NC 4.0, Zitationspflicht - siehe CITATION.md.

* Die beiden anderen Datensaetze gibt es nur auf Kaggle. Dafuer wird ein
  API-Token gebraucht (kostenlos), aber keinerlei Competition-Zustimmung.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

ISIC_S3 = "https://isic-challenge-data.s3.amazonaws.com/2020"
ISIC_FILES = {
    "images": "ISIC_2020_Training_JPEG.zip",
    "ground_truth": "ISIC_2020_Training_GroundTruth_v2.csv",
    "duplicates": "ISIC_2020_Training_Duplicates.csv",
}

DATASET_SLUGS = {
    "mittal": "bhaveshmittal/melanoma-cancer-dataset",
    "javed": "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images",
}

KAGGLE_SOURCES = set(DATASET_SLUGS)


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def remote_size(url: str) -> int | None:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        return int(resp.headers.get("content-length", 0)) or None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# ISIC 2020 (= SIIM-ISIC Competition-Daten, aber von der Originalquelle)
# --------------------------------------------------------------------------- #
def _fetch(url: str, dest: Path) -> None:
    """Laedt eine Datei; bricht der Download ab, setzt der naechste Aufruf fort."""
    expected = remote_size(url)
    if dest.exists() and expected and dest.stat().st_size == expected:
        print(f"  {dest.name}: bereits vollstaendig ({_human(expected)})")
        return

    print(f"  {dest.name}: lade {_human(expected) if expected else 'unbekannte Groesse'} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # curl statt requests: liefert eine Fortschrittsanzeige und kann mit "-C -"
    # einen abgebrochenen Download fortsetzen - bei 23 GB nicht unwichtig.
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5", "-C", "-",
         "--progress-bar", "-o", str(dest), url],
        check=True,
    )
    if expected and dest.stat().st_size != expected:
        raise SystemExit(
            f"{dest.name}: Groesse stimmt nicht ({dest.stat().st_size} statt {expected}). "
            "Datei loeschen und erneut versuchen."
        )


def download_isic2020(raw_dir: Path, keep_zip: bool = False, dry_run: bool = False) -> Path:
    target = raw_dir / "isic2020"
    target.mkdir(parents=True, exist_ok=True)

    if dry_run:
        total = 0
        print("ISIC 2020 (direkt vom ISIC-Archiv, kein Account noetig):")
        for name in ISIC_FILES.values():
            size = remote_size(f"{ISIC_S3}/{name}")
            total += size or 0
            print(f"  {name:<45} {_human(size) if size else '?'}")
        print(f"  {'Summe':<45} {_human(total)} (plus ebenso viel beim Entpacken)")
        return target

    print("ISIC 2020 (direkt vom ISIC-Archiv, kein Account noetig):")
    for key in ("ground_truth", "duplicates"):
        _fetch(f"{ISIC_S3}/{ISIC_FILES[key]}", target / ISIC_FILES[key])

    images_dir = target / "images"
    if images_dir.exists() and any(images_dir.rglob("*.jpg")):
        print(f"  Bilder bereits entpackt unter {images_dir}")
        return target

    zip_path = target / ISIC_FILES["images"]
    _fetch(f"{ISIC_S3}/{ISIC_FILES['images']}", zip_path)

    print("  entpacke ... (dauert einige Minuten)")
    images_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(images_dir)
    count = sum(1 for _ in images_dir.rglob("*.jpg"))
    print(f"  {count} Bilder entpackt")

    if not keep_zip:
        zip_path.unlink()
        print("  ZIP geloescht (mit --keep-zip behalten)")
    return target


# --------------------------------------------------------------------------- #
# Kaggle-Datensaetze
# --------------------------------------------------------------------------- #
def check_kaggle_credentials() -> None:
    import os

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    cred = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if cred.exists():
        return
    raise SystemExit(
        "Fuer die Kaggle-Datensaetze fehlt ein API-Token (kostenlos, kein Competition-Beitritt):\n"
        "  1. https://www.kaggle.com/settings -> Abschnitt 'API' -> 'Create New Token'\n"
        "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
        "  3. chmod 600 ~/.kaggle/kaggle.json\n\n"
        "Nur ISIC laden (ganz ohne Kaggle):\n"
        "  python -m melanoma.data.download --sources isic2020"
    )


def download_kaggle_dataset(name: str, raw_dir: Path) -> Path:
    slug = DATASET_SLUGS[name]
    target = raw_dir / name
    if target.exists() and any(target.rglob("*.jpg")):
        print(f"[{name}] bereits vorhanden -> {target}")
        return target

    if shutil.which("kaggle") is None:
        raise SystemExit("Das 'kaggle'-Kommando fehlt. Im venv: pip install kaggle")

    target.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] lade {slug} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"],
        check=True,
    )
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse

    from melanoma.config import load_config, resolve_paths

    parser = argparse.ArgumentParser(description="Rohdaten herunterladen")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["isic2020", "mittal", "javed"],
        choices=["isic2020", "mittal", "javed"],
    )
    parser.add_argument("--keep-zip", action="store_true", help="ISIC-ZIP nach dem Entpacken behalten")
    parser.add_argument("--dry-run", action="store_true", help="nur Groessen anzeigen, nichts laden")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    paths.mkdirs()

    if not args.dry_run and (set(args.sources) & KAGGLE_SOURCES):
        check_kaggle_credentials()

    for source in args.sources:
        if source == "isic2020":
            download_isic2020(paths.raw, keep_zip=args.keep_zip, dry_run=args.dry_run)
        elif not args.dry_run:
            download_kaggle_dataset(source, paths.raw)
        else:
            print(f"{source}: Kaggle-Datensatz {DATASET_SLUGS[source]} (Groesse erst beim Laden bekannt)")

    if not args.dry_run:
        print("\nFertig. Naechster Schritt: make data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
