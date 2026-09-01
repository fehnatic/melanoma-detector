"""Download der Rohdaten.

Zwei verschiedene Bezugswege, aus gutem Grund:

* Der SIIM-ISIC-Datensatz wird NICHT ueber Kaggle geladen, sondern direkt vom
  ISIC-Archiv - der Originalquelle. Vorteile: kein Kaggle-Account noetig, kein
  Beitritt zur Competition, und die Ground-Truth-Datei ist dort in einer
  neueren Fassung (v2) verfuegbar, die zusaetzlich `lesion_id` enthaelt. Ausserdem
  liefert das Archiv eine offizielle Liste bekannter Duplikate mit.
  Lizenz: CC-BY-NC 4.0, Zitationspflicht - siehe CITATION.md.

* Die beiden anderen Datensaetze liegen auf Kaggle, sind dort aber oeffentlich
  freigegeben und lassen sich ohne Account und ohne API-Token ueber den
  Download-Endpunkt ziehen (verifiziert: 99 MB bzw. 79 MB, HTTP 206 ohne
  Authorization-Header). Sollte Kaggle das irgendwann dichtmachen, siehe den
  Hinweis in download_kaggle_dataset().

* PAD-UFES-20 (klinische Handyfotos, fuer die Cross-Domain-Auswertung in
  melanoma.eval_external) liegt ebenfalls im ISIC-Archiv, als Collection 406 -
  gleicher Zugriffsweg wie ISIC 2020, kein Account noetig. Lizenz: CC-BY.

Ergebnis: Das gesamte Projekt braucht keinerlei Zugangsdaten.
"""

from __future__ import annotations

import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ISIC_S3 = "https://isic-challenge-data.s3.amazonaws.com/2020"
ISIC_FILES = {
    "images": "ISIC_2020_Training_JPEG.zip",
    "ground_truth": "ISIC_2020_Training_GroundTruth_v2.csv",
    "duplicates": "ISIC_2020_Training_Duplicates.csv",
}

KAGGLE_DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download"
DATASET_SLUGS = {
    "mittal": "bhaveshmittal/melanoma-cancer-dataset",
    "javed": "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images",
}

# ISIC-Archiv-Collection-ID von PAD-UFES-20 (https://api.isic-archive.com/collections/406/).
ISIC_API = "https://api.isic-archive.com/api/v2"
PAD_UFES20_COLLECTION = 406


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def remote_size(url: str) -> int | None:
    """Groesse einer Datei ermitteln, ohne sie zu laden."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=30)
        if resp.ok:
            size = int(resp.headers.get("content-length", 0))
            if size:
                return size
    except Exception:  # noqa: BLE001
        pass
    try:
        # Kaggle beantwortet HEAD mit 404, liefert aber auf einen Range-GET
        # brav einen Content-Range-Header mit der Gesamtgroesse.
        resp = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=30)
        resp.close()
        content_range = resp.headers.get("content-range", "")
        if "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
    except Exception:  # noqa: BLE001
        pass
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
def download_kaggle_dataset(name: str, raw_dir: Path, dry_run: bool = False) -> Path:
    """Laedt einen oeffentlichen Kaggle-Datensatz ohne Account.

    Falls Kaggle den offenen Zugriff eines Tages sperrt, meldet sich das hier
    als HTTP 401/403. Ausweg waere dann ein kostenloser API-Token
    (https://www.kaggle.com/settings -> API -> Create New Token, ablegen unter
    ~/.kaggle/kaggle.json) plus `pip install kaggle` und
    `kaggle datasets download -d <slug> --unzip`.
    """
    slug = DATASET_SLUGS[name]
    url = f"{KAGGLE_DOWNLOAD}/{slug}"
    target = raw_dir / name

    if dry_run:
        size = remote_size(url)
        print(f"{name} ({slug}):")
        print(f"  {'ZIP':<45} {_human(size) if size else '?'}")
        return target

    if target.exists() and any(target.rglob("*.jpg")):
        print(f"[{name}] bereits vorhanden -> {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] {slug} (oeffentlich, kein Account noetig):")
    zip_path = target / f"{name}.zip"
    _fetch(url, zip_path)

    print("  entpacke ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    print(f"  {sum(1 for _ in target.rglob('*.jpg'))} Bilder entpackt")
    zip_path.unlink()
    return target


# --------------------------------------------------------------------------- #
# PAD-UFES-20 (klinische Handyfotos, ISIC-Archiv-Mirror)
# --------------------------------------------------------------------------- #
def _pad_ufes20_records() -> list[dict]:
    """Paginiert durch die ISIC-API und sammelt alle Bild-Metadaten der Collection."""
    records: list[dict] = []
    url = f"{ISIC_API}/images/search/?collections={PAD_UFES20_COLLECTION}&limit=100"
    while url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        page = resp.json()
        records.extend(page["results"])
        url = page.get("next")
    return records


def download_pad_ufes20(raw_dir: Path, dry_run: bool = False, workers: int = 8) -> Path:
    target = raw_dir / "pad_ufes_20"
    records = _pad_ufes20_records()
    total_size = sum(r["files"]["full"]["size"] for r in records)

    if dry_run:
        print("PAD-UFES-20 (ISIC-Archiv Collection 406, klinische Handyfotos, CC-BY):")
        print(f"  {len(records)} Bilder, {_human(total_size)}")
        return target

    images_dir = target / "images"
    gt_path = target / "ground_truth.csv"
    if gt_path.exists() and images_dir.exists() and len(list(images_dir.glob("*.jpg"))) == len(records):
        print(f"[pad_ufes_20] bereits vollstaendig -> {target}")
        return target

    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"PAD-UFES-20: lade {len(records)} Bilder ({_human(total_size)}) ...")

    def _one(rec: dict) -> dict | None:
        image_id = rec["isic_id"]
        info = rec["files"]["full"]
        dest = images_dir / f"{image_id}.jpg"
        if not (dest.exists() and dest.stat().st_size == info["size"]):
            # Bei 2298 Einzelverbindungen reisst hin und wieder eine ab (verifiziert:
            # ConnectionResetError) - ein paar Versuche abfedern das, statt den
            # gesamten Lauf abzubrechen und die bereits geladenen Bilder zu verwerfen.
            for attempt in range(3):
                try:
                    resp = requests.get(info["url"], timeout=60)
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
                    break
                except requests.exceptions.RequestException as exc:
                    if attempt == 2:
                        print(f"[pad_ufes_20] {image_id}: {exc} - uebersprungen")
                        return None
                    time.sleep(2 * (attempt + 1))

        clinical = rec["metadata"].get("clinical", {})
        diagnosis = clinical.get("diagnosis_3") or clinical.get("diagnosis_2") or clinical.get("diagnosis_1")
        return {
            "image_name": image_id,
            "diagnosis": diagnosis,
            "patient_id": clinical.get("patient_id", image_id),
            "lesion_id": clinical.get("lesion_id", image_id),
            "anatom_site": clinical.get("anatom_site_1"),
            "age_approx": clinical.get("age_approx"),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(tqdm(pool.map(_one, records), total=len(records), desc="pad_ufes_20", unit="img"))

    failed = rows.count(None)
    rows = [r for r in rows if r is not None]
    pd.DataFrame(rows).to_csv(gt_path, index=False)
    if failed:
        print(f"  {failed} Bilder trotz Retries fehlgeschlagen - erneuter Aufruf laedt nur die fehlenden nach")
    print(f"  fertig: {len(rows)} Bilder -> {target}")
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
        choices=["isic2020", "mittal", "javed", "pad_ufes_20"],
    )
    parser.add_argument("--keep-zip", action="store_true", help="ISIC-ZIP nach dem Entpacken behalten")
    parser.add_argument("--dry-run", action="store_true", help="nur Groessen anzeigen, nichts laden")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    paths.mkdirs()

    for source in args.sources:
        if source == "isic2020":
            download_isic2020(paths.raw, keep_zip=args.keep_zip, dry_run=args.dry_run)
        elif source == "pad_ufes_20":
            download_pad_ufes20(paths.raw, dry_run=args.dry_run)
        else:
            download_kaggle_dataset(source, paths.raw, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nFertig. Naechster Schritt: make data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
