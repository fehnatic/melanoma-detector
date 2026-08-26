"""Download der drei Kaggle-Quellen.

Besonderheit SIIM-ISIC: Der komplette Competition-Download waere ~108 GB, weil er
die DICOM-Originale enthaelt. Die brauchen wir nicht - fuer Transfer Learning
reichen die JPEGs. Deshalb laden wir zuerst train.csv, leiten daraus die noetigen
JPEG-Pfade ab und ziehen nur diese Dateien einzeln ueber die Kaggle-REST-API.
Das spart ~85 GB und ist resume-faehig (bereits vorhandene Dateien werden
uebersprungen).

Das SIIM-Testset laden wir standardmaessig NICHT - es ist unlabeled und damit
fuer uns wertlos.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

KAGGLE_API = "https://www.kaggle.com/api/v1"
SIIM_COMPETITION = "siim-isic-melanoma-classification"

DATASET_SLUGS = {
    "mittal": "bhaveshmittal/melanoma-cancer-dataset",
    "javed": "hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images",
}


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def kaggle_credentials() -> tuple[str, str]:
    """Liest Kaggle-Zugangsdaten aus der Umgebung oder ~/.kaggle/kaggle.json."""
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key

    cred_file = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not cred_file.exists():
        raise SystemExit(
            "Keine Kaggle-Credentials gefunden.\n"
            "  1. https://www.kaggle.com/settings -> API -> 'Create New Token'\n"
            "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json\n"
            "Ausserdem noetig: die SIIM-ISIC-Competition-Regeln einmal im Browser akzeptieren."
        )
    data = json.loads(cred_file.read_text())
    return data["username"], data["key"]


# --------------------------------------------------------------------------- #
# SIIM-ISIC
# --------------------------------------------------------------------------- #
def _download_competition_file(
    auth: HTTPBasicAuth,
    file_name: str,
    dest: Path,
    retries: int,
    timeout: int = 60,
) -> str:
    """Laedt eine einzelne Competition-Datei. Gibt 'ok' | 'skip' | 'fail:<grund>' zurueck."""
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"

    url = f"{KAGGLE_API}/competitions/data/download/{SIIM_COMPETITION}/{file_name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_error = "unbekannt"
    for attempt in range(retries):
        try:
            with requests.get(url, auth=auth, stream=True, timeout=timeout) as resp:
                if resp.status_code == 403:
                    return (
                        "fail:403 - Competition-Regeln nicht akzeptiert? "
                        "Einmal im Browser auf der Competition-Seite auf 'Join' klicken."
                    )
                if resp.status_code == 404:
                    return f"fail:404 - {file_name} existiert nicht"
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)

            # Kaggle liefert einzelne Dateien manchmal als ZIP aus.
            if zipfile.is_zipfile(tmp):
                with zipfile.ZipFile(tmp) as zf:
                    names = [n for n in zf.namelist() if not n.endswith("/")]
                    if len(names) == 1:
                        with zf.open(names[0]) as src, open(dest, "wb") as out:
                            shutil.copyfileobj(src, out)
                        tmp.unlink()
                        return "ok"
                    zf.extractall(dest.parent)
                tmp.unlink()
                return "ok"

            tmp.replace(dest)
            return "ok"
        except Exception as exc:  # noqa: BLE001 - Netzwerkfehler sind hier erwartbar
            last_error = f"{type(exc).__name__}: {exc}"

    tmp.unlink(missing_ok=True)
    return f"fail:{last_error}"


def download_siim(
    raw_dir: Path,
    mode: str = "jpeg",
    workers: int = 8,
    retries: int = 4,
    include_test_images: bool = False,
    limit: int | None = None,
) -> Path:
    """Laedt SIIM-ISIC nach raw_dir/siim. Gibt das Zielverzeichnis zurueck."""
    target = raw_dir / "siim"
    target.mkdir(parents=True, exist_ok=True)

    if mode == "full":
        print("[siim] Kompletter Competition-Download (~108 GB inkl. DICOM) ...")
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", SIIM_COMPETITION, "-p", str(target)],
            check=True,
        )
        for zpath in target.glob("*.zip"):
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(target)
            zpath.unlink()
        return target

    user, key = kaggle_credentials()
    auth = HTTPBasicAuth(user, key)

    # 1) CSVs
    for csv_name in ("train.csv", "test.csv"):
        status = _download_competition_file(auth, csv_name, target / csv_name, retries)
        if status.startswith("fail"):
            raise SystemExit(f"[siim] {csv_name} konnte nicht geladen werden -> {status}")
    print(f"[siim] CSVs unter {target}")

    # 2) Aus train.csv die benoetigten JPEG-Pfade ableiten
    wanted: list[tuple[str, Path]] = []
    train_df = pd.read_csv(target / "train.csv")
    for name in train_df["image_name"]:
        wanted.append((f"jpeg/train/{name}.jpg", target / "jpeg" / "train" / f"{name}.jpg"))
    if include_test_images:
        test_df = pd.read_csv(target / "test.csv")
        for name in test_df["image_name"]:
            wanted.append((f"jpeg/test/{name}.jpg", target / "jpeg" / "test" / f"{name}.jpg"))
    if limit:
        wanted = wanted[:limit]

    todo = [(rel, dst) for rel, dst in wanted if not (dst.exists() and dst.stat().st_size > 0)]
    print(f"[siim] {len(wanted)} Bilder insgesamt, {len(todo)} noch zu laden.")
    if not todo:
        return target

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_competition_file, auth, rel, dst, retries): rel
            for rel, dst in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="siim jpegs", unit="img"):
            status = fut.result()
            if status.startswith("fail"):
                failures.append(f"{futures[fut]} -> {status}")

    if failures:
        report = target / "download_failures.txt"
        report.write_text("\n".join(failures))
        print(
            f"[siim] WARNUNG: {len(failures)} Dateien fehlgeschlagen (Liste: {report}). "
            "Skript einfach nochmal starten - es setzt fort."
        )
    return target


# --------------------------------------------------------------------------- #
# Die beiden Ordner-Datensaetze
# --------------------------------------------------------------------------- #
def download_dataset(name: str, raw_dir: Path) -> Path:
    slug = DATASET_SLUGS[name]
    target = raw_dir / name
    if target.exists() and any(target.rglob("*.jpg")):
        print(f"[{name}] bereits vorhanden -> {target}")
        return target
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

    parser = argparse.ArgumentParser(description="Kaggle-Rohdaten herunterladen")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["siim", "mittal", "javed"],
        choices=["siim", "mittal", "javed"],
    )
    parser.add_argument("--siim-mode", default=None, choices=["jpeg", "full"])
    parser.add_argument("--include-siim-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="nur N SIIM-Bilder (Smoke-Test)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    paths.mkdirs()

    for source in args.sources:
        if source == "siim":
            download_siim(
                paths.raw,
                mode=args.siim_mode or cfg["download"]["siim_mode"],
                workers=cfg["download"]["workers"],
                retries=cfg["download"]["retries"],
                include_test_images=args.include_siim_test,
                limit=args.limit,
            )
        else:
            download_dataset(source, paths.raw)
    print("Fertig.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
