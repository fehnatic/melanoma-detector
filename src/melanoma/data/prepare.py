"""Einmaliges Verkleinern der Bilder auf eine Arbeitsaufloesung.

Die ISIC-Originale sind bis zu 6000x4000 Pixel gross. Sie bei jeder Epoche neu
zu dekodieren kostet gemessen 36 ms pro Bild - bei 22.000 Bildern also vier
Minuten pro Epoche, in denen die GPU Daumen dreht. Auf 512 px vorverkleinert
sind es 6.6 ms, ein Faktor 10.

Bilder, die ohnehin schon kleiner als die Zielgroesse sind (javed mit 300 px,
mittal mit 224 px), werden nicht angefasst - das Manifest verweist dann
weiterhin auf das Original.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def _resize_one(args: tuple[str, str, int, int]) -> tuple[str, str]:
    """Gibt (image_id, effektiver Pfad) zurueck."""
    image_id, src, dest, max_side, quality = args[0], args[1], args[2], args[3], args[4]
    dest_path = Path(dest)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return image_id, dest

    try:
        with Image.open(src) as img:
            if max(img.size) <= max_side:
                return image_id, src  # schon klein genug, Original behalten
            img = img.convert("RGB")
            # draft() laesst libjpeg direkt verkleinert dekodieren - deutlich
            # schneller, als erst voll zu dekodieren und dann zu skalieren.
            img.draft("RGB", (max_side, max_side))
            img.thumbnail((max_side, max_side), Image.LANCZOS)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest_path.with_suffix(".tmp.jpg")
            img.save(tmp, "JPEG", quality=quality, optimize=True)
            tmp.replace(dest_path)
        return image_id, dest
    except Exception as exc:  # noqa: BLE001
        print(f"[prepare] {image_id}: {type(exc).__name__}: {exc}")
        return image_id, src


def prepare(
    manifest: pd.DataFrame,
    cache_dir: Path,
    max_side: int = 512,
    quality: int = 92,
    workers: int = 8,
) -> pd.DataFrame:
    tasks = []
    for row in manifest.itertuples():
        dest = cache_dir / row.source / f"{Path(row.image_path).stem}.jpg"
        tasks.append((row.image_id, row.image_path, str(dest), max_side, quality))

    resolved: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for image_id, path in tqdm(
            pool.map(_resize_one, tasks, chunksize=32),
            total=len(tasks),
            desc=f"resize -> {max_side}px",
            unit="img",
        ):
            resolved[image_id] = path

    out = manifest.copy()
    out["image_path_original"] = out["image_path"]
    out["image_path"] = out["image_id"].map(resolved)
    changed = int((out["image_path"] != out["image_path_original"]).sum())
    print(f"[prepare] {changed} Bilder verkleinert, {len(out) - changed} unveraendert uebernommen")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from melanoma.config import load_config, resolve_paths

    parser = argparse.ArgumentParser(description="Bilder einmalig auf Arbeitsaufloesung bringen")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-side", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    paths.mkdirs()

    src = paths.interim / "manifest_dedup.csv"
    if not src.exists():
        raise SystemExit(f"{src} fehlt - erst 'python -m melanoma.data.dedup' laufen lassen.")

    max_side = args.max_side or cfg["data"].get("cache_max_side", 512)
    out = prepare(
        pd.read_csv(src),
        cache_dir=paths.processed / "images",
        max_side=max_side,
        workers=args.workers,
    )
    dest = paths.interim / "manifest_prepared.csv"
    out.to_csv(dest, index=False)
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
