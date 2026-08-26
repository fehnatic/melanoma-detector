"""Deduplizierung ueber alle drei Quellen hinweg.

Warum das noetig ist: Alle drei Datensaetze stammen (direkt oder indirekt) aus dem
ISIC-Archiv. Datensatz 2 und 3 ueberschneiden sich stark miteinander, und beide
enthalten Bilder, die auch in SIIM-ISIC vorkommen. Wuerde man das naiv
zusammenwerfen und zufaellig splitten, landeten identische Bilder gleichzeitig in
Train und Val -> die Validierungsmetrik waere massiv zu optimistisch.

Verfahren:
  1. Perceptual Hash (pHash, 64 Bit) pro Bild - robust gegen Reskalierung und
     leichte JPEG-Artefakte, genau die Unterschiede zwischen den Quellen.
  2. LSH-Banding (8 Baender a 8 Bit) als Kandidatenfilter. Bei 8 Baendern teilt
     sich jedes Paar mit Hamming-Distanz <= 7 garantiert mindestens ein Band,
     der Filter verliert bei Schwellwert <= 5 also nichts.
  3. Exakte Hamming-Distanz auf den Kandidaten, dann Union-Find -> Cluster.
  3b. Verifikation gegen Verkettung. Union-Find ist Single-Linkage: aus
     A~B und B~C folgt A~C, auch wenn A und C weit auseinanderliegen. Auf
     kontrastarmen Bildern, wo pHash unzuverlaessig ist, entstehen dadurch
     Ketten von ueber hundert Bildern, die keine Duplikate sind (gemessen:
     nur 6.6% der Paare im groessten Cluster lagen unter dem Schwellwert).
     Deshalb wird jeder Cluster ab drei Bildern nachgeprueft: greedy wird
     jeweils das Bild mit den meisten Nachbarn innerhalb des Schwellwerts
     gewaehlt und nur dessen Umgebungskugel als Cluster akzeptiert. Der
     Durchmesser eines Clusters ist damit auf 2*max_hamming begrenzt statt
     unbegrenzt.
  4. Pro Cluster bleibt genau ein Bild uebrig. Prioritaet nach Quelle
     (Default: isic2020 > javed > mittal), weil ISIC als einzige Quelle
     Patienten-IDs mitbringt, die wir fuer den Split brauchen.

Zusaetzlich liefert das ISIC-Archiv eine offizielle Liste von 424 bekannten
Duplikatpaaren im Trainingsset. Die wird direkt eingespeist - sie ist
Ground Truth und braucht keinen Schwellwert.

Cluster mit widerspruechlichen Labels werden protokolliert und komplett
verworfen - ein Bild, das einmal als benigne und einmal als maligne annotiert
ist, ist als Trainingssignal wertlos.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

HASH_BITS = 64


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def _phash_one(args: tuple[str, int]) -> tuple[str, int | None]:
    """Gibt (image_path, hash als uint64) zurueck; None wenn das Bild kaputt ist."""
    path, hash_size = args
    try:
        import imagehash

        with Image.open(path) as img:
            value = imagehash.phash(img.convert("RGB"), hash_size=hash_size)
        bits = value.hash.flatten()
        out = 0
        for bit in bits:
            out = (out << 1) | int(bit)
        return path, out
    except Exception:  # noqa: BLE001 - kaputte Dateien sollen den Lauf nicht killen
        return path, None


def compute_hashes(paths: list[str], hash_size: int = 8, workers: int = 8) -> dict[str, int | None]:
    tasks = [(p, hash_size) for p in paths]

    def run(pool_cls) -> dict[str, int | None]:
        results: dict[str, int | None] = {}
        with pool_cls(max_workers=workers) as pool:
            for path, value in tqdm(
                pool.map(_phash_one, tasks, chunksize=64),
                total=len(tasks),
                desc="phash",
                unit="img",
            ):
                results[path] = value
        return results

    try:
        return run(ProcessPoolExecutor)
    except (BrokenProcessPool, RuntimeError):
        # macOS startet Prozesse per spawn, dafuer muss __main__ importierbar
        # sein. Aus einem Notebook, einer REPL oder einem Skript ohne
        # if __name__ == "__main__"-Guard ist das nicht gegeben. Dann eben
        # Threads: PIL gibt die GIL beim Dekodieren frei, das bleibt brauchbar.
        print("[dedup] Prozess-Pool nicht verfuegbar, weiche auf Threads aus")
        return run(ThreadPoolExecutor)


# --------------------------------------------------------------------------- #
# Kandidatensuche + Clustering
# --------------------------------------------------------------------------- #
def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return lut[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def _resolve_known_pairs(
    df: pd.DataFrame, known_pairs: list[tuple[str, str]] | None
) -> list[tuple[int, int]]:
    """Uebersetzt bekannte Duplikatpaare (image_id) in Zeilenindizes."""
    if not known_pairs:
        return []
    position = {image_id: i for i, image_id in enumerate(df["image_id"])}
    resolved = [
        (position[a], position[b])
        for a, b in known_pairs
        if a in position and b in position
    ]
    print(f"[dedup] {len(resolved)} von {len(known_pairs)} offiziellen Duplikatpaaren zugeordnet")
    return resolved


def _refine_clusters(
    hashes: np.ndarray, cluster_ids: np.ndarray, max_hamming: int
) -> np.ndarray:
    """Bricht durch Single-Linkage entstandene Ketten wieder auf.

    Fuer jeden Cluster ab drei Elementen wird wiederholt das Element mit den
    meisten Nachbarn innerhalb des Schwellwerts gewaehlt und dessen Umgebung
    als eigener Cluster herausgeloest, bis nichts mehr uebrig ist.
    """
    out = cluster_ids.copy()
    next_id = int(cluster_ids.max()) + 1 if len(cluster_ids) else 0
    split_count = 0

    by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(cluster_ids.tolist()):
        by_cluster[cid].append(idx)

    for cid, members in by_cluster.items():
        if len(members) < 3:
            continue
        values = hashes[np.asarray(members)]
        dist = np.array([_popcount(v ^ values) for v in values])
        close = dist <= max_hamming
        if close.all():
            continue  # alle gegenseitig aehnlich - nichts zu tun

        remaining = set(range(len(members)))
        first = True
        while remaining:
            order = sorted(remaining)
            mask = np.zeros(len(members), dtype=bool)
            mask[order] = True
            counts = (close & mask).sum(axis=1)
            counts[~mask] = -1
            center = int(counts.argmax())
            ball = [i for i in order if close[center, i]]

            target = cid if first else next_id
            if not first:
                next_id += 1
                split_count += 1
            for i in ball:
                out[members[i]] = target
            remaining -= set(ball)
            first = False

    if split_count:
        print(f"[dedup] {split_count} verkettete Cluster aufgebrochen")
    return out


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_by_hash(
    hashes: np.ndarray,
    max_hamming: int,
    bands: int = 8,
    extra_pairs: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Gibt fuer jedes Element die Cluster-ID zurueck.

    `extra_pairs` sind vorab bekannte Duplikate (Zeilenindizes), die
    unabhaengig vom Hash-Schwellwert zusammengefuehrt werden.
    """
    n = len(hashes)
    uf = _UnionFind(n)
    band_width = HASH_BITS // bands

    for band in range(bands):
        shift = band * band_width
        keys = (hashes >> np.uint64(shift)) & np.uint64((1 << band_width) - 1)
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, key in enumerate(keys.tolist()):
            buckets[key].append(idx)

        for members in buckets.values():
            if len(members) < 2:
                continue
            idx = np.asarray(members)
            values = hashes[idx]
            # Paarweise nur innerhalb des Buckets - obere Dreiecksmatrix.
            for i in range(len(idx) - 1):
                dist = _popcount(values[i] ^ values[i + 1 :])
                close = np.nonzero(dist <= max_hamming)[0]
                for j in close.tolist():
                    uf.union(int(idx[i]), int(idx[i + 1 + j]))

    clusters = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    clusters = _refine_clusters(hashes, clusters, max_hamming)

    # Bekannte Duplikate erst jetzt zusammenfuehren: sie sind Ground Truth und
    # duerfen von der Verifikation oben nicht wieder auseinandergerissen werden.
    if extra_pairs:
        uf2 = _UnionFind(n)
        by_cluster: dict[int, int] = {}
        for idx, cid in enumerate(clusters.tolist()):
            if cid in by_cluster:
                uf2.union(by_cluster[cid], idx)
            else:
                by_cluster[cid] = idx
        for a, b in extra_pairs:
            uf2.union(a, b)
        clusters = np.array([uf2.find(i) for i in range(n)], dtype=np.int64)

    return clusters


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def deduplicate(
    manifest: pd.DataFrame,
    hash_size: int = 8,
    max_hamming: int = 5,
    source_priority: list[str] | None = None,
    workers: int = 8,
    known_pairs: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gibt (bereinigtes Manifest, Duplikat-Report) zurueck."""
    source_priority = source_priority or ["isic2020", "javed", "mittal"]
    prio = {src: i for i, src in enumerate(source_priority)}

    df = manifest.copy().reset_index(drop=True)
    hashes = compute_hashes(df["image_path"].tolist(), hash_size=hash_size, workers=workers)
    df["phash"] = df["image_path"].map(hashes)

    broken = df["phash"].isna()
    if broken.any():
        print(f"[dedup] {int(broken.sum())} unlesbare Bilder entfernt")
    df = df[~broken].copy().reset_index(drop=True)
    df["phash"] = df["phash"].astype("uint64")

    values = df["phash"].to_numpy(dtype=np.uint64)
    print(f"[dedup] clustere {len(values)} Bilder (max_hamming={max_hamming}) ...")
    df["cluster_id"] = cluster_by_hash(
        values,
        max_hamming=max_hamming,
        extra_pairs=_resolve_known_pairs(df, known_pairs),
    )

    df["_prio"] = df["source"].map(lambda s: prio.get(s, len(prio)))
    grouped = df.groupby("cluster_id")

    # Cluster mit widerspruechlichen Labels komplett verwerfen.
    label_nunique = grouped["label"].transform("nunique")
    conflicted = df[label_nunique > 1].copy()
    if not conflicted.empty:
        print(
            f"[dedup] {conflicted['cluster_id'].nunique()} Cluster mit widerspruechlichen "
            f"Labels ({len(conflicted)} Bilder) verworfen"
        )
    clean = df[label_nunique == 1].copy()

    # Pro Cluster den Repraesentanten waehlen.
    clean = clean.sort_values(["cluster_id", "_prio", "image_id"], kind="stable")
    is_rep = ~clean.duplicated("cluster_id", keep="first")
    kept = clean[is_rep].copy()
    dropped = clean[~is_rep].copy()

    rep_by_cluster = kept.set_index("cluster_id")["image_id"]
    dropped["kept_instead"] = dropped["cluster_id"].map(rep_by_cluster)
    conflicted["kept_instead"] = "<verworfen: Label-Konflikt>"

    report_cols = ["image_id", "source", "label", "cluster_id", "kept_instead", "image_path"]
    report = pd.concat([dropped[report_cols], conflicted[report_cols]], ignore_index=True)

    kept = kept.drop(columns=["_prio", "phash"]).reset_index(drop=True)

    print(
        f"[dedup] {len(df)} -> {len(kept)} Bilder "
        f"({len(dropped)} Duplikate, {len(conflicted)} wegen Label-Konflikt)"
    )
    return kept, report


def load_isic_duplicate_pairs(raw_dir: Path) -> list[tuple[str, str]]:
    """Offizielle Duplikatliste des ISIC-Archivs, uebersetzt in image_id-Paare."""
    path = raw_dir / "isic2020" / "ISIC_2020_Training_Duplicates.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        (f"isic2020/{a}", f"isic2020/{b}")
        for a, b in zip(df["image_name_1"], df["image_name_2"])
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    from melanoma.config import load_config, resolve_paths

    parser = argparse.ArgumentParser(description="Duplikate ueber alle Quellen entfernen")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="nur den Duplikat-Report schreiben, manifest_dedup.csv nicht anfassen",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    paths.mkdirs()

    src = paths.interim / "manifest_raw.csv"
    if not src.exists():
        raise SystemExit(f"{src} fehlt - erst 'python -m melanoma.data.manifest' laufen lassen.")
    manifest = pd.read_csv(src)

    d = cfg["dedup"]
    kept, report = deduplicate(
        manifest,
        hash_size=d["hash_size"],
        max_hamming=d["max_hamming"],
        source_priority=d["source_priority"],
        workers=d["workers"],
        known_pairs=load_isic_duplicate_pairs(paths.raw),
    )

    report_path = paths.reports / "duplicates.csv"
    report.to_csv(report_path, index=False)
    print(f"[dedup] Report -> {report_path}")

    if not args.report_only:
        out = paths.interim / "manifest_dedup.csv"
        kept.to_csv(out, index=False)
        summary = kept.groupby("source")["label"].agg(["count", "sum", "mean"])
        summary.columns = ["bilder", "maligne", "anteil_maligne"]
        print("\n--- nach Deduplizierung ---")
        print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"\nGesamt: {len(kept)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
