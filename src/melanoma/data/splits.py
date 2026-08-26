"""Gruppenweiser, stratifizierter Train/Val/Test-Split.

Zwei Fallstricke, die hier bewusst adressiert werden:

  * ISIC 2020 enthaelt mehrere Aufnahmen pro Patient. Landen Bilder desselben
    Patienten in Train und Val, misst die Validierung teilweise
    Patienten-Wiedererkennung statt Melanom-Erkennung. Deshalb wird nach
    patient_id gruppiert.
  * Das Label ist stark unbalanciert (ISIC 2020 ~1.8% maligne). Ohne
    Stratifizierung schwankt der Anteil maligner Faelle zwischen den Splits
    so stark, dass die Metriken nicht mehr vergleichbar sind.

sklearn's StratifiedGroupKFold erfuellt beides gleichzeitig.
"""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _holdout(df: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.Index, pd.Index]:
    """Zieht ungefaehr `fraction` der Zeilen als Holdout heraus."""
    n_splits = max(2, round(1.0 / fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, hold_idx = next(splitter.split(df, y=df["label"], groups=df["patient_id"]))
    return df.index[rest_idx], df.index[hold_idx]


def make_splits(
    manifest: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    df = manifest.copy().reset_index(drop=True)

    rest_idx, test_idx = _holdout(df, test_fraction, seed)
    rest = df.loc[rest_idx]

    # val_fraction bezieht sich auf den Gesamtdatensatz, gezogen wird aber
    # aus dem Rest - deshalb hochrechnen.
    val_share_of_rest = val_fraction / (1.0 - test_fraction)
    train_idx, val_idx = _holdout(rest, val_share_of_rest, seed)

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Sicherheitsnetz: keine Patientengruppe darf in zwei Splits auftauchen.
    overlap = df.groupby("patient_id")["split"].nunique()
    leaking = overlap[overlap > 1]
    if len(leaking):
        raise SystemExit(f"Split-Leak: {len(leaking)} Gruppen in mehreren Splits.")

    return df


def main(argv: list[str] | None = None) -> int:
    import argparse

    from melanoma.config import load_config, resolve_paths

    parser = argparse.ArgumentParser(description="Train/Val/Test-Split erzeugen")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    paths.mkdirs()

    src = paths.interim / "manifest_dedup.csv"
    if not src.exists():
        raise SystemExit(f"{src} fehlt - erst 'python -m melanoma.data.dedup' laufen lassen.")

    s = cfg["split"]
    df = make_splits(
        pd.read_csv(src),
        val_fraction=s["val_fraction"],
        test_fraction=s["test_fraction"],
        seed=s["seed"],
    )

    out = paths.processed / "splits.csv"
    df.to_csv(out, index=False)

    summary = df.groupby("split")["label"].agg(["count", "sum", "mean"])
    summary.columns = ["bilder", "maligne", "anteil_maligne"]
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
