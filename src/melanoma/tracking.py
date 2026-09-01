"""Persoenlicher Verlauf: dasselbe Muttermal ueber die Zeit fotografieren und
vergleichen (die "Evolving"-Komponente der ABCDE-Regel).

Bewusst OHNE automatische Groessen- oder Farb-Delta-Berechnung: ohne fixen
Kameraabstand und Referenzmassstab waere eine "Flaeche +12%"-Aussage
vorgetaeuschte Praezision. Stattdessen zwei ehrliche Signale:

  * die Fotos chronologisch nebeneinander, zum eigenen visuellen Vergleich
  * die Modell-Wahrscheinlichkeit ueber die Zeit - ein konsistent gemessenes
    Signal, weil dasselbe Modell auf dieselbe Weise misst

Speichert lokal unter data/tracking/ - nie im Git-Repo (siehe .gitignore),
das sind persoenliche Gesundheitsdaten.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from PIL import Image

INDEX_FILE = "index.json"


_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.strip().lower().translate(_UMLAUTS)).strip("-")
    return slug or "unbenannt"


def _load_index(root: Path) -> dict:
    path = root / INDEX_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_index(root: Path, index: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / INDEX_FILE).write_text(json.dumps(index, indent=2, ensure_ascii=False))


def list_labels(root: Path) -> list[str]:
    return sorted(_load_index(root).keys())


def get_history(root: Path, label: str) -> list[dict]:
    """Chronologisch sortierte Eintraege fuer ein Muttermal."""
    return sorted(_load_index(root).get(label, []), key=lambda e: e["timestamp"])


def add_entry(root: Path, label: str, image: Image.Image, prob_malignant: float, source: str) -> dict:
    label = label.strip()
    slug = slugify(label)
    photos_dir = root / slug
    photos_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{timestamp}.jpg"
    image.convert("RGB").save(photos_dir / filename, "JPEG", quality=92)

    entry = {
        "timestamp": timestamp,
        "image_path": str(photos_dir / filename),
        "prob_malignant": float(prob_malignant),
        "source": source,
    }

    index = _load_index(root)
    index.setdefault(label, []).append(entry)
    _save_index(root, index)
    return entry
