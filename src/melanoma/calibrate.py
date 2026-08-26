"""Kalibrierung der Ausgabewahrscheinlichkeit.

Zwei getrennte Probleme, die hier geloest werden:

1. Rohe Sigmoid-Ausgaben eines neuronalen Netzes sind keine ehrlichen
   Wahrscheinlichkeiten - sie sind typischerweise ueberkonfident. Isotonic
   Regression bzw. Temperature Scaling auf dem Validierungsset korrigiert das.

2. Beim Training wird die Minderheitsklasse ueberabgetastet (Sampler), das Modell
   sieht also faktisch ~50% maligne Faelle. Die kalibrierte Wahrscheinlichkeit
   bezieht sich danach auf die Praevalenz des Validierungssets. Wer sie auf eine
   andere Praevalenz beziehen will, kann sie ueber `target_prevalence`
   umrechnen (Prior-Shift auf den Odds).

Wichtig fuer die Interpretation: Die ausgegebene Zahl ist eine
Wahrscheinlichkeit BEZOGEN AUF DIE DATENVERTEILUNG DIESER DATENSAETZE, nicht
bezogen auf die Melanom-Praevalenz in der Allgemeinbevoelkerung.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def prior_shift(probs: np.ndarray, source_prevalence: float, target_prevalence: float) -> np.ndarray:
    """Rechnet Wahrscheinlichkeiten von einer Praevalenz auf eine andere um."""
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    odds = p / (1 - p)
    factor = (target_prevalence / (1 - target_prevalence)) * (
        (1 - source_prevalence) / source_prevalence
    )
    shifted = odds * factor
    return shifted / (1 + shifted)


class Calibrator:
    """Bildet rohe Modellwahrscheinlichkeiten auf kalibrierte ab.

    Bewusst ohne pickle serialisiert: das Objekt wird als JSON abgelegt
    (Isotonic = Stuetzstellen, Temperature = ein Skalar). Das haelt die
    Artefakte lesbar, versionsstabil und macht das Laden ungefaehrlich.
    """

    def __init__(
        self,
        method: str,
        params: dict | None = None,
        prevalence: float = 0.5,
    ) -> None:
        self.method = method
        self.params = params or {}
        self.prevalence = prevalence  # Praevalenz, auf die sich die Ausgabe bezieht

    @classmethod
    def from_isotonic(cls, iso, prevalence: float) -> "Calibrator":
        return cls(
            "isotonic",
            {"x": np.asarray(iso.X_thresholds_).tolist(), "y": np.asarray(iso.y_thresholds_).tolist()},
            prevalence,
        )

    @classmethod
    def from_temperature(cls, temperature: float, prevalence: float) -> "Calibrator":
        return cls("temperature", {"temperature": float(temperature)}, prevalence)

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        if self.method == "none":
            return probs
        if self.method == "isotonic":
            x = np.asarray(self.params["x"], dtype=np.float64)
            y = np.asarray(self.params["y"], dtype=np.float64)
            # np.interp klemmt ausserhalb des Stuetzstellenbereichs auf die
            # Randwerte - identisch zu IsotonicRegression(out_of_bounds="clip").
            return np.clip(np.interp(probs, x, y), 0.0, 1.0)
        if self.method == "temperature":
            eps = 1e-6
            p = np.clip(probs, eps, 1 - eps)
            logits = np.log(p / (1 - p))
            return 1.0 / (1.0 + np.exp(-logits / self.params["temperature"]))
        raise ValueError(f"unbekannte Methode: {self.method}")

    def to_dict(self) -> dict:
        return {"method": self.method, "params": self.params, "prevalence": self.prevalence}

    @classmethod
    def from_dict(cls, data: dict) -> "Calibrator":
        return cls(data["method"], data.get("params"), data.get("prevalence", 0.5))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def load_calibrator(run_dir: Path) -> Calibrator | None:
    """Laedt den Kalibrator eines Runs, oder None wenn keiner existiert."""
    path = Path(run_dir) / "calibrator.json"
    if not path.exists():
        return None
    return Calibrator.from_dict(json.loads(path.read_text()))


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Sucht das Temperatur-T, das die NLL minimiert (einfacher 1D-Grid + Verfeinerung)."""
    eps = 1e-6
    logits = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))

    def nll(temp: float) -> float:
        p = 1.0 / (1.0 + np.exp(-logits / temp))
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    grid = np.linspace(0.25, 6.0, 116)
    best = min(grid, key=nll)
    fine = np.linspace(max(0.05, best - 0.1), best + 0.1, 41)
    return float(min(fine, key=nll))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


def main(argv: list[str] | None = None) -> int:
    from sklearn.isotonic import IsotonicRegression

    from melanoma.config import load_config, resolve_paths

    parser = argparse.ArgumentParser(description="Wahrscheinlichkeiten kalibrieren")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--method", default=None, choices=["isotonic", "temperature", "none"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    run_dir = paths.artifacts / args.run_name

    pred_file = run_dir / "val_predictions.npz"
    if not pred_file.exists():
        raise SystemExit(f"{pred_file} fehlt - erst trainieren.")

    data = np.load(pred_file)
    probs, labels = data["probs"].astype(np.float64), data["labels"].astype(np.float64)
    method = args.method or cfg["calibration"]["method"]
    val_prevalence = float(labels.mean())

    before = expected_calibration_error(probs, labels)

    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(probs, labels)
        calibrator = Calibrator.from_isotonic(iso, val_prevalence)
    elif method == "temperature":
        calibrator = Calibrator.from_temperature(fit_temperature(probs, labels), val_prevalence)
    else:
        calibrator = Calibrator("none", None, val_prevalence)

    after = expected_calibration_error(calibrator(probs), labels)

    out = run_dir / "calibrator.json"
    calibrator.save(out)

    info = {
        "method": method,
        "val_prevalence": val_prevalence,
        "ece_before": before,
        "ece_after": after,
        "n_val": int(len(labels)),
    }
    (run_dir / "calibration.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
