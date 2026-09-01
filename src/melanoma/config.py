"""Laden der YAML-Konfiguration und Aufloesen der Projektpfade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict mit Attributzugriff, damit cfg.train.epochs statt cfg["train"]["epochs"] geht."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    artifacts: Path
    reports: Path
    tracking: Path

    def mkdirs(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.artifacts, self.reports, self.tracking):
            p.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def resolve_paths(cfg: Config, root: Path | None = None) -> Paths:
    root = root or REPO_ROOT
    p = cfg["paths"]
    return Paths(
        root=root,
        raw=root / p["raw"],
        interim=root / p["interim"],
        processed=root / p["processed"],
        artifacts=root / p["artifacts"],
        reports=root / p["reports"],
        tracking=root / p["tracking"],
    )
