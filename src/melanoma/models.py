"""Modell-Factory und Geraetewahl."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(backbone: str, pretrained: bool = True, drop_rate: float = 0.3) -> nn.Module:
    """Ein Logit-Ausgang - binaere Klassifikation via BCEWithLogitsLoss."""
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=1,
        drop_rate=drop_rate,
    )


class ModelEma:
    """Exponentiell gemitteltes Schattenmodell. Stabilisiert die Validierungs-
    metrik spuerbar, wenn wie hier mit kleinen Batches trainiert wird."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        import copy

        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            else:
                ema_p.copy_(p)
