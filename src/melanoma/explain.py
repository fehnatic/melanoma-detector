"""Grad-CAM-Heatmap: zeigt, welche Bildregionen die Vorhersage getragen haben.

Sinnvoll vor allem als Plausibilitaetscheck - wenn die Heatmap auf Bildrand,
Lineal oder Haare zeigt statt auf die Laesion, hat das Modell eine Abkuerzung
gelernt und die Zahl ist nicht vertrauenswuerdig.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def find_target_layer(model: nn.Module) -> nn.Module:
    """Letzte Faltungsschicht - passt fuer alle CNN-Backbones aus timm."""
    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise RuntimeError(
            "Keine Conv2d-Schicht gefunden. Bei Transformer-Backbones braucht "
            "Grad-CAM ein reshape_transform - siehe pytorch-grad-cam Doku."
        )
    return last


def gradcam_overlay(predictor, image: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Gibt das Eingabebild mit ueberlagerter Heatmap zurueck."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    # Grad-CAM braucht Backward-Passes; auf MPS ist das bei manchen Ops noch
    # unzuverlaessig, deshalb hier bewusst auf CPU.
    model = predictor.model.to("cpu").eval()
    tensor = predictor.transform(image.convert("RGB")).unsqueeze(0)

    size = predictor.image_size
    base = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0

    with GradCAM(model=model, target_layers=[find_target_layer(model)]) as cam:
        grayscale = cam(input_tensor=tensor, targets=[BinaryClassifierOutputTarget(1)])[0]

    predictor.model.to(predictor.device)
    overlay = show_cam_on_image(base, grayscale, use_rgb=True, image_weight=1.0 - alpha)
    return Image.fromarray(overlay)
