"""Gradio-Oberflaeche: Bild hochladen -> Wahrscheinlichkeit + Heatmap.

Start:  python app/app.py            (oder: make app)
Danach: http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from melanoma.config import load_config, resolve_paths  # noqa: E402
from melanoma.explain import gradcam_overlay  # noqa: E402
from melanoma.predict import Predictor  # noqa: E402

DISCLAIMER = """
### ⚠️ Kein Medizinprodukt

Dieses Werkzeug ist ein Forschungs- und Lernprojekt. Es ist **nicht** klinisch
validiert, nicht als Medizinprodukt zugelassen und **nicht** für die Diagnose
geeignet. Ein niedriger Wert schließt ein Melanom nicht aus.

**Bei jeder Veränderung eines Muttermals: ärztlich abklären lassen.**
"""


def build_interface(predictor: Predictor, show_cam: bool = True) -> gr.Blocks:
    def analyse(image: Image.Image, with_heatmap: bool):
        if image is None:
            return None, None, "Bitte zuerst ein Bild hochladen."

        result = predictor.predict(image)
        scores = {
            "maligne (Melanom-verdächtig)": result["prob_malignant"],
            "benigne (unauffällig)": result["prob_benign"],
        }

        heatmap = None
        if with_heatmap:
            try:
                heatmap = gradcam_overlay(predictor, image)
            except Exception as exc:  # noqa: BLE001
                print(f"Grad-CAM fehlgeschlagen: {exc}")

        prevalence = result["reference_prevalence"]
        details = [
            f"**Wahrscheinlichkeit maligne: {result['prob_malignant']:.1%}**",
            "",
            f"- unkalibrierter Rohwert: {result['prob_raw']:.1%}",
            f"- Backbone: `{predictor.meta['backbone']}`",
        ]
        if predictor.meta.get("val_auroc"):
            details.append(f"- AUROC auf dem Validierungsset: {predictor.meta['val_auroc']:.3f}")
        if prevalence is not None:
            details.append(
                f"- Bezugsprävalenz: {prevalence:.2%} maligne. Die Zahl gilt für die "
                "Verteilung der Trainingsdatensätze, nicht für die Allgemeinbevölkerung."
            )
        return scores, heatmap, "\n".join(details)

    with gr.Blocks(title="Melanom-Erkennung") as demo:
        gr.Markdown("# Melanom-Erkennung\nBild eines Muttermals hochladen.")
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Bild", height=380)
                heatmap_toggle = gr.Checkbox(value=show_cam, label="Grad-CAM-Heatmap berechnen")
                run = gr.Button("Analysieren", variant="primary")
            with gr.Column():
                label_out = gr.Label(num_top_classes=2, label="Einschätzung")
                heatmap_out = gr.Image(label="Worauf das Modell schaut", height=380)
                details_out = gr.Markdown()

        run.click(analyse, [image_in, heatmap_toggle], [label_out, heatmap_out, details_out])
        image_in.change(analyse, [image_in, heatmap_toggle], [label_out, heatmap_out, details_out])

    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gradio-App starten")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="oeffentlichen Gradio-Link erzeugen")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    predictor = Predictor(
        paths.artifacts / args.run_name,
        device=cfg["train"]["device"],
        tta=cfg["inference"]["tta"],
    )
    build_interface(predictor).launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
