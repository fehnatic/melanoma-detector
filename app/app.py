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
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from melanoma.config import load_config, resolve_paths  # noqa: E402
from melanoma.explain import gradcam_overlay  # noqa: E402
from melanoma.predict import Predictor  # noqa: E402
from melanoma.tracking import add_entry, get_history, list_labels  # noqa: E402

DISCLAIMER = """
### ⚠️ Kein Medizinprodukt

Dieses Werkzeug ist ein Forschungs- und Lernprojekt. Es ist **nicht** klinisch
validiert, nicht als Medizinprodukt zugelassen und **nicht** für die Diagnose
geeignet. Ein niedriger Wert schließt ein Melanom nicht aus.

**Bei jeder Veränderung eines Muttermals: ärztlich abklären lassen.**
"""

# Quelle -> (Anzeigename, Radio-Wert). Das Dermatoskop-Modell ist auf Dermoskopie-
# Bildern (ISIC) trainiert, das Handy-Modell zusaetzlich auf PAD-UFES-20 (klinische
# Handyfotos) fine-getuned - siehe README, Abschnitt "Zwei Modelle, zwei Bildarten".
SOURCE_LABELS = {
    "dermatoskop": "🔬 Dermatoskop-Aufnahme",
    "handy": "📱 normales Handyfoto",
}
SOURCE_HINTS = {
    "dermatoskop": (
        "Dermatoskop-Modell: trainiert auf ISIC-2020-Dermoskopie-Bildern, "
        "AUROC 0.907 auf dem Testsplit."
    ),
    "handy": (
        "Handyfoto-Modell: dasselbe Modell, zusaetzlich auf PAD-UFES-20 "
        "(klinische Handyfotos) fine-getuned, Kreuzvalidierungs-AUROC 0.913. "
        "Ohne diese Anpassung liegt ein reines Dermatoskop-Modell auf Handyfotos "
        "nur bei ~0.79 - die Bildart ist also wichtig fuer die Verlaesslichkeit."
    ),
}


def _run_prediction(predictor: Predictor, image: Image.Image, with_heatmap: bool):
    """Gemeinsame Analyse-Logik fuer den Analyse- und den Verlauf-Tab."""
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
    return result, scores, heatmap, "\n".join(details)


def build_analyse_tab(predictors: dict[str, Predictor], show_cam: bool) -> None:
    default_source = "dermatoskop" if "dermatoskop" in predictors else next(iter(predictors))

    def analyse(image: Image.Image, source: str, with_heatmap: bool):
        if image is None:
            return None, None, "Bitte zuerst ein Bild hochladen.", ""
        _, scores, heatmap, details = _run_prediction(predictors[source], image, with_heatmap)
        return scores, heatmap, details, SOURCE_HINTS[source]

    with gr.Row():
        with gr.Column():
            source_in = gr.Radio(
                choices=[(label, key) for key, label in SOURCE_LABELS.items() if key in predictors],
                value=default_source,
                label="Wie wurde das Bild aufgenommen?",
            )
            source_hint = gr.Markdown(SOURCE_HINTS[default_source])
            image_in = gr.Image(type="pil", label="Bild", height=380)
            heatmap_toggle = gr.Checkbox(value=show_cam, label="Grad-CAM-Heatmap berechnen")
            run = gr.Button("Analysieren", variant="primary")
        with gr.Column():
            label_out = gr.Label(num_top_classes=2, label="Einschätzung")
            heatmap_out = gr.Image(label="Worauf das Modell schaut", height=380)
            details_out = gr.Markdown()

    inputs = [image_in, source_in, heatmap_toggle]
    outputs = [label_out, heatmap_out, details_out, source_hint]
    run.click(analyse, inputs, outputs)
    image_in.change(analyse, inputs, outputs)
    source_in.change(analyse, inputs, outputs)


def _render_history(tracking_root: Path, label: str):
    history = get_history(tracking_root, label) if label else []
    if not history:
        empty_df = pd.DataFrame({"Datum": [], "Wahrscheinlichkeit maligne": []})
        return [], empty_df, "Noch keine Aufnahmen fuer dieses Muttermal."

    gallery = [(e["image_path"], e["timestamp"][:10]) for e in history]
    df = pd.DataFrame(
        {
            "Datum": [e["timestamp"][:10] for e in history],
            "Wahrscheinlichkeit maligne": [e["prob_malignant"] for e in history],
        }
    )
    note = f"{len(history)} Aufnahme(n), erste am {history[0]['timestamp'][:10]}."
    if len(history) >= 2:
        delta = history[-1]["prob_malignant"] - history[0]["prob_malignant"]
        if delta > 0.05:
            trend = f"seit der ersten Aufnahme gestiegen ({delta:+.1%}) - im Auge behalten."
        elif delta < -0.05:
            trend = f"seit der ersten Aufnahme gesunken ({delta:+.1%})."
        else:
            trend = "seit der ersten Aufnahme stabil."
        note += f" Modell-Einschätzung {trend}"
    return gallery, df, note


def build_tracking_tab(predictors: dict[str, Predictor], tracking_root: Path) -> None:
    default_source = "dermatoskop" if "dermatoskop" in predictors else next(iter(predictors))
    source_choices = [(label, key) for key, label in SOURCE_LABELS.items() if key in predictors]

    gr.Markdown(
        "Dasselbe Muttermal über Zeit fotografieren, um **Veränderungen** zu erkennen "
        "(das \"Evolving\" in der ABCDE-Regel) - oft aussagekräftiger als eine einzelne "
        "Momentaufnahme. Es werden bewusst keine automatischen Größen-/Farbwerte berechnet - "
        "ohne festen Kameraabstand wäre das nur vorgetäuschte Genauigkeit. Vergleiche die "
        "Fotos stattdessen selbst, und beobachte die Modell-Wahrscheinlichkeit über die Zeit."
    )

    with gr.Row():
        with gr.Column():
            label_in = gr.Dropdown(
                choices=list_labels(tracking_root),
                label="Bezeichnung des Muttermals (z.B. \"Rücken oben links\")",
                allow_custom_value=True,
            )
            source_in = gr.Radio(choices=source_choices, value=default_source, label="Wie wurde das Bild aufgenommen?")
            image_in = gr.Image(type="pil", label="Neues Foto", height=320)
            add_btn = gr.Button("Zum Verlauf hinzufügen", variant="primary")
            status_out = gr.Markdown()
        with gr.Column():
            gallery_out = gr.Gallery(label="Aufnahmen im zeitlichen Verlauf", columns=4, height=320)
            plot_out = gr.LinePlot(
                x="Datum", y="Wahrscheinlichkeit maligne", title="Wahrscheinlichkeit über die Zeit"
            )
            note_out = gr.Markdown()

    def on_select(label):
        gallery, df, note = _render_history(tracking_root, label)
        return gallery, df, note

    def on_add(label, image, source):
        if not label or not label.strip():
            return gr.update(), gr.update(), gr.update(), "Bitte zuerst eine Bezeichnung eingeben."
        if image is None:
            return gr.update(), gr.update(), gr.update(), "Bitte zuerst ein Foto hochladen."

        result, _, _, _ = _run_prediction(predictors[source], image, with_heatmap=False)
        add_entry(tracking_root, label.strip(), image, result["prob_malignant"], source)
        gallery, df, note = _render_history(tracking_root, label.strip())
        return gr.update(choices=list_labels(tracking_root), value=label.strip()), gallery, df, note

    label_in.change(on_select, [label_in], [gallery_out, plot_out, note_out])
    add_btn.click(on_add, [label_in, image_in, source_in], [label_in, gallery_out, plot_out, note_out])


def build_interface(predictors: dict[str, Predictor], tracking_root: Path, show_cam: bool = True) -> gr.Blocks:
    with gr.Blocks(title="Melanom-Erkennung") as demo:
        gr.Markdown("# Melanom-Erkennung\nBild eines Muttermals hochladen.")
        gr.Markdown(DISCLAIMER)

        with gr.Tabs():
            with gr.Tab("Analyse"):
                build_analyse_tab(predictors, show_cam)
            with gr.Tab("🕒 Verlauf"):
                build_tracking_tab(predictors, tracking_root)

    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gradio-App starten")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dermatoscope-run", default="isic_baseline", help="Modell fuer Dermatoskop-Bilder")
    parser.add_argument("--phone-run", default="pad_ufes_final", help="Modell fuer normale Handyfotos")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="oeffentlichen Gradio-Link erzeugen")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)

    run_names = {"dermatoskop": args.dermatoscope_run, "handy": args.phone_run}
    predictors = {}
    for source, run_name in run_names.items():
        try:
            predictors[source] = Predictor(
                paths.artifacts / run_name,
                device=cfg["train"]["device"],
                tta=cfg["inference"]["tta"],
            )
        except FileNotFoundError as exc:
            print(f"Hinweis: {source}-Modell ({run_name}) nicht verfuegbar - {exc}")
    if not predictors:
        raise SystemExit("Kein Modell gefunden - erst trainieren.")

    paths.tracking.mkdir(parents=True, exist_ok=True)
    build_interface(predictors, paths.tracking).launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
