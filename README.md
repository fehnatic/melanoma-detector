# Melanom-Erkennung

Bild eines Muttermals hochladen → Wahrscheinlichkeit für „benigne" bzw. „maligne".
Computer-Vision-Projekt auf Basis dreier öffentlicher Kaggle-Datensätze.

---

## ⚠️ Kein Medizinprodukt

Dieses Projekt ist ein **Lern- und Forschungsprojekt**. Es ist nicht klinisch
validiert, nicht als Medizinprodukt zugelassen (MDR/FDA) und nicht für
diagnostische Zwecke geeignet. Ein niedriger Wert schließt ein Melanom **nicht**
aus. Bei jeder Veränderung eines Muttermals gilt: ärztlich abklären lassen.

---

## Datenquellen

| Kürzel   | Quelle | Umfang | Labels |
|----------|--------|--------|--------|
| `siim`   | [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data) | ~33.000 Bilder | `target` aus `train.csv`, ~1,8 % maligne, mit Patienten-IDs |
| `mittal` | [Melanoma Cancer Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset) | ~13.900 Bilder | Ordnerstruktur `benign` / `malignant`, ca. 50/50 |
| `javed`  | [Melanoma Skin Cancer Dataset (10.000 Bilder)](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) | ~10.600 Bilder | Ordnerstruktur `benign` / `malignant`, ca. 50/50 |

### Zwei Dinge, die dabei leicht schiefgehen

**1. Die Quellen überlappen sich.** Alle drei stammen direkt oder indirekt aus dem
ISIC-Archiv. Datensatz 2 und 3 überschneiden sich stark miteinander, und beide
enthalten Bilder, die auch in SIIM vorkommen. Wirft man das naiv zusammen und
splittet zufällig, landen identische Bilder gleichzeitig in Training und
Validierung — die AUROC sieht dann hervorragend aus und das Modell ist real
schlecht. Deshalb läuft vor dem Split eine **quellenübergreifende
Deduplizierung** (`melanoma.data.dedup`, siehe unten).

**2. SIIM enthält mehrere Aufnahmen pro Patient.** Ohne gruppenweisen Split misst
die Validierung teilweise Patienten-Wiedererkennung statt Melanom-Erkennung.
`melanoma.data.splits` gruppiert daher nach `patient_id` und stratifiziert
gleichzeitig nach Label.

### Speicherbedarf

Der vollständige SIIM-Download wäre ~108 GB, weil er die DICOM-Originale enthält.
Die brauchen wir nicht. `melanoma.data.download` lädt deshalb nur `train.csv`,
leitet daraus die benötigten JPEG-Pfade ab und zieht ausschließlich diese Dateien
einzeln über die Kaggle-REST-API (**~8–10 GB**, resume-fähig). Das SIIM-Testset
wird standardmäßig nicht geladen — es ist unlabeled und damit für uns wertlos.

Gesamt auf der Platte: ~15 GB Rohdaten.

---

## Setup

```bash
make setup        # venv + Abhängigkeiten
```

### Kaggle-Zugang

1. https://www.kaggle.com/settings → *API* → **Create New Token** → `kaggle.json`
2. ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. **Einmalig im Browser** auf der [SIIM-Competition-Seite](https://www.kaggle.com/competitions/siim-isic-melanoma-classification)
   die Regeln akzeptieren („Join Competition"). Ohne das liefert die API `403`.

---

## Ablauf

```bash
make download     # Rohdaten von Kaggle       -> data/raw/          (~15 GB, dauert)
make data         # manifest + dedup + split  -> data/processed/splits.csv
make train        # Training                  -> artifacts/baseline/best.pt
make calibrate    # Kalibrierung auf Val      -> artifacts/baseline/calibrator.json
make evaluate     # Testmetriken              -> reports/metrics_*.json
make app          # Gradio-Oberfläche         -> http://127.0.0.1:7860
```

Einzelne Bilder ohne UI:

```bash
.venv/bin/python -m melanoma.predict pfad/zum/bild.jpg
```

### Schnell ausprobieren, bevor die großen Daten da sind

```bash
.venv/bin/python -m melanoma.data.download --sources javed   # kleinster Datensatz
make data
.venv/bin/python -m melanoma.train --backbone resnet18 --epochs 2 --image-size 224 --limit 2000
```

---

## Wie die Wahrscheinlichkeit zustande kommt

Die rohen Sigmoid-Ausgaben eines Netzes sind keine ehrlichen Wahrscheinlichkeiten —
sie sind typischerweise überkonfident. Dazu kommt, dass beim Training die
Minderheitsklasse überabgetastet wird (`imbalance: sampler`), das Modell also
faktisch ~50 % maligne Fälle sieht.

`make calibrate` korrigiert beides: Isotonic Regression auf dem Validierungsset
bildet die Rohwerte auf kalibrierte Wahrscheinlichkeiten ab. Die Kalibrierung wird
als **JSON** abgelegt (Stützstellen bzw. ein Temperatur-Skalar), nicht als Pickle —
lesbar, versionsstabil und ungefährlich beim Laden.

**Bezugsgröße beachten:** Die ausgegebene Zahl bezieht sich auf die
Klassenverteilung dieser Datensätze, nicht auf die Melanom-Prävalenz in der
Allgemeinbevölkerung. Wer sie umrechnen will, setzt `calibration.target_prevalence`
in `configs/default.yaml` (Prior-Shift auf den Odds, siehe `prior_shift()`).

Metrik der Wahl ist **AUROC**, ergänzt um **Average Precision** — bei ~2 %
Positivrate ist Accuracy nutzlos (ein Modell, das immer „benigne" sagt, hat 98 %).
`make evaluate` gibt zusätzlich Schwellwerte für 90/95/98 % Sensitivität aus, weil
in diesem Anwendungsfall falsch-negative Befunde deutlich teurer sind als
falsch-positive.

---

## Projektstruktur

```
configs/default.yaml        alle Hyperparameter und Pfade an einem Ort
src/melanoma/
  config.py                 YAML laden, Pfade auflösen
  data/download.py          Kaggle-Download (SIIM selektiv, resume-fähig)
  data/manifest.py          drei Quellen -> ein einheitliches Manifest
  data/dedup.py             pHash + LSH + Union-Find, quellenübergreifend
  data/splits.py            gruppenweiser, stratifizierter Split
  data/dataset.py           Torch-Dataset, Augmentierungen, Balanced Sampler
  models.py                 timm-Backbone, Geräteauswahl (MPS/CUDA/CPU), EMA
  train.py                  Trainingsschleife
  calibrate.py              Isotonic / Temperature Scaling, Prior-Shift
  evaluate.py               Testmetriken inkl. Aufschlüsselung nach Quelle
  predict.py                Inferenz für ein Einzelbild (mit TTA)
  explain.py                Grad-CAM-Heatmap
app/app.py                  Gradio-Oberfläche
tests/                      laufen ohne heruntergeladene Daten
```

### Deduplizierung im Detail

1. **pHash** (64 Bit) pro Bild — robust gegen Reskalierung und JPEG-Artefakte,
   also genau gegen die Unterschiede zwischen den Quellen.
2. **LSH-Banding** (8 Bänder à 8 Bit) als Kandidatenfilter. Bei 8 Bändern teilt
   sich jedes Paar mit Hamming-Distanz ≤ 7 garantiert mindestens ein Band — bei
   Schwellwert 5 verliert der Filter also nichts.
3. Exakte Hamming-Distanz auf den Kandidaten, dann **Union-Find** → Cluster.
4. Pro Cluster bleibt ein Bild übrig, Priorität `siim > javed > mittal` (SIIM
   zuerst, weil nur dort Patienten-IDs hängen).
5. Cluster mit **widersprüchlichen Labels** werden komplett verworfen und
   protokolliert — ein Bild, das einmal benigne und einmal maligne annotiert ist,
   ist als Trainingssignal wertlos.

Report unter `reports/duplicates.csv`: jede entfernte Datei mit dem Bild, das
stattdessen behalten wurde. Vorher anschauen, ohne etwas zu ändern:

```bash
.venv/bin/python -m melanoma.data.dedup --report-only
```

Der Schwellwert lässt sich in `configs/default.yaml` unter `dedup.max_hamming`
justieren — höher = aggressiver.

---

## Training auf dem Mac

Läuft über **MPS** (Apple Silicon). Voreingestellt ist `tf_efficientnetv2_s` bei
384 px. Anhaltspunkt auf einem M2 Pro: mit ~45.000 Bildern nach Deduplizierung
grob 1–2 Stunden pro Epoche. Wenn das zu lang ist:

- `--image-size 256` (deutlich schneller, kostet etwas AUROC)
- `--backbone resnet50` oder `efficientnet_b0`
- `train.epochs` reduzieren — die ersten 4–6 Epochen bringen den Großteil

`train.amp` steht bewusst auf `false`: bf16-Autocast ist auf MPS noch nicht
zuverlässig.

---

## Bekannte Grenzen

- **Datenherkunft:** Die ISIC-Bilder stammen überwiegend von hellhäutigen
  Patienten aus Europa, Nordamerika und Australien. Auf dunkleren Hauttypen ist
  die Leistung vermutlich deutlich schlechter — ungetestet.
- **Aufnahmebedingungen:** Trainiert wird auf dermatoskopischen Aufnahmen.
  Handyfotos haben andere Beleuchtung, Schärfe und Skalierung; die Werte darauf
  sind nicht belastbar.
- **Shortcuts:** Teile des ISIC-Archivs enthalten Lineale und Marker neben
  Läsionen, die mit der Malignität korrelieren. Die Grad-CAM-Heatmap in der App
  ist dafür der Plausibilitätscheck — zeigt sie auf Bildrand oder Lineal statt
  auf die Läsion, ist die Zahl nicht vertrauenswürdig.
- **Kein externer Testsatz:** Test- und Trainingsdaten stammen aus derselben
  Sammlung. Die Testmetrik ist damit eine Obergrenze, keine Schätzung der
  Leistung auf neuen Daten.

---

## Lizenz / Daten

Der Code liegt in einem privaten Repository. Die Datensätze unterliegen den
jeweiligen Kaggle-Lizenzbedingungen und werden **nicht** mitversioniert
(`data/` steht in `.gitignore`).
