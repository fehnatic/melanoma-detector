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

| Kürzel     | Quelle | Umfang | Zugang | Labels |
|------------|--------|--------|--------|--------|
| `isic2020` | [ISIC-Archiv](https://challenge.isic-archive.com/data/) (= die SIIM-ISIC-Competition-Daten) | 33.126 Bilder, 23 GB | **offen, kein Account** | Ground-Truth-CSV, 584 maligne (1,76 %), mit Patienten- und Läsions-IDs |
| `mittal`   | [Melanoma Cancer Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset) | 13.879 Bilder, 79 MB | **offen, kein Account** | Ordner `Benign` / `Malignant`, 47,5 % maligne |
| `javed`    | [Melanoma Skin Cancer Dataset (10.000 Bilder)](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) | 10.605 Bilder, 99 MB | **offen, kein Account** | Ordner `benign` / `malignant`, 48,1 % maligne |

Der Hauptdatensatz wird bewusst **direkt vom ISIC-Archiv** geladen, nicht über
Kaggle. Es sind dieselben Bilder, aber: kein Kaggle-Account nötig, kein Beitritt
zur Competition, und die Ground-Truth-Datei liegt dort in der neueren Fassung
`v2` vor, die zusätzlich `lesion_id` enthält. Obendrein gibt es eine offizielle
Liste von 424 bekannten Duplikatpaaren.

Lizenz: **CC-BY-NC 4.0**, Zitationspflicht → [`CITATION.md`](CITATION.md).

### Zwei Dinge, die dabei leicht schiefgehen

**1. Die Quellen überlappen sich — massiv.** Alle drei stammen direkt oder
indirekt aus dem ISIC-Archiv. Gemessen an `mittal` + `javed` allein:

```
24.484 Bilder  ->  12.476 nach Deduplizierung   (49 % waren Duplikate)
mittal: 13.879 ->   2.187                        (84 % steckten schon in javed)
```

Wirft man das naiv zusammen und splittet zufällig, landet rund die Hälfte der
Validierungsbilder auch im Training — die AUROC sieht dann hervorragend aus und
das Modell ist real schlecht. Deshalb läuft vor dem Split zwingend eine
**quellenübergreifende Deduplizierung** (`melanoma.data.dedup`, siehe unten).

**2. SIIM enthält mehrere Aufnahmen pro Patient.** Ohne gruppenweisen Split misst
die Validierung teilweise Patienten-Wiedererkennung statt Melanom-Erkennung.
`melanoma.data.splits` gruppiert daher nach `patient_id` und stratifiziert
gleichzeitig nach Label.

### Speicherbedarf

Der komplette Kaggle-Download der Competition wäre ~108 GB — aber der Großteil
davon ist **redundant, nicht unlabelt**: dieselben Bilder liegen dort dreifach
vor (JPEG, DICOM und TFRecords). Wir laden nur die JPEG-Fassung des
Trainingssets.

| | Größe | |
|---|---|---|
| `ISIC_2020_Training_JPEG.zip` | 23,0 GB | wird geladen |
| Ground Truth + Duplikatliste | 2,3 MB | wird geladen |
| DICOM-Trainingsset | ~23–49 GB | dieselben Bilder, anderer Container |
| TFRecords | ~30 GB | dieselben Bilder, TPU-Verpackung |
| Testset (JPEG + DICOM) | ~22 GB | Labels hält die Challenge zurück |

Während des Entpackens werden kurzzeitig ~46 GB belegt; das ZIP wird danach
automatisch gelöscht (`--keep-zip` behält es). Plus die beiden Kaggle-Datensätze (79 + 99 MB):
**gesamt ~23,2 GB** Rohdaten auf der Platte.

Genaue Größen vorab abfragen, ohne etwas zu laden:

```bash
.venv/bin/python -m melanoma.data.download --dry-run
```

Aus den DICOM-Dateien ginge uns nichts verloren: Auflösung ist bei 384 px
Trainingsgröße irrelevant, und die Metadaten (Alter, Geschlecht, Körperstelle)
stehen ohnehin in der Ground-Truth-CSV.

---

## Setup

```bash
make setup        # venv + Abhängigkeiten
```

### Zugangsdaten

**Keine nötig.** Der ISIC-Datensatz kommt offen vom Archiv, und die beiden
Kaggle-Datensätze sind dort öffentlich freigegeben und lassen sich ohne Account
und ohne API-Token über den Download-Endpunkt ziehen (verifiziert: HTTP 206 ohne
`Authorization`-Header).

Falls Kaggle das eines Tages sperrt, meldet sich das als HTTP 401/403; der
Ausweg über einen kostenlosen API-Token steht im Docstring von
`download_kaggle_dataset()`.

---

## Ablauf

```bash
make download     # Rohdaten laden            -> data/raw/          (~23 GB, einmalig)
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
  data/download.py          ISIC direkt vom Archiv + Kaggle-Datensätze, resume-fähig
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

0. **Offizielle Duplikatliste** des ISIC-Archivs (424 Paare) — das ist Ground
   Truth und wird ohne Schwellwert direkt eingespeist.
1. **pHash** (64 Bit) pro Bild — robust gegen Reskalierung und JPEG-Artefakte,
   also genau gegen die Unterschiede zwischen den Quellen.
2. **LSH-Banding** (8 Bänder à 8 Bit) als Kandidatenfilter. Bei 8 Bändern teilt
   sich jedes Paar mit Hamming-Distanz ≤ 7 garantiert mindestens ein Band — bei
   Schwellwert 5 verliert der Filter also nichts.
3. Exakte Hamming-Distanz auf den Kandidaten, dann **Union-Find** → Cluster.
3b. **Verkettungsprüfung.** Union-Find ist Single-Linkage: aus A~B und B~C folgt
   A~C, auch wenn A und C weit auseinanderliegen. Auf kontrastarmen Bildern, wo
   pHash unzuverlässig wird, entstanden dadurch Ketten von über hundert Bildern
   — im größten lagen nur 6,6 % der Paare tatsächlich unter dem Schwellwert.
   Jeder Cluster ab drei Bildern wird deshalb nachgeprüft: greedy wird das Bild
   mit den meisten Nachbarn innerhalb des Schwellwerts gewählt und nur dessen
   Umgebung als Cluster akzeptiert. Der Clusterdurchmesser ist damit auf
   2·`max_hamming` begrenzt statt unbegrenzt (gemessen: größter Cluster
   129 → 32, Label-Konflikte 457 → 141 Bilder).
4. Pro Cluster bleibt ein Bild übrig, Priorität `isic2020 > javed > mittal`
   (ISIC zuerst, weil nur dort Patienten-IDs hängen).
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

Der Code liegt in einem privaten Repository. Die Daten werden **nicht**
mitversioniert (`/data/` steht in `.gitignore`).

Der ISIC-Datensatz steht unter **CC-BY-NC 4.0** — Namensnennung erforderlich,
kommerzielle Nutzung ausgeschlossen. Details und Zitationsvorgabe in
[`CITATION.md`](CITATION.md).
