# Melanom-Erkennung

Bild eines Muttermals hochladen → Wahrscheinlichkeit für „benigne" bzw. „maligne",
plus Verlaufstracking über Zeit. Computer-Vision-Projekt auf Basis von vier
öffentlichen Datensätzen (dermatoskopisch **und** klinisch/Handyfoto).

---

## ⚠️ Kein Medizinprodukt

Dieses Projekt ist ein **Lern- und Forschungsprojekt**. Es ist nicht klinisch
validiert, nicht als Medizinprodukt zugelassen (MDR/FDA) und nicht für
diagnostische Zwecke geeignet. Ein niedriger Wert schließt ein Melanom **nicht**
aus. Bei jeder Veränderung eines Muttermals gilt: ärztlich abklären lassen.

---

## Datenquellen

| Kürzel       | Quelle | Umfang | Zugang | Labels | Bildart |
|--------------|--------|--------|--------|--------|---------|
| `isic2020`   | [ISIC-Archiv](https://challenge.isic-archive.com/data/) (= die SIIM-ISIC-Competition-Daten) | 33.126 Bilder, 23 GB | **offen, kein Account** | Ground-Truth-CSV, 584 maligne (1,76 %), mit Patienten- und Läsions-IDs | Dermatoskop |
| `mittal`     | [Melanoma Cancer Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset) | 13.879 Bilder, 79 MB | **offen, kein Account** | Ordner `Benign` / `Malignant`, 47,5 % maligne | Dermatoskop |
| `javed`      | [Melanoma Skin Cancer Dataset (10.000 Bilder)](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) | 10.605 Bilder, 99 MB | **offen, kein Account** | Ordner `benign` / `malignant`, 48,1 % maligne | Dermatoskop |
| `pad_ufes_20`| [PAD-UFES-20](https://arxiv.org/abs/2007.00478), gespiegelt im [ISIC-Archiv](https://api.isic-archive.com/collections/406/) | 2.298 Bilder, 220 MB | **offen, kein Account** | 52 Melanome (2,3 %), Rest BCC/SCC/Nävus/Keratose, biopsie-bestätigt | **Handyfoto (klinisch)** |

Der Hauptdatensatz `isic2020` wird bewusst **direkt vom ISIC-Archiv** geladen,
nicht über Kaggle. Es sind dieselben Bilder, aber: kein Kaggle-Account nötig,
kein Beitritt zur Competition, und die Ground-Truth-Datei liegt dort in der
neueren Fassung `v2` vor, die zusätzlich `lesion_id` enthält. Obendrein gibt es
eine offizielle Liste von 424 bekannten Duplikatpaaren.

`pad_ufes_20` läuft über denselben ISIC-Archiv-Zugriffsweg (Collection 406),
ebenfalls ohne Account. Es ist der einzige Datensatz hier mit **echten
Handyfotos** statt Dermoskopie-Aufnahmen — siehe [„Zwei Modelle, zwei
Bildarten"](#zwei-modelle-zwei-bildarten) unten, warum das wichtig ist.

Lizenzen: `isic2020`/`mittal`/`javed` **CC-BY-NC 4.0**, `pad_ufes_20` **CC-BY**.
Zitationspflicht → [`CITATION.md`](CITATION.md).

### Zwei Dinge, die dabei leicht schiefgehen

**1. Die Quellen überlappen sich — massiv.** Alle drei Dermoskopie-Quellen
stammen direkt oder indirekt aus dem ISIC-Archiv. Gemessen an `mittal` +
`javed` allein:

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
| `mittal` + `javed` | 178 MB | wird geladen |
| `pad_ufes_20` | 220 MB | wird geladen (nur fürs Handyfoto-Modell) |
| DICOM-Trainingsset | ~23–49 GB | dieselben Bilder, anderer Container |
| TFRecords | ~30 GB | dieselben Bilder, TPU-Verpackung |
| Testset (JPEG + DICOM) | ~22 GB | Labels hält die Challenge zurück |

Während des Entpackens werden kurzzeitig ~46 GB belegt; das ZIP wird danach
automatisch gelöscht (`--keep-zip` behält es). **Gesamt ~23,4 GB** Rohdaten auf
der Platte für alle vier Quellen.

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

**Keine nötig.** Alle vier Datensätze sind offen zugänglich und lassen sich ohne
Account und ohne API-Token laden (ISIC-Archiv direkt, bzw. bei Kaggle über den
Download-Endpunkt ohne `Authorization`-Header).

Falls Kaggle das eines Tages sperrt, meldet sich das als HTTP 401/403; der
Ausweg über einen kostenlosen API-Token steht im Docstring von
`download_kaggle_dataset()`.

---

## Ablauf

```bash
make download     # Rohdaten laden (isic2020, mittal, javed)      -> data/raw/        (~23 GB, einmalig)
make data         # manifest + dedup + prepare + split            -> data/processed/splits_default.csv
make train        # Training (Standard-Backbone: ISIC-Split)      -> artifacts/baseline/best.pt
make calibrate    # Kalibrierung auf Val                          -> artifacts/baseline/calibrator.json
make evaluate     # Testmetriken                                  -> reports/metrics_*.json
make app          # Gradio-Oberfläche (beide Modelle)             -> http://127.0.0.1:7860
```

`make data` läuft in vier Schritten: **Manifest bauen → Deduplizieren →
Vorverkleinern (`prepare`, ~10× schnelleres Laden beim Training) → Splitten.**
Die Split-Config in `configs/default.yaml` (`sources` / `eval_sources`) steuert,
welche Quellen überhaupt einfließen und woraus Val/Test stammen dürfen — siehe
den nächsten Abschnitt, warum das hier entscheidend ist.

Die tatsächlich verwendeten Produktions-Splits (bereits mit den richtigen
`--sources`/`--eval-sources`-Flags erzeugt):

```bash
.venv/bin/python -m melanoma.data.splits --name isic --sources isic2020 --eval-sources isic2020
.venv/bin/python -m melanoma.data.splits --name all  --eval-sources isic2020   # + javed/mittal nur im Training
```

Einzelne Bilder ohne UI:

```bash
.venv/bin/python -m melanoma.predict pfad/zum/bild.jpg --run-name isic_baseline
```

### Schnell ausprobieren, bevor die großen Daten da sind

```bash
.venv/bin/python -m melanoma.data.download --sources javed   # kleinster Datensatz
make data
.venv/bin/python -m melanoma.train --backbone resnet18 --epochs 2 --image-size 224 --limit 2000
```

---

## Zwei Modelle, zwei Bildarten

Dermatoskop-Aufnahmen (nah, durchleuchtet, ohne Umgebungslicht) und normale
Handyfotos sehen strukturell komplett unterschiedlich aus. Ein Modell, das nur
auf Dermoskopie trainiert ist, überträgt sich nicht einfach auf Handyfotos —
das haben wir hier **gemessen statt angenommen**:

| Modell | Trainingsdaten | AUROC | Wo |
|---|---|---|---|
| `isic_baseline` | nur `isic2020` | **0.9075** | ISIC-Testsplit (Dermoskopie) |
| `isic_baseline` (unangepasst) | — | **0.7884** | `pad_ufes_20` (Handyfotos) — deutlicher Einbruch |
| `pad_ufes_final` | `isic_baseline` + Fine-Tuning auf `pad_ufes_20` | **0.9132** (5-Fold-Kreuzvalidierung) | `pad_ufes_20` (Handyfotos) |

Das reine Dermatoskop-Modell verliert **~0.12 AUROC**, sobald es auf Handyfotos
trifft. Mit Fine-Tuning auf PAD-UFES-20 (nur 52 Melanom-Fälle insgesamt, daher
per Gruppen-Kreuzvalidierung statt einem einzelnen Split ausgewertet — siehe
`melanoma.finetune_pad_ufes`) schließt sich die Lücke fast vollständig.

**Deshalb gibt es in der App eine explizite Auswahl "Wie wurde das Bild
aufgenommen?"**, die zwischen zwei Checkpoints umschaltet — `isic_baseline` für
Dermatoskop-Bilder, `pad_ufes_final` für normale Fotos. Ein Bild dem falschen
Modell zu geben, verschlechtert die Zahl messbar (siehe Tabelle).

So läuft die Cross-Domain-Auswertung und das Fine-Tuning:

```bash
# Wie gut ist ein bestehendes Modell auf einem strukturell anderen Datensatz? (kein Training)
.venv/bin/python -m melanoma.eval_external --dataset pad_ufes_20 --run-name isic_baseline

# Fine-Tuning mit 5-Fold-Gruppen-Kreuzvalidierung (Out-of-Fold-AUROC über alle 2.298 Bilder)
.venv/bin/python -m melanoma.finetune_pad_ufes --run-name pad_ufes_finetune --epochs 12

# Finales Produktionsmodell auf (fast) allen PAD-UFES-20-Daten
.venv/bin/python -m melanoma.finetune_pad_ufes --final --run-name pad_ufes_final --epochs 12
.venv/bin/python -m melanoma.calibrate --run-name pad_ufes_final
```

### Was bringen die anderen Rohdatensätze? (negatives Ergebnis)

Naheliegender Gedanke: einfach `javed` und `mittal` mit ins Training packen,
mehr Daten sind doch besser. Getestet, mit sauberer Auswertung (Val/Test bleibt
in beiden Läufen **identisch** reines `isic2020`, nur die Trainingsdaten
unterscheiden sich — über `split.eval_sources`):

| Trainingsdaten | Val-AUROC | **Test-AUROC** |
|---|---|---|
| nur `isic2020` (`isic_baseline`) | 0.9069 | **0.9075** |
| `isic2020` + `javed` + `mittal` (`all_sources_baseline`) | 0.9054 | **0.8778** |

Auf Val sehen beide Läufe fast gleich aus — auf dem unabhängigen Testsplit
fällt der Lauf mit den Zusatzdaten deutlich ab (**−0.03 AUROC**). Grund
vermutlich: `javed`/`mittal` bestehen nach Deduplizierung fast nur noch aus
malignen Bildern und sind an der Auflösung trivial vom Rest unterscheidbar
(300×300 vs. 6000×4000) — ein Domain-Shift, den die zusätzlichen positiven
Beispiele nicht aufwiegen. **Konsequenz:** Produktionsmodell nutzt nur
`isic2020`; `javed`/`mittal` bleiben Kandidaten für gezieltere Experimente
(z. B. mit Sample-Gewichtung statt voller Beimischung), aber nicht für
naives Zusammenwerfen.

### Nicht-quadratische Fotos: Crop statt Padding (auch ausprobiert, verworfen)

Bei einem breiten oder hohen Foto muss vor der Klassifikation irgendwie ein
Quadrat draus werden. Standard hier ist **Resize + CenterCrop**: die Ränder
werden abgeschnitten. Bei einem sehr breiten, unzentrierten Handyfoto kann das
im Extremfall die Läsion mit wegschneiden — deshalb lag die Idee nahe, den
Zuschnitt durch **Padding** zu ersetzen (Bild vollständig erhalten, Ränder mit
Spiegelung auffüllen, siehe `melanoma.data.dataset.build_transforms(...,
pad_to_square=True)`). Gemessen statt angenommen:

| Eval-Transform | ISIC-Test-AUROC |
|---|---|
| Resize + CenterCrop (Standard) | **0.9075** |
| Padding statt Crop | **0.878** (−0.03) |

Padding macht es **schlechter**, nicht besser: Training läuft über
`RandomResizedCrop` auf denselben (rechteckigen) ISIC-Rohbildern — das Modell
hat also gelernt, einen bestimmten Bildausschnitt/Zoom zu erwarten. Ein
komplett reingepolstertes Bild sieht dafür ungewohnt "herausgezoomt" aus, das
kostet AUROC. **Konsequenz:** Crop bleibt der Standard; `pad_to_square` existiert
als getestete, aber bewusst nicht aktivierte Option (`--pad-to-square`-Flag in
`evaluate.py`/`eval_external.py`). Die eigentlich sinnvolle Reaktion auf schlecht
gerahmte Fotos ist eine Empfehlung an die Nutzer:in (Läsion mittig,
bildfüllend fotografieren), nicht ein Eingriff in die Vorverarbeitung.

Beim Nachschauen ist dabei ein echter Bug aufgefallen: Die Grad-CAM-Heatmap
("Worauf das Modell schaut" in der App) hat den Hintergrund über ein
eigenständiges `image.resize((size, size))` erzeugt — das **staucht** ein
nicht-quadratisches Foto, statt es zuzuschneiden, und zeigt damit einen
anderen Bildausschnitt als den, der tatsächlich klassifiziert wurde. Behoben,
indem Heatmap und Klassifikation jetzt denselben Zuschnitt teilen
(`build_preview_crop()` in `dataset.py`).

---

## Verlaufstracking

Ein einzelnes Foto sagt wenig über **Veränderung** — und "Evolving" ist eines
der fünf ABCDE-Kriterien für Melanom-Verdacht. Der `🕒 Verlauf`-Tab in der App
lässt dasselbe Muttermal (per frei wählbarer Bezeichnung, z. B. „Rücken oben
links") über mehrere Aufnahmen hinweg verfolgen:

- Fotos landen chronologisch in einer Galerie — zum eigenen visuellen Vergleich.
- Die Modell-Wahrscheinlichkeit über die Zeit wird als Kurve angezeigt, mit
  einem Hinweis, wenn sie seit der ersten Aufnahme spürbar gestiegen ist.

**Bewusst keine automatische Größen- oder Farbmessung.** Ohne fixen
Kameraabstand und Referenzmaßstab wäre eine „Fläche +12 %"-Aussage
vorgetäuschte Präzision — genau die Art Zahl, die man in einem
Forschungsprojekt ohne Kalibrierung nicht ausgeben sollte. Die Kurve der
Modell-Wahrscheinlichkeit ist dagegen ein ehrliches Signal: dasselbe Modell
misst bei jeder Aufnahme auf dieselbe Weise.

Daten liegen lokal unter `tracking/` (Fotos + `index.json`) — **nie im
Git-Repo** (`.gitignore`), das sind persönliche Gesundheitsdaten.

---

## Wie die Wahrscheinlichkeit zustande kommt

Die rohen Sigmoid-Ausgaben eines Netzes sind keine ehrlichen Wahrscheinlichkeiten —
sie sind typischerweise überkonfident. Dazu kommt, dass beim Training die
Minderheitsklasse überabgetastet wird (`imbalance: sampler`), das Modell also
faktisch ~50 % maligne Fälle sieht.

`make calibrate` (bzw. `melanoma.calibrate --run-name <name>`, für **jedes**
Modell einzeln) korrigiert beides: Isotonic Regression auf dem Validierungsset
bildet die Rohwerte auf kalibrierte Wahrscheinlichkeiten ab. Die Kalibrierung wird
als **JSON** abgelegt (Stützstellen bzw. ein Temperatur-Skalar), nicht als Pickle —
lesbar, versionsstabil und ungefährlich beim Laden.

**Bezugsgröße beachten:** Die ausgegebene Zahl bezieht sich auf die
Klassenverteilung der jeweiligen Trainingsdaten, nicht auf die Melanom-Prävalenz
in der Allgemeinbevölkerung — und die beiden Modelle haben unterschiedliche
Bezugsprävalenzen (ISIC ~1,8 %, PAD-UFES-Holdout ~2,4 %). Wer sie umrechnen
will, setzt `calibration.target_prevalence` in `configs/default.yaml`
(Prior-Shift auf den Odds, siehe `prior_shift()`).

Metrik der Wahl ist **AUROC**, ergänzt um **Average Precision** — bei ~2 %
Positivrate ist Accuracy nutzlos (ein Modell, das immer „benigne" sagt, hat 98 %).
`make evaluate` / `melanoma.eval_external` geben zusätzlich Schwellwerte für
90/95/98 % Sensitivität aus, weil in diesem Anwendungsfall falsch-negative
Befunde deutlich teurer sind als falsch-positive.

---

## Projektstruktur

```
configs/default.yaml           alle Hyperparameter und Pfade an einem Ort
src/melanoma/
  config.py                    YAML laden, Pfade auflösen
  data/download.py             ISIC direkt vom Archiv (isic2020 + pad_ufes_20) + Kaggle, resume-fähig
  data/manifest.py             drei Dermoskopie-Quellen -> ein einheitliches Manifest
  data/dedup.py                pHash + LSH + Union-Find, quellenübergreifend
  data/prepare.py              Bilder einmalig auf Arbeitsauflösung verkleinern (~10x schnelleres Training)
  data/splits.py                gruppenweiser, stratifizierter Split, mit Quellenfilter (sources/eval_sources)
  data/dataset.py               Torch-Dataset, Augmentierungen, Balanced Sampler
  models.py                     timm-Backbone, Geräteauswahl (MPS/CUDA/CPU), EMA
  train.py                      Trainingsschleife (ISIC, von ImageNet-Gewichten)
  finetune_pad_ufes.py          Fine-Tuning auf PAD-UFES-20: Gruppen-CV + finales Produktionsmodell
  eval_external.py              bestehendes Modell auf einem externen Datensatz auswerten (kein Training)
  calibrate.py                  Isotonic / Temperature Scaling, Prior-Shift (pro Modell)
  evaluate.py                   Testmetriken inkl. Aufschlüsselung nach Quelle
  predict.py                    Inferenz für ein Einzelbild (mit TTA)
  explain.py                    Grad-CAM-Heatmap
  tracking.py                   Persönlicher Verlauf: Fotos + Wahrscheinlichkeit über Zeit
app/app.py                      Gradio-Oberfläche: Analyse-Tab (2 Modelle) + Verlauf-Tab
tests/                          laufen ohne heruntergeladene Daten
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
384 px. Gemessen auf einem M2 Pro:

| Lauf | Trainingsbilder | Zeit/Epoche | Epochen bis Early Stopping | Bestes AUROC |
|---|---|---|---|---|
| `isic_baseline` | 21.964 | ~27 Min | 7 (best: 3) | 0.9075 (Test) |
| `all_sources_baseline` | 28.515 | ~31 Min | 10 (best: 6) | 0.8778 (Test) |
| PAD-UFES-Fine-Tuning, pro Fold | ~1.840 | ~2,5 Min/Epoche | 12 (fest, keine Sättigung erreicht) | 0.9132 (OOF, 5 Folds) |

Wenn das zu lang ist:

- `--image-size 256` (deutlich schneller, kostet etwas AUROC)
- `--backbone resnet50` oder `efficientnet_b0`
- `train.epochs` reduzieren — die ersten 4–6 Epochen bringen den Großteil

`train.amp` steht bewusst auf `false`: bf16-Autocast ist auf MPS noch nicht
zuverlässig.

---

## Bekannte Grenzen

- **Datenherkunft:** Die ISIC-Bilder stammen überwiegend von hellhäutigen
  Patienten aus Europa, Nordamerika und Australien; PAD-UFES-20 aus einem
  brasilianischen Programm für einkommensschwache Patient:innen — eine andere,
  aber ebenfalls nicht global-repräsentative Verteilung. Auf anderen Hauttypen
  ist die Leistung beider Modelle ungetestet.
- **Handyfoto-Modell ist an einer kleinen Zahl Melanome kalibriert:** PAD-UFES-20
  hat insgesamt nur 52 bestätigte Melanom-Fälle. Die 0.913-AUROC-Schätzung
  stammt aus 5-Fold-Kreuzvalidierung (robuster als ein Einzelsplit), bleibt aber
  eine kleine Stichprobe.
- **Shortcuts:** Teile des ISIC-Archivs enthalten Lineale und Marker neben
  Läsionen, die mit der Malignität korrelieren. Die Grad-CAM-Heatmap in der App
  ist dafür der Plausibilitätscheck — zeigt sie auf Bildrand oder Lineal statt
  auf die Läsion, ist die Zahl nicht vertrauenswürdig.
- **Kein externer Testsatz fürs Dermatoskop-Modell:** `isic_baseline`s Test- und
  Trainingsdaten stammen aus derselben Sammlung. Die Testmetrik ist damit eine
  Obergrenze, keine Schätzung der Leistung auf neuen Daten. Für das
  Handyfoto-Modell ist PAD-UFES-20 selbst der externe Testsatz relativ zu ISIC —
  aber beide Modelle wurden nie an einer dritten, komplett unabhängigen Quelle
  gemessen.
- **Verlaufstracking ersetzt keine Verlaufskontrolle beim Arzt:** Die Kurve
  zeigt die Modell-Wahrscheinlichkeit über Zeit, keine kalibrierte
  Größen-/Farbmessung — bei jeder wahrgenommenen Veränderung gilt weiterhin:
  ärztlich abklären lassen.

---

## Lizenz / Daten

Der Code liegt in einem privaten Repository. Die Daten werden **nicht**
mitversioniert (`/data/`, `/tracking/` stehen in `.gitignore`).

Der ISIC-2020-Datensatz (inkl. `mittal`/`javed`, die selbst daraus stammen)
steht unter **CC-BY-NC 4.0** — Namensnennung erforderlich, kommerzielle Nutzung
ausgeschlossen. PAD-UFES-20 steht unter **CC-BY** — Namensnennung erforderlich,
kommerzielle Nutzung erlaubt. Details und Zitationsvorgabe in
[`CITATION.md`](CITATION.md).
