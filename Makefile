PY := .venv/bin/python

.PHONY: help setup download manifest dedup prepare split data train calibrate evaluate eval-external app test clean-derived

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:            ## venv anlegen und Abhaengigkeiten installieren
	python3.13 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

download:         ## Rohdaten von Kaggle laden (braucht ~/.kaggle/kaggle.json)
	$(PY) -m melanoma.data.download

manifest:         ## Einheitliches Manifest ueber alle Quellen bauen
	$(PY) -m melanoma.data.manifest

dedup:            ## Duplikate quellenuebergreifend entfernen
	$(PY) -m melanoma.data.dedup

prepare:          ## Bilder einmalig auf Arbeitsaufloesung verkleinern
	$(PY) -m melanoma.data.prepare

split:            ## Patientenweisen, stratifizierten Split erzeugen
	$(PY) -m melanoma.data.splits

data: manifest dedup prepare split  ## Kompletter Preprocessing-Lauf

train:            ## Training starten
	$(PY) -m melanoma.train

calibrate:        ## Wahrscheinlichkeiten auf dem Val-Set kalibrieren
	$(PY) -m melanoma.calibrate

evaluate:         ## Auf dem Testsplit auswerten
	$(PY) -m melanoma.evaluate

eval-external:    ## Cross-Domain-Auswertung (z.B. PAD-UFES-20 Handyfotos), kein Training
	$(PY) -m melanoma.eval_external

app:              ## Gradio-Oberflaeche starten
	$(PY) app/app.py

test:             ## Tests
	$(PY) -m pytest -q

clean-derived:    ## Abgeleitete Daten loeschen (Rohdaten bleiben)
	rm -rf data/interim data/processed
