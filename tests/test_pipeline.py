"""Tests, die ohne heruntergeladene Daten laufen."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melanoma.calibrate import expected_calibration_error, fit_temperature, prior_shift
from melanoma.config import load_config, resolve_paths
from melanoma.data.dedup import _popcount, cluster_by_hash, deduplicate
from melanoma.data.splits import make_splits


def test_config_loads():
    cfg = load_config()
    paths = resolve_paths(cfg)
    assert cfg["train"]["epochs"] > 0
    assert paths.raw.name == "raw"
    assert cfg.data.image_size > 0  # Attributzugriff


def test_cluster_groups_near_duplicates():
    base = np.uint64(0b1010101010101010101010101010101010101010101010101010101010101010)
    hashes = np.array(
        [
            base,                              # 0
            base ^ np.uint64(0b11),            # 1: Distanz 2 -> gleicher Cluster wie 0
            np.uint64(0),                      # 2: weit weg
            np.uint64(0b1111),                 # 3: Distanz 4 zu 2 -> gleicher Cluster
        ],
        dtype=np.uint64,
    )
    clusters = cluster_by_hash(hashes, max_hamming=5)
    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    assert clusters[0] != clusters[2]


def test_cluster_respects_threshold():
    hashes = np.array([np.uint64(0), np.uint64(0b111111)], dtype=np.uint64)  # Distanz 6
    assert len(set(cluster_by_hash(hashes, max_hamming=5).tolist())) == 2
    assert len(set(cluster_by_hash(hashes, max_hamming=6).tolist())) == 1


def test_deduplicate_prefers_priority_source(tmp_path, monkeypatch):
    """Gleiches Bild in zwei Quellen -> nur die hoeher priorisierte bleibt."""
    frame = pd.DataFrame(
        {
            "image_path": ["a.jpg", "b.jpg", "c.jpg"],
            "source": ["mittal", "isic2020", "javed"],
            "label": [1, 1, 0],
            "patient_id": ["p1", "p2", "p3"],
            "orig_split": ["train"] * 3,
            "image_id": ["mittal/a", "isic2020/b", "javed/c"],
        }
    )
    # a und b sind dasselbe Bild, c ist verschieden.
    fake = {"a.jpg": 42, "b.jpg": 42, "c.jpg": (1 << 63) | 0xFFFF}
    monkeypatch.setattr("melanoma.data.dedup.compute_hashes", lambda paths, **kw: fake)

    kept, report = deduplicate(frame, source_priority=["isic2020", "javed", "mittal"])
    assert set(kept["image_id"]) == {"isic2020/b", "javed/c"}
    assert report["image_id"].tolist() == ["mittal/a"]
    assert report["kept_instead"].tolist() == ["isic2020/b"]


def test_deduplicate_drops_label_conflicts(monkeypatch):
    frame = pd.DataFrame(
        {
            "image_path": ["a.jpg", "b.jpg"],
            "source": ["isic2020", "mittal"],
            "label": [0, 1],
            "patient_id": ["p1", "p2"],
            "orig_split": ["train", "train"],
            "image_id": ["isic2020/a", "mittal/b"],
        }
    )
    monkeypatch.setattr(
        "melanoma.data.dedup.compute_hashes", lambda paths, **kw: {"a.jpg": 7, "b.jpg": 7}
    )
    kept, report = deduplicate(frame)
    assert kept.empty
    assert len(report) == 2


def test_split_keeps_patients_together():
    rng = np.random.default_rng(0)
    n_patients = 300
    rows = []
    for p in range(n_patients):
        label = int(rng.random() < 0.2)
        for k in range(rng.integers(1, 4)):
            rows.append(
                {
                    "image_path": f"{p}_{k}.jpg",
                    "source": "isic2020",
                    "label": label,
                    "patient_id": f"p{p}",
                    "orig_split": "train",
                    "image_id": f"isic2020/{p}_{k}",
                }
            )
    df = make_splits(pd.DataFrame(rows), val_fraction=0.15, test_fraction=0.15, seed=1)

    assert set(df["split"]) == {"train", "val", "test"}
    assert (df.groupby("patient_id")["split"].nunique() == 1).all()
    # Stratifizierung: Anteil maligner Faelle bleibt grob erhalten.
    overall = df["label"].mean()
    for _, group in df.groupby("split"):
        assert abs(group["label"].mean() - overall) < 0.12


def test_prior_shift_moves_in_the_right_direction():
    probs = np.array([0.5])
    # Von 50% Trainingspraevalenz auf 2% echte Praevalenz -> Wert muss fallen.
    assert prior_shift(probs, 0.5, 0.02)[0] < 0.05
    # Identitaet, wenn Quelle und Ziel gleich sind.
    assert prior_shift(np.array([0.3]), 0.2, 0.2)[0] == pytest.approx(0.3, abs=1e-6)


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(3)
    labels = (rng.random(2000) < 0.5).astype(float)
    # Ueberkonfidente Wahrscheinlichkeiten konstruieren.
    probs = np.where(labels == 1, rng.uniform(0.75, 0.999, 2000), rng.uniform(0.001, 0.25, 2000))
    probs = np.clip(probs + rng.normal(0, 0.05, 2000), 1e-4, 1 - 1e-4)

    temp = fit_temperature(probs, labels)
    scaled = 1.0 / (1.0 + np.exp(-np.log(probs / (1 - probs)) / temp))
    assert expected_calibration_error(scaled, labels) <= expected_calibration_error(probs, labels)


def test_transforms_produce_expected_shape():
    import torch
    from PIL import Image

    from melanoma.data.dataset import build_transforms

    img = Image.new("RGB", (600, 450), (120, 90, 80))
    for train in (True, False):
        t = build_transforms(128, [0.5] * 3, [0.5] * 3, train=train)
        out = t(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 128, 128)


def test_calibrator_json_roundtrip(tmp_path):
    """Der Kalibrator wird als JSON abgelegt - Laden muss identische Werte liefern."""
    from sklearn.isotonic import IsotonicRegression

    from melanoma.calibrate import Calibrator, load_calibrator

    rng = np.random.default_rng(7)
    probs = rng.uniform(0, 1, 500)
    labels = (rng.random(500) < probs).astype(float)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(probs, labels)
    calibrator = Calibrator.from_isotonic(iso, prevalence=float(labels.mean()))

    probe = np.linspace(0, 1, 101)
    # Unsere np.interp-Umsetzung muss sklearn exakt nachbilden.
    np.testing.assert_allclose(calibrator(probe), iso.predict(probe), atol=1e-9)

    calibrator.save(tmp_path / "calibrator.json")
    loaded = load_calibrator(tmp_path)
    assert loaded is not None
    assert loaded.method == "isotonic"
    assert loaded.prevalence == pytest.approx(calibrator.prevalence)
    np.testing.assert_allclose(loaded(probe), calibrator(probe), atol=1e-12)

    assert load_calibrator(tmp_path / "leer") is None


def test_calibrator_temperature_roundtrip(tmp_path):
    from melanoma.calibrate import Calibrator, load_calibrator

    calibrator = Calibrator.from_temperature(2.5, prevalence=0.1)
    calibrator.save(tmp_path / "calibrator.json")
    loaded = load_calibrator(tmp_path)

    probe = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(loaded(probe), calibrator(probe), atol=1e-12)
    # T > 1 zieht Werte Richtung 0.5, 0.5 bleibt fix.
    assert calibrator(np.array([0.5]))[0] == pytest.approx(0.5)
    assert calibrator(np.array([0.9]))[0] < 0.9


def test_known_duplicate_pairs_are_merged_regardless_of_hash():
    """Die offizielle ISIC-Duplikatliste ist Ground Truth und muss auch dann
    greifen, wenn die pHash-Distanz ueber dem Schwellwert liegt."""
    hashes = np.array([np.uint64(0), np.uint64(0xFFFFFFFFFFFFFFFF)], dtype=np.uint64)
    assert len(set(cluster_by_hash(hashes, max_hamming=5).tolist())) == 2
    merged = cluster_by_hash(hashes, max_hamming=5, extra_pairs=[(0, 1)])
    assert merged[0] == merged[1]


def test_deduplicate_honours_known_pairs(monkeypatch):
    frame = pd.DataFrame(
        {
            "image_path": ["a.jpg", "b.jpg"],
            "source": ["isic2020", "isic2020"],
            "label": [1, 1],
            "patient_id": ["p1", "p2"],
            "orig_split": ["train", "train"],
            "image_id": ["isic2020/A", "isic2020/B"],
        }
    )
    # Hashes weit auseinander -> ohne die Liste wuerde nichts zusammengefuehrt.
    monkeypatch.setattr(
        "melanoma.data.dedup.compute_hashes",
        lambda paths, **kw: {"a.jpg": 0, "b.jpg": 0xFFFFFFFFFFFFFFFF},
    )
    kept, _ = deduplicate(frame)
    assert len(kept) == 2

    kept, report = deduplicate(frame, known_pairs=[("isic2020/A", "isic2020/B")])
    assert len(kept) == 1
    assert report["image_id"].tolist() == ["isic2020/B"]


def test_chained_clusters_are_broken_up():
    """Single-Linkage verkettet A~B~C zu einem Cluster, auch wenn A und C weit
    auseinanderliegen. Die Verifikation muss das wieder aufbrechen."""
    # Eine Kette: jeder Nachbar 4 Bit entfernt, die Enden aber 12.
    chain = [0b0, 0b1111, 0b11111111, 0b111111111111]
    hashes = np.array(chain, dtype=np.uint64)

    clusters = cluster_by_hash(hashes, max_hamming=5)
    # Enden duerfen nicht im selben Cluster landen (Distanz 12 > 5).
    assert clusters[0] != clusters[3]
    # Jeder Cluster haelt seinen Durchmesser unter 2*max_hamming.
    for cid in set(clusters.tolist()):
        members = hashes[clusters == cid]
        for a in members:
            for b in members:
                assert _popcount(np.uint64(a) ^ np.uint64(b)) <= 10


def test_refinement_does_not_split_mutually_similar_clusters():
    """Ein echter Duplikat-Cluster, in dem alle paarweise aehnlich sind,
    muss unangetastet bleiben."""
    base = 0b1010101010101010101010101010101010101010101010101010101010101010
    hashes = np.array([base, base ^ 0b1, base ^ 0b10, base ^ 0b100], dtype=np.uint64)
    clusters = cluster_by_hash(hashes, max_hamming=5)
    assert len(set(clusters.tolist())) == 1


def test_known_pairs_survive_refinement():
    """Die offizielle Duplikatliste ist Ground Truth und darf von der
    Verkettungspruefung nicht wieder aufgetrennt werden."""
    hashes = np.array([np.uint64(0), np.uint64(0xFFFFFFFFFFFFFFFF)], dtype=np.uint64)
    merged = cluster_by_hash(hashes, max_hamming=5, extra_pairs=[(0, 1)])
    assert merged[0] == merged[1]
