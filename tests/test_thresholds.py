"""Limiar por F1 máximo (Seção 7) e relatório em JSON."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from fraud_qml.reporting import json_ready, write_json_report
from fraud_qml.thresholds import best_f1_threshold, predictions_at_threshold, threshold_curve

SMALL_LABELS = np.array([1, 0, 1, 0, 0])
SMALL_SCORES = np.array([0.9, 0.8, 0.7, 0.2, 0.1])


def test_small_example_by_hand() -> None:
    # Cortes: ≥0,9 → F1 2/3; ≥0,8 → 1/2; ≥0,7 → 4/5 (o melhor); ≥0,2 → 2/3; ≥0,1 → 4/7
    choice = best_f1_threshold(SMALL_LABELS, SMALL_SCORES)
    assert choice.threshold == 0.7 and choice.alerts == 3 and choice.true_positives == 2
    assert choice.f1 == pytest.approx(0.8) and choice.precision == pytest.approx(2 / 3) and choice.recall == 1.0
    assert threshold_curve(SMALL_LABELS, SMALL_SCORES)["f1"].tolist() == pytest.approx([2 / 3, 1 / 2, 4 / 5, 2 / 3, 4 / 7])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_curve_matches_sklearn_at_every_cut_and_the_best_is_the_maximum(seed: int) -> None:
    rng = np.random.default_rng(seed)
    labels = (rng.random(400) < 0.1).astype(int)
    # Pontuações arredondadas: força empates, que precisam virar alerta juntos
    scores = np.round(labels * 0.8 + rng.normal(size=400), 1)
    curve = threshold_curve(labels, scores)
    assert len(curve) == len(np.unique(scores))
    for _, row in curve.iterrows():
        predictions = predictions_at_threshold(scores, row["threshold"])
        assert int(predictions.sum()) == row["alerts"]
        assert row["f1"] == pytest.approx(f1_score(labels, predictions))
        assert row["precision"] == pytest.approx(precision_score(labels, predictions))
        assert row["recall"] == pytest.approx(recall_score(labels, predictions))
    assert best_f1_threshold(labels, scores).f1 == pytest.approx(curve["f1"].max())


def test_a_tie_in_f1_keeps_the_higher_threshold() -> None:
    # ≥0,9 → VP 1, 1 alerta: F1 = 2/(1+2) = 2/3; ≥0,6 → VP 2, 4 alertas: F1 = 4/(4+2) = 2/3
    labels = np.array([1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert threshold_curve(labels, scores)["f1"].tolist() == pytest.approx([2 / 3, 1 / 2, 2 / 5, 2 / 3, 4 / 7])
    assert best_f1_threshold(labels, scores).threshold == 0.9


def test_without_frauds_the_threshold_is_undefined() -> None:
    with pytest.raises(ValueError):
        best_f1_threshold(np.zeros(5, dtype=int), np.arange(5.0))


def test_report_converts_numpy_tuples_and_nan(tmp_path: Path) -> None:
    payload = {"a": np.float64(0.5), "b": (np.int64(1), 2), "c": float("nan"), "d": np.array([1.0, 2.0]), 3: "x"}
    assert json_ready(payload) == {"a": 0.5, "b": [1, 2], "c": None, "d": [1.0, 2.0], "3": "x"}
    target = tmp_path / "sub" / "report.json"
    write_json_report(target, {"métrica": np.float32(0.25), "nada": math.nan})
    assert json.loads(target.read_text(encoding="utf-8")) == {"métrica": 0.25, "nada": None}
