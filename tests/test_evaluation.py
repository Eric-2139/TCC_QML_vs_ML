"""Métricas de avaliação."""

import numpy as np
import pytest
from sklearn.metrics import precision_score

from fraud_qml.evaluation import precision_at_prevalence, undersampling_corrected_probability


def test_precision_matches_a_confusion_matrix_built_at_that_prevalence() -> None:
    # 1.000 linhas com 10 positivos: 8 positivos detectados e 50 falsos alarmes entre 990 negativos
    labels = np.array([1] * 10 + [0] * 990)
    predictions = np.array([1] * 8 + [0] * 2 + [1] * 50 + [0] * 940)
    recall, false_positive_rate = 8 / 10, 50 / 990
    expected = precision_score(labels, predictions)
    assert precision_at_prevalence(recall, false_positive_rate, 10 / 1000) == pytest.approx(expected)


def test_balanced_test_set_inflates_precision() -> None:
    # O mesmo classificador: quase perfeito num teste 50/50, fraco na proporção real de fraudes
    balanced, real = precision_at_prevalence(0.9, 0.01, [0.5, 473 / 283_726])
    assert balanced > 0.95
    assert real < 0.2


def test_perfect_specificity_gives_precision_one() -> None:
    np.testing.assert_allclose(precision_at_prevalence(0.5, 0.0, [0.001, 0.5]), 1.0)


@pytest.mark.parametrize(
    ("recall", "false_positive_rate", "prevalence"),
    [(1.2, 0.1, 0.5), (0.5, -0.1, 0.5), (0.5, 0.1, 1.5), (0.0, 0.0, 0.5)],
)
def test_invalid_inputs_are_rejected(recall: float, false_positive_rate: float, prevalence: float) -> None:
    with pytest.raises(ValueError):
        precision_at_prevalence(recall, false_positive_rate, prevalence)


def test_balanced_half_maps_back_to_the_original_prevalence() -> None:
    # 10 positivos e 990 negativos; o undersampling 1:1 mantém β = 10/990 dos negativos. Um modelo que
    # só aprendeu a proporção do treino balanceado diz 0,5; corrigido, volta à proporção original 10/1000
    corrected = undersampling_corrected_probability(0.5, kept_negative_fraction=10 / 990)
    assert corrected == pytest.approx(10 / 1000)


def test_correction_undoes_bayes_under_undersampling() -> None:
    # pₛ = p / (p + β(1 − p)) é o que o modelo aprende; a correção precisa devolver p
    original = np.linspace(0.001, 0.999, 50)
    kept = 0.02
    undersampled = original / (original + kept * (1 - original))
    np.testing.assert_allclose(undersampling_corrected_probability(undersampled, kept), original)


def test_correction_preserves_order_and_is_identity_without_undersampling() -> None:
    probabilities = np.array([0.1, 0.4, 0.8, 0.99])
    corrected = undersampling_corrected_probability(probabilities, 0.01)
    assert np.all(np.diff(corrected) > 0)
    np.testing.assert_allclose(undersampling_corrected_probability(probabilities, 1.0), probabilities)


@pytest.mark.parametrize(("probability", "kept"), [(0.5, 0.0), (0.5, 1.5), (1.2, 0.5)])
def test_correction_rejects_invalid_inputs(probability: float, kept: float) -> None:
    with pytest.raises(ValueError):
        undersampling_corrected_probability(probability, kept)
