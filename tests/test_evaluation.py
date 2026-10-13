"""Métricas de avaliação."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, precision_score
from sklearn.svm import SVC

from fraud_qml.evaluation import (
    best_threshold_accuracy,
    bootstrap_pr_auc,
    compute_classification_metrics,
    fraud_scores,
    percentile_interval,
    precision_at_prevalence,
    undersampling_corrected_probability,
    weighted_average_precision,
)


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


def test_classification_metrics_match_a_case_computed_by_hand() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    predictions = np.array([0, 0, 0, 1])
    metrics = compute_classification_metrics(labels, scores, predictions)
    # Curva precisão-recall: no limiar 0,8 a precisão é 1 (recall 0,5); para chegar a recall 1 ela cai a 2/3
    assert metrics.pr_auc == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)
    # Dos 4 pares (fraude, legítima), 3 estão na ordem certa
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(2 / 3)
    # VP = 1, VN = 2, FP = 0, FN = 1: (1·2 − 0·1) / √(1·2·2·3)
    assert metrics.mcc == pytest.approx(2 / np.sqrt(12))
    assert metrics.accuracy == pytest.approx(0.75)


def test_classification_metrics_without_alerts_report_zero_precision() -> None:
    labels = np.array([0, 0, 0, 1])
    metrics = compute_classification_metrics(labels, np.full(4, 0.5), np.zeros(4, dtype=int))
    assert (metrics.precision, metrics.recall, metrics.f1, metrics.mcc) == (0.0, 0.0, 0.0, 0.0)
    # Pontuação constante não ordena nada: a PR-AUC é a proporção de fraudes e a ROC-AUC é 0,5
    assert metrics.pr_auc == pytest.approx(0.25)
    assert metrics.roc_auc == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.75)


def test_fraud_scores_prefers_the_decision_function() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 2))
    labels = (features[:, 0] > 0).astype(int)
    svm = SVC().fit(features, labels)
    np.testing.assert_array_equal(fraud_scores(svm, features), svm.decision_function(features))
    dummy = DummyClassifier(strategy="prior").fit(features, labels)
    np.testing.assert_array_equal(fraud_scores(dummy, features), dummy.predict_proba(features)[:, 1])


def _noisy_scores(n_rows: int = 600, n_positives: int = 30) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = np.array([1] * n_positives + [0] * (n_rows - n_positives))
    return labels, labels + rng.normal(scale=0.8, size=n_rows)


def test_best_threshold_accuracy_matches_a_brute_force_search() -> None:
    labels, scores = _noisy_scores(n_rows=120, n_positives=50)
    scores = np.round(scores, 1)  # empates: um corte não pode separar pontuações iguais
    cuts = np.concatenate([[-np.inf], np.unique(scores)])
    brute_force = max(((scores > cut).astype(int) == labels).mean() for cut in cuts)
    assert best_threshold_accuracy(labels, scores) == pytest.approx(brute_force)


def test_best_threshold_accuracy_separates_ordering_from_cut_position() -> None:
    # A ordem é perfeita, mas todas as pontuações ficam abaixo de 0,5: o corte fixo erra todas as fraudes
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.10, 0.12, 0.15, 0.30, 0.35, 0.40])
    assert ((scores > 0.5).astype(int) == labels).mean() == pytest.approx(0.5)
    assert best_threshold_accuracy(labels, scores) == pytest.approx(1.0)
    assert best_threshold_accuracy(labels, np.zeros(6)) == pytest.approx(0.5)


def test_weighted_average_precision_matches_scikit_learn() -> None:
    rng = np.random.default_rng(0)
    labels, scores = _noisy_scores()
    scores = np.round(scores, 1)  # com empates
    for _ in range(3):
        counts = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels))).astype(float)  # inclui zeros
        expected = average_precision_score(labels, scores, sample_weight=counts)
        assert weighted_average_precision(labels, scores, counts) == pytest.approx(expected, abs=1e-12)
    assert weighted_average_precision(labels, scores, np.ones(len(labels))) == pytest.approx(
        average_precision_score(labels, scores), abs=1e-12
    )


def test_bootstrap_is_reproducible_and_brackets_the_estimate() -> None:
    labels, scores = _noisy_scores()
    samples = bootstrap_pr_auc(labels, {"modelo": scores}, n_resamples=300, random_state=0)
    assert samples.shape == (300, 1)
    lower, upper = percentile_interval(samples["modelo"])
    assert lower < average_precision_score(labels, scores) < upper
    pd.testing.assert_frame_equal(samples, bootstrap_pr_auc(labels, {"modelo": scores}, n_resamples=300, random_state=0))
    wider_lower, wider_upper = percentile_interval(samples["modelo"], confidence=0.99)
    assert wider_lower <= lower and wider_upper >= upper


def test_bootstrap_is_stratified() -> None:
    # Com pontuação constante a PR-AUC é a proporção de positivos. Se o sorteio não fosse feito dentro de
    # cada classe, a proporção mudaria de uma reamostra para outra
    labels, scores = _noisy_scores()
    samples = bootstrap_pr_auc(labels, {"constante": np.zeros(len(labels)), "perfeito": labels.astype(float)}, 50, 0)
    np.testing.assert_allclose(samples["constante"], 30 / 600)
    np.testing.assert_allclose(samples["perfeito"], 1.0)


def test_bootstrap_is_paired_across_models() -> None:
    # Os mesmos sorteios para todos: duas cópias do mesmo modelo têm diferença exatamente zero em toda reamostra
    labels, scores = _noisy_scores()
    samples = bootstrap_pr_auc(labels, {"a": scores, "b": scores.copy(), "c": -scores}, 40, 0)
    np.testing.assert_array_equal(samples["a"], samples["b"])
    assert (samples["a"] - samples["c"]).min() > 0
