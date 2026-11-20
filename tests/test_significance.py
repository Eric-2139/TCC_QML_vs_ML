"""McNemar exato e correção de Holm (Seção 9), contra contas feitas à mão."""

import math

import numpy as np
import pytest

from fraud_qml.data import FRAUD_LABEL, LEGITIMATE_LABEL
from fraud_qml.significance import holm_adjusted, mcnemar_exact


def _predictions_with_discordants(only_first: int, only_second: int, agreeing: int) -> tuple[np.ndarray, ...]:
    """Rótulos e duas previsões com o número pedido de discordantes; todas as linhas são fraudes."""
    n_rows = only_first + only_second + agreeing
    labels = np.full(n_rows, FRAUD_LABEL)
    first = labels.copy()
    second = labels.copy()
    second[:only_first] = LEGITIMATE_LABEL
    first[only_first : only_first + only_second] = LEGITIMATE_LABEL
    return labels, first, second


def test_mcnemar_matches_the_binomial_by_hand() -> None:
    # b = 1, c = 7: p bilateral = 2 · P(X ≤ 1), X ~ Binomial(8, 1/2) = 2 · (1 + 8) / 256
    labels, first, second = _predictions_with_discordants(only_first=1, only_second=7, agreeing=30)
    result = mcnemar_exact(labels, first, second)
    assert (result.first_only_correct, result.second_only_correct, result.discordant) == (1, 7, 8)
    assert result.p_value == pytest.approx(2 * 9 / 256, abs=1e-15)


def test_mcnemar_is_symmetric_in_the_two_models() -> None:
    labels, first, second = _predictions_with_discordants(only_first=2, only_second=9, agreeing=10)
    forward, backward = mcnemar_exact(labels, first, second), mcnemar_exact(labels, second, first)
    assert (forward.first_only_correct, forward.second_only_correct) == (backward.second_only_correct, backward.first_only_correct)
    assert forward.p_value == pytest.approx(backward.p_value, abs=1e-15)


def test_mcnemar_without_discordant_rows_gives_one() -> None:
    labels = np.array([FRAUD_LABEL, LEGITIMATE_LABEL, LEGITIMATE_LABEL])
    predictions = np.array([FRAUD_LABEL, FRAUD_LABEL, LEGITIMATE_LABEL])
    result = mcnemar_exact(labels, predictions, predictions)
    assert result.discordant == 0
    assert result.p_value == 1.0


def test_mcnemar_with_balanced_discordants_gives_one() -> None:
    labels, first, second = _predictions_with_discordants(only_first=5, only_second=5, agreeing=3)
    assert mcnemar_exact(labels, first, second).p_value == pytest.approx(1.0, abs=1e-12)


def test_mcnemar_splits_the_discordants_by_class() -> None:
    labels = np.array([FRAUD_LABEL, FRAUD_LABEL, LEGITIMATE_LABEL, LEGITIMATE_LABEL, LEGITIMATE_LABEL])
    # Linha 0: só o primeiro alerta a fraude. Linha 1: só o segundo. Linhas 2 e 3: o segundo dá alarme falso. Linha 4: nenhum
    first = np.array([FRAUD_LABEL, LEGITIMATE_LABEL, LEGITIMATE_LABEL, LEGITIMATE_LABEL, LEGITIMATE_LABEL])
    second = np.array([LEGITIMATE_LABEL, FRAUD_LABEL, FRAUD_LABEL, FRAUD_LABEL, LEGITIMATE_LABEL])
    result = mcnemar_exact(labels, first, second)
    assert (result.first_only_correct, result.first_only_correct_frauds, result.first_only_correct_legitimate) == (3, 1, 2)
    assert (result.second_only_correct, result.second_only_correct_frauds, result.second_only_correct_legitimate) == (1, 1, 0)


def test_mcnemar_rejects_vectors_of_different_sizes() -> None:
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        mcnemar_exact(np.zeros(3), np.zeros(3), np.zeros(4))


def test_holm_matches_the_step_down_by_hand() -> None:
    # Ordenados: 0,005 · 4 = 0,02; 0,01 · 3 = 0,03; 0,03 · 2 = 0,06; 0,04 · 1 = 0,04 → 0,06 (não pode cair)
    adjusted = holm_adjusted([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02], atol=1e-15)


def test_holm_never_lowers_a_p_value_and_caps_at_one() -> None:
    raw = np.array([0.2, 0.001, 0.7, 0.04, 0.5, 0.012, 0.3, 0.9])
    adjusted = holm_adjusted(raw)
    assert np.all(adjusted >= raw)
    assert np.all(adjusted <= 1.0)
    # A ordem dos p ajustados acompanha a dos originais
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)


def test_holm_with_one_comparison_changes_nothing() -> None:
    assert holm_adjusted([0.037])[0] == pytest.approx(0.037)


def test_holm_is_at_most_bonferroni() -> None:
    raw = np.array([0.01, 0.02, 0.03])
    assert np.all(holm_adjusted(raw) <= np.minimum(raw * len(raw), 1.0) + 1e-15)
    assert math.isclose(float(holm_adjusted(raw).min()), 0.03)
