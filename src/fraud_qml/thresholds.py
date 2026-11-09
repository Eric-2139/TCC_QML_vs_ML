"""Limiar de decisão (Seção 7): de uma pontuação contínua para "alerta" ou "não alerta".

Os modelos devolvem uma pontuação (maior = mais suspeita), e o limiar decide a partir de que valor uma transação vira
alerta. O limiar padrão de cada modelo é o de um modelo treinado com 50% de fraudes (Seção 4.10); na proporção real ele
gera alertas demais. O protocolo escolhe o limiar **na validação, pelo F1 máximo** (PLAN, decisões tomadas).

Este módulo fica fora do que a busca do grid importa (``tuning.package_dependencies``): mudá-lo não invalida o cache.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from fraud_qml.data import FRAUD_LABEL, LEGITIMATE_LABEL

THRESHOLD_CURVE_COLUMNS: Final[tuple[str, ...]] = ("threshold", "alerts", "true_positives", "precision", "recall", "f1")


@dataclass(frozen=True)
class ThresholdChoice:
    """O limiar de maior F1 e o que ele produz no conjunto em que foi escolhido.

    Attributes:
        threshold: Uma linha vira alerta quando a pontuação é maior ou igual a ele.
        alerts: Quantas linhas viram alerta.
        true_positives: Quantos alertas são fraude.
        precision: Fração dos alertas que são fraude.
        recall: Fração das fraudes que viraram alerta.
        f1: Média harmônica de precisão e recall.
    """

    threshold: float
    alerts: int
    true_positives: int
    precision: float
    recall: float
    f1: float


def threshold_curve(labels: ArrayLike, scores: ArrayLike) -> pd.DataFrame:
    """Precisão, recall e F1 em cada limiar possível, do mais alto (menos alertas) ao mais baixo.

    Um limiar só muda o resultado quando passa de uma pontuação para outra, então os limiares possíveis são as
    pontuações distintas; linhas empatadas viram alerta juntas. O F1 é calculado como 2·VP / (alertas + fraudes), que é
    a mesma média harmônica 2·P·R / (P + R) escrita com contagens (VP = verdadeiros positivos).

    Args:
        labels: Rótulos verdadeiros.
        scores: Pontuação de cada linha (maior = mais suspeita de fraude).

    Returns:
        Uma linha por limiar, com as colunas de ``THRESHOLD_CURVE_COLUMNS``.

    Raises:
        ValueError: Se não houver nenhuma fraude (recall e F1 indefinidos).
    """
    values = np.asarray(scores, dtype=float)
    is_fraud = np.asarray(labels) == FRAUD_LABEL
    n_frauds = int(is_fraud.sum())
    if n_frauds == 0:
        raise ValueError("sem fraudes, recall e F1 são indefinidos")
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    # Cada corte cai no fim de um grupo de pontuações iguais
    group_ends = np.append(np.flatnonzero(np.diff(sorted_values)), len(values) - 1)
    true_positives = np.cumsum(is_fraud[order])[group_ends]
    alerts = group_ends + 1
    return pd.DataFrame(
        {
            "threshold": sorted_values[group_ends],
            "alerts": alerts,
            "true_positives": true_positives,
            "precision": true_positives / alerts,
            "recall": true_positives / n_frauds,
            "f1": 2 * true_positives / (alerts + n_frauds),
        },
        columns=list(THRESHOLD_CURVE_COLUMNS),
    )


def best_f1_threshold(labels: ArrayLike, scores: ArrayLike) -> ThresholdChoice:
    """O limiar de maior F1 nestas linhas.

    No empate de F1, fica o limiar mais alto (menos alertas). Medido nas mesmas linhas em que foi escolhido, o F1
    devolvido é otimista: é o melhor que algum limiar conseguiu ali, e não uma estimativa para dados novos.

    Args:
        labels: Rótulos verdadeiros.
        scores: Pontuação de cada linha (maior = mais suspeita de fraude).

    Returns:
        O limiar e o resultado dele.
    """
    curve = threshold_curve(labels, scores)
    # argmax devolve o primeiro máximo, e a curva vai do limiar mais alto ao mais baixo
    best = curve.iloc[int(np.argmax(curve["f1"].to_numpy()))]
    return ThresholdChoice(
        threshold=float(best["threshold"]),
        alerts=int(best["alerts"]),
        true_positives=int(best["true_positives"]),
        precision=float(best["precision"]),
        recall=float(best["recall"]),
        f1=float(best["f1"]),
    )


def predictions_at_threshold(scores: ArrayLike, threshold: float) -> NDArray[np.int_]:
    """Classe prevista com um limiar: fraude quando a pontuação é maior ou igual a ele."""
    return np.where(np.asarray(scores, dtype=float) >= threshold, FRAUD_LABEL, LEGITIMATE_LABEL)
