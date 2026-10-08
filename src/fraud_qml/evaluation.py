"""Métricas de avaliação e as relações entre elas."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def precision_at_prevalence(
    recall: float, false_positive_rate: float, prevalence: ArrayLike
) -> NDArray[np.float64]:
    """Precisão que um classificador fixo teria num conjunto com outra proporção de positivos.

    Recall (taxa de verdadeiros positivos) e taxa de falsos positivos são calculados **dentro de
    cada classe**, então não mudam com a proporção. A precisão mistura as duas classes:

        precisão = recall·π / (recall·π + FPR·(1 − π)),   π = prevalência

    Por isso a precisão medida num teste balanceado (π = 0,5) não diz nada sobre a precisão na
    proporção real (π ≈ 0,0017): o mesmo modelo, com os mesmos erros por classe, tem precisões
    muito diferentes.

    Args:
        recall: Fração dos positivos que o classificador detecta.
        false_positive_rate: Fração dos negativos que ele marca como positivos.
        prevalence: Proporção de positivos no conjunto avaliado (escalar ou vetor).

    Returns:
        Precisão para cada prevalência.

    Raises:
        ValueError: Se alguma taxa estiver fora de [0, 1] ou se não houver nenhum positivo previsto.
    """
    prevalences = np.asarray(prevalence, dtype=float)
    for name, value in (("recall", recall), ("false_positive_rate", false_positive_rate)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} = {value} fora de [0, 1]")
    if np.any((prevalences < 0.0) | (prevalences > 1.0)):
        raise ValueError("prevalência fora de [0, 1]")
    true_positives = recall * prevalences
    false_positives = false_positive_rate * (1.0 - prevalences)
    predicted_positives = true_positives + false_positives
    if np.any(predicted_positives == 0):
        raise ValueError("precisão indefinida: nenhum positivo previsto")
    return true_positives / predicted_positives


def undersampling_corrected_probability(
    undersampled_probability: ArrayLike, kept_negative_fraction: float
) -> NDArray[np.float64]:
    """Converte a probabilidade estimada por um modelo treinado com undersampling para a proporção original.

    Com undersampling, cada negativo fica no treino com probabilidade β, e os positivos ficam todos.
    Pela regra de Bayes, a probabilidade que o modelo aprende é pₛ = p / (p + β(1 − p)); invertendo
    (Dal Pozzolo et al., 2015):

        p = β·pₛ / (β·pₛ − pₛ + 1)

    A função é estritamente crescente em pₛ: a **ordem** das transações não muda, então ROC-AUC e
    PR-AUC também não. O que muda é o significado do número e, com ele, o limiar.

    Args:
        undersampled_probability: Probabilidade de positivo estimada pelo modelo (escalar ou vetor).
        kept_negative_fraction: β, a fração dos negativos mantida pelo undersampling.

    Returns:
        Probabilidade de positivo na proporção original.

    Raises:
        ValueError: Se β não estiver em (0, 1] ou alguma probabilidade estiver fora de [0, 1].
    """
    probabilities = np.asarray(undersampled_probability, dtype=float)
    if not 0.0 < kept_negative_fraction <= 1.0:
        raise ValueError(f"fração mantida {kept_negative_fraction} fora de (0, 1]")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilidade fora de [0, 1]")
    scaled = kept_negative_fraction * probabilities
    return scaled / (scaled - probabilities + 1.0)
