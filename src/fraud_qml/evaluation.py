"""Métricas de avaliação e as relações entre elas."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_qml.data import FRAUD_LABEL


@dataclass(frozen=True)
class ClassificationMetrics:
    """Métricas de um classificador binário num conjunto de avaliação.

    As duas primeiras usam só a **ordem** das pontuações e não dependem de limiar; as outras
    cinco avaliam as classes previstas com um limiar já aplicado.

    Attributes:
        pr_auc: Área sob a curva precisão-recall (*average precision*); métrica principal do TCC.
        roc_auc: Área sob a curva ROC.
        precision: Fração dos alertas que são fraude.
        recall: Fração das fraudes que viraram alerta.
        f1: Média harmônica de precisão e recall.
        mcc: Coeficiente de correlação de Matthews, de −1 a 1; usa as quatro células da matriz de confusão.
        accuracy: Fração de acertos; com 0,17% de fraudes, só faz sentido ao lado das outras.
    """

    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    mcc: float
    accuracy: float


def compute_classification_metrics(labels: ArrayLike, scores: ArrayLike, predictions: ArrayLike) -> ClassificationMetrics:
    """Calcula as métricas do TCC com a fraude como classe positiva.

    Uma única função para todos os modelos, com a classe positiva explícita: assim clássicos e
    quânticos não podem ser medidos com médias diferentes (AC3).

    Args:
        labels: Rótulos verdadeiros.
        scores: Pontuação de cada linha (maior = mais suspeita de fraude).
        predictions: Classe prevista para cada linha.

    Returns:
        As sete métricas. Sem nenhum alerta, a precisão (indefinida) é reportada como 0.
    """
    return ClassificationMetrics(
        pr_auc=float(average_precision_score(labels, scores, pos_label=FRAUD_LABEL)),
        roc_auc=float(roc_auc_score(labels, scores)),
        precision=float(precision_score(labels, predictions, pos_label=FRAUD_LABEL, zero_division=0)),
        recall=float(recall_score(labels, predictions, pos_label=FRAUD_LABEL)),
        f1=float(f1_score(labels, predictions, pos_label=FRAUD_LABEL, zero_division=0)),
        mcc=float(matthews_corrcoef(labels, predictions)),
        accuracy=float(accuracy_score(labels, predictions)),
    )


def best_threshold_accuracy(labels: ArrayLike, scores: ArrayLike) -> float:
    """Maior acurácia que um único corte na pontuação consegue nestas linhas.

    Separa duas limitações de um classificador: **ordenar** mal as linhas (nenhum corte resolve) e
    **posicionar** mal o corte (a ordem está certa, mas o limiar fixo cai no lugar errado). Medida
    nas mesmas linhas em que o corte é procurado, ela é um teto, e não uma estimativa de desempenho.

    Args:
        labels: Rótulos verdadeiros.
        scores: Pontuação de cada linha (maior = mais suspeita de fraude).

    Returns:
        A maior acurácia entre todos os cortes possíveis (inclusive "nenhum alerta" e "tudo é alerta").
    """
    values = np.asarray(scores, dtype=float)
    is_fraud = np.asarray(labels) == FRAUD_LABEL
    order = np.argsort(-values, kind="stable")
    # Um corte só pode cair entre pontuações diferentes: cada posição é o fim de um grupo de empate
    cut_positions = np.append(np.flatnonzero(np.diff(values[order])), len(values) - 1)
    true_positives = np.cumsum(is_fraud[order])[cut_positions]
    false_positives = (cut_positions + 1) - true_positives
    n_legitimate = int((~is_fraud).sum())
    accuracies = (true_positives + n_legitimate - false_positives) / len(values)
    return float(max(accuracies.max(), n_legitimate / len(values)))


def _tie_groups(scores: ArrayLike) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    values = np.asarray(scores, dtype=float)
    order = np.argsort(-values, kind="stable")
    return order, np.append(np.flatnonzero(np.diff(values[order])), len(values) - 1)


def _average_precision_from_sorted(
    sorted_weights: NDArray[np.float64], sorted_is_fraud: NDArray[np.bool_], group_ends: NDArray[np.intp]
) -> float:
    true_positives = np.cumsum(sorted_weights * sorted_is_fraud)[group_ends]
    alerts = np.cumsum(sorted_weights)[group_ends]
    precision = np.divide(true_positives, alerts, out=np.zeros_like(true_positives), where=alerts > 0)
    recall = true_positives / true_positives[-1]
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def weighted_average_precision(labels: ArrayLike, scores: ArrayLike, sample_weight: ArrayLike) -> float:
    """PR-AUC (*average precision*) com um peso por linha; igual à do scikit-learn (há teste).

    Existe para o bootstrap: reamostrar com reposição é dar a cada linha um peso inteiro (quantas
    vezes ela foi sorteada), e com as pontuações ordenadas uma única vez cada reamostra custa só
    duas somas acumuladas.

    Args:
        labels: Rótulos verdadeiros.
        scores: Pontuação de cada linha (maior = mais suspeita de fraude).
        sample_weight: Peso de cada linha (não negativo; pelo menos uma fraude com peso positivo).

    Returns:
        Soma, ao longo dos limiares, do ganho de recall multiplicado pela precisão naquele limiar.
    """
    order, group_ends = _tie_groups(scores)
    is_fraud = np.asarray(labels) == FRAUD_LABEL
    return _average_precision_from_sorted(np.asarray(sample_weight, dtype=float)[order], is_fraud[order], group_ends)


def bootstrap_pr_auc(
    labels: ArrayLike, scores_by_model: Mapping[str, ArrayLike], n_resamples: int, random_state: int
) -> pd.DataFrame:
    """PR-AUC de cada modelo em reamostras estratificadas do conjunto de avaliação (bootstrap).

    O bootstrap sorteia, com reposição, um conjunto do mesmo tamanho e recalcula a métrica; a
    dispersão entre os sorteios estima quanto a métrica mudaria com outro conjunto de avaliação.

    - **Estratificado:** o sorteio é feito dentro de cada classe. A PR-AUC depende da proporção de
      fraudes, e um sorteio simples, com tão poucas fraudes, faria a proporção variar.
    - **Pareado:** todos os modelos são avaliados **nas mesmas reamostras**. A diferença entre duas
      colunas, linha a linha, é a distribuição da diferença entre os dois modelos, que é mais
      informativa do que comparar os intervalos de cada um.

    Mede só a incerteza do conjunto de avaliação. Não inclui a do treino (outro sorteio do
    undersampling daria outro modelo).

    Args:
        labels: Rótulos verdadeiros.
        scores_by_model: Nome do modelo → pontuação de cada linha.
        n_resamples: Número de reamostras.
        random_state: Semente.

    Returns:
        Uma linha por reamostra e uma coluna por modelo.
    """
    is_fraud = np.asarray(labels) == FRAUD_LABEL
    class_rows = [np.flatnonzero(is_fraud), np.flatnonzero(~is_fraud)]
    tie_groups = {name: _tie_groups(scores) for name, scores in scores_by_model.items()}
    rng = np.random.default_rng(random_state)
    estimates = {name: np.empty(n_resamples) for name in scores_by_model}
    for resample in range(n_resamples):
        # Quantas vezes cada linha foi sorteada: equivale a sortear índices com reposição dentro da classe
        counts = np.zeros(len(is_fraud))
        for rows in class_rows:
            counts[rows] = rng.multinomial(len(rows), np.full(len(rows), 1.0 / len(rows)))
        for name, (order, group_ends) in tie_groups.items():
            estimates[name][resample] = _average_precision_from_sorted(counts[order], is_fraud[order], group_ends)
    return pd.DataFrame(estimates)


def percentile_interval(samples: ArrayLike, confidence: float = 0.95) -> tuple[float, float]:
    """Intervalo de confiança pelo método do percentil (os dois percentis que deixam ``confidence`` no meio)."""
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(np.asarray(samples, dtype=float), [tail, 1.0 - tail])
    return float(lower), float(upper)


def fraud_scores(model: BaseEstimator, features: ArrayLike) -> NDArray[np.float64]:
    """Pontuação de fraude de cada linha: ``decision_function`` quando existe, senão P(fraude).

    Args:
        model: Classificador ajustado, com a fraude como segunda classe.
        features: Linhas a pontuar.

    Returns:
        Uma pontuação por linha (maior = mais suspeita).
    """
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


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
