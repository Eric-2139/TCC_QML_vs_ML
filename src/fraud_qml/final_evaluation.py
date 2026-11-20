"""Avaliação final (Seção 9): pontuar com modelos já ajustados e a robustez nas sementes do undersampling.

**Sem reajuste no teste.** O teste avalia os modelos da Seção 7 como foram validados (CC20). O ``fit_and_score`` do
``models`` ajusta antes de pontuar; ``score_fitted_pipeline`` faz só a segunda metade, com o mesmo laço de
transformações, para que a pontuação seja a mesma conta (há teste) e o custo seja medido do mesmo jeito.

**Robustez.** Em cada semente de ``config.UNDERSAMPLING_SEEDS``, cada configuração congelada é ajustada no treino
inteiro com a semente (que sorteia o undersampling, o redutor e o boosting), escolhe o **próprio** limiar de F1 máximo na
validação e é medida no teste nesse limiar. Com a semente do experimento, o resultado é o modelo e o limiar da Seção 7.

Os dois laços repetem o do ``models.fit_and_score`` em vez de reaproveitá-lo porque o ``models`` está na chave do cache
da Seção 6: mudá-lo refaria as buscas. Este módulo fica fora do que a busca importa (``tuning.package_dependencies``).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import AdaBoostClassifier

from fraud_qml import config
from fraud_qml.data import FRAUD_LABEL
from fraud_qml.evaluation import compute_classification_metrics, fraud_scores
from fraud_qml.models import InputEncoding, ModelSettings, build_pipeline
from fraud_qml.profiling import CostMeter, CostReport
from fraud_qml.thresholds import best_f1_threshold, predictions_at_threshold

ROBUSTNESS_COLUMNS: Final[tuple[str, ...]] = (
    "pipeline",
    "seed",
    "validation_pr_auc",
    "validation_roc_auc",
    "threshold",
    "validation_alerts",
    "validation_f1",
    "test_pr_auc",
    "test_roc_auc",
    "test_alerts",
    "test_true_positives",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_mcc",
    "test_accuracy",
    "kept_estimators",
    "fit_seconds",
    "score_seconds",
)


@dataclass(frozen=True)
class FittedScores:
    """Pontuação de um conjunto por um pipeline já ajustado.

    Attributes:
        scores: Pontuação de fraude de cada linha (maior = mais suspeita).
        predictions: Classe prevista com o limiar padrão do modelo.
        predict_cost: Custo de pontuar (pré-processamento, codificação e modelo).
    """

    scores: NDArray[np.float64]
    predictions: NDArray[np.int_]
    predict_cost: CostReport


def score_fitted_pipeline(pipeline: ImbPipeline, features: ArrayLike) -> FittedScores:
    """Pontua um conjunto com um pipeline já ajustado, sem reajustar nada.

    É a segunda metade do ``models.fit_and_score``: o undersampler é pulado (só age no ajuste), as outras etapas
    transformam, e a classe prevista sai das mesmas linhas transformadas, fora da medição do tempo.

    Args:
        pipeline: Pipeline ajustado.
        features: Linhas a pontuar, no mesmo tipo de dado do ajuste (CC21: no Quantum AdaBoost, tabela e array dão
            arredondamentos diferentes).

    Returns:
        Pontuações, classes previstas no limiar padrão e o custo.
    """
    model = pipeline.named_steps["model"]
    with CostMeter(trace_python_memory=False) as meter:
        transformed = features
        for _, step in pipeline.steps[:-1]:
            if not hasattr(step, "fit_resample"):
                transformed = step.transform(transformed)
        scores = fraud_scores(model, transformed)
    return FittedScores(scores=scores, predictions=np.asarray(model.predict(transformed)), predict_cost=meter.report)


def robustness_point(
    input_encoding: InputEncoding,
    model: ModelSettings,
    seed: int,
    train_features: ArrayLike,
    train_labels: ArrayLike,
    validation_features: ArrayLike,
    validation_labels: ArrayLike,
    test_features: ArrayLike,
    test_labels: ArrayLike,
) -> dict[str, object]:
    """Uma configuração com uma semente: ajusta no treino, escolhe o limiar na validação e mede no teste.

    Args:
        input_encoding: Entrada e codificação (congeladas na Seção 6).
        model: Modelo (congelado na Seção 6).
        seed: Semente do undersampling, do redutor e do boosting.
        train_features: Treino inteiro (o tipo é preservado).
        train_labels: Rótulos do treino.
        validation_features: Validação.
        validation_labels: Rótulos da validação (os únicos que escolhem o limiar).
        test_features: Teste.
        test_labels: Rótulos do teste (só medem).

    Returns:
        As colunas de ``ROBUSTNESS_COLUMNS``, menos ``pipeline``.
    """
    pipeline = build_pipeline(input_encoding, model, seed)
    with CostMeter(trace_python_memory=False) as fit_meter:
        pipeline.fit(train_features, train_labels)
    validation = score_fitted_pipeline(pipeline, validation_features)
    test = score_fitted_pipeline(pipeline, test_features)
    validation_target, test_target = np.asarray(validation_labels), np.asarray(test_labels)
    choice = best_f1_threshold(validation_target, validation.scores)
    test_predictions = predictions_at_threshold(test.scores, choice.threshold)
    validation_metrics = compute_classification_metrics(
        validation_target, validation.scores, predictions_at_threshold(validation.scores, choice.threshold)
    )
    test_metrics = compute_classification_metrics(test_target, test.scores, test_predictions)
    is_alert = test_predictions == FRAUD_LABEL
    fitted_model = pipeline.named_steps["model"]
    return {
        "seed": seed,
        "validation_pr_auc": validation_metrics.pr_auc,
        "validation_roc_auc": validation_metrics.roc_auc,
        "threshold": choice.threshold,
        "validation_alerts": choice.alerts,
        "validation_f1": choice.f1,
        "test_pr_auc": test_metrics.pr_auc,
        "test_roc_auc": test_metrics.roc_auc,
        "test_alerts": int(is_alert.sum()),
        "test_true_positives": int((is_alert & (test_target == FRAUD_LABEL)).sum()),
        "test_precision": test_metrics.precision,
        "test_recall": test_metrics.recall,
        "test_f1": test_metrics.f1,
        "test_mcc": test_metrics.mcc,
        "test_accuracy": test_metrics.accuracy,
        "kept_estimators": float(len(fitted_model.estimators_)) if isinstance(fitted_model, AdaBoostClassifier) else np.nan,
        "fit_seconds": fit_meter.report.wall_seconds,
        "score_seconds": validation.predict_cost.wall_seconds + test.predict_cost.wall_seconds,
    }


def robustness_scores(
    configurations: Mapping[str, tuple[InputEncoding, ModelSettings]],
    seeds: Sequence[int],
    train_features: ArrayLike,
    train_labels: ArrayLike,
    validation_features: ArrayLike,
    validation_labels: ArrayLike,
    test_features: ArrayLike,
    test_labels: ArrayLike,
    n_jobs: int = config.N_JOBS,
) -> pd.DataFrame:
    """Robustez: cada configuração com cada semente, em paralelo.

    Args:
        configurations: Nome do pipeline → (entrada, modelo).
        seeds: Sementes do undersampling (``config.UNDERSAMPLING_SEEDS``).
        train_features: Treino inteiro.
        train_labels: Rótulos do treino.
        validation_features: Validação.
        validation_labels: Rótulos da validação.
        test_features: Teste.
        test_labels: Rótulos do teste.
        n_jobs: Processos paralelos; o resultado não depende deste valor.

    Returns:
        Uma linha por pipeline e semente, com as colunas de ``ROBUSTNESS_COLUMNS``.
    """
    tasks = [(name, seed) for name in configurations for seed in seeds]
    points = Parallel(n_jobs=n_jobs)(
        delayed(robustness_point)(
            *configurations[name], seed, train_features, train_labels, validation_features, validation_labels,
            test_features, test_labels,
        )
        for name, seed in tasks
    )
    rows = [{"pipeline": name, **point} for (name, _), point in zip(tasks, points, strict=True)]
    return pd.DataFrame(rows, columns=list(ROBUSTNESS_COLUMNS))
