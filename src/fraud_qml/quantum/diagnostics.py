"""Varreduras de diagnóstico: concentração e alinhamento dos kernels (KTA) e codificação no classificador fraco.

Existem como funções do pacote, e não como laços no notebook, para que o procedimento que
delimita as opções do grid (Seções 4 e 5 do notebook, só no treino) seja testado e rode em paralelo.
"""

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold, cross_validate

from fraud_qml.models import build_weak_quantum_classifier_pipeline, quantum_encoding_steps
from fraud_qml.preprocessing import (
    ReducerName,
    ScalerName,
    build_amplitude_preprocessor,
    build_angle_preprocessor,
)
from fraud_qml.quantum.encodings import AngleFeatureMapName, build_angle_feature_map
from fraud_qml.quantum.kernels import (
    amplitude_kernel_matrix,
    angle_kernel_matrix,
    centered_kernel_target_alignment,
    off_diagonal_values,
)

KernelSubsample = tuple[NDArray[np.float64], NDArray[np.int_]]
AngleConfiguration = tuple[AngleFeatureMapName, ReducerName]

ANGLE_SWEEP_COLUMNS: Final[tuple[str, ...]] = (
    "configuration",
    "range_width",
    "subsample",
    "mean_similarity",
    "alignment",
)
AMPLITUDE_SWEEP_COLUMNS: Final[tuple[str, ...]] = (
    "scaler",
    "padding_value",
    "subsample",
    "mean_similarity",
    "alignment",
)

# (feature map, repetições do feature map)
EncodingConfiguration = tuple[AngleFeatureMapName, int]

WEAK_CLASSIFIER_SWEEP_COLUMNS: Final[tuple[str, ...]] = (
    "configuration",
    "reducer",
    "scaler",
    "range_width",
    "ansatz_reps",
    "subsample",
    "initialization_seed",
    "fold",
    "roc_auc",
    "accuracy",
    "train_accuracy",
)


def _kernel_scores(kernel: NDArray[np.float64], labels: NDArray[np.int_]) -> tuple[float, float]:
    return float(off_diagonal_values(kernel).mean()), centered_kernel_target_alignment(kernel, labels)


def _angle_scores(
    subsample: KernelSubsample,
    configuration: AngleConfiguration,
    range_width: float,
    scaler: ScalerName,
    n_qubits: int,
    random_state: int,
) -> tuple[float, float]:
    features, labels = subsample
    map_name, reducer = configuration
    # O circuito é montado dentro da tarefa: cada processo do joblib cria o seu, sem serializar objetos do Qiskit
    angles = build_angle_preprocessor(scaler, reducer, n_qubits, (0.0, range_width), random_state).fit_transform(
        features, labels
    )
    return _kernel_scores(angle_kernel_matrix(build_angle_feature_map(map_name, n_qubits), angles), labels)


def angle_alignment_sweep(
    subsamples: Mapping[int, KernelSubsample],
    configurations: Mapping[str, AngleConfiguration],
    range_widths: Sequence[float],
    n_qubits: int,
    random_state: int,
    scaler: ScalerName = "standard",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Mede similaridade média e KTA da codificação de ângulo para cada combinação.

    Cada combinação (configuração × largura da faixa × subamostra) ajusta o pré-processamento na
    própria subamostra e calcula a matriz de Gram exata. As combinações são independentes e rodam
    em paralelo; o resultado não depende de ``n_jobs``.

    Args:
        subsamples: Subamostras balanceadas (features, rótulos), identificadas pela semente que as gerou.
        configurations: Nome de exibição → (feature map, redutor).
        range_widths: Larguras da faixa de ângulos; a faixa é [0, largura].
        n_qubits: Número de qubits (= componentes do redutor).
        random_state: Semente do redutor.
        scaler: Scaler de entrada.
        n_jobs: Processos paralelos.

    Returns:
        Uma linha por combinação, com as colunas de ``ANGLE_SWEEP_COLUMNS``.
    """
    tasks = [
        (name, width, seed)
        for name in configurations
        for width in range_widths
        for seed in subsamples
    ]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_angle_scores)(subsamples[seed], configurations[name], width, scaler, n_qubits, random_state)
        for name, width, seed in tasks
    )
    rows = [(*task, similarity, alignment) for task, (similarity, alignment) in zip(tasks, scores, strict=True)]
    return pd.DataFrame(rows, columns=list(ANGLE_SWEEP_COLUMNS))


def amplitude_alignment_sweep(
    subsamples: Mapping[int, KernelSubsample],
    scalers: Sequence[ScalerName],
    padding_values: Sequence[float],
    n_qubits: int,
) -> pd.DataFrame:
    """Mede similaridade média e KTA da codificação de amplitude para cada combinação.

    Não há paralelismo: o kernel de amplitude é um produto de matrizes clássico e custa milissegundos.

    Args:
        subsamples: Subamostras balanceadas (features, rótulos), identificadas pela semente que as gerou.
        scalers: Scalers de entrada.
        padding_values: Constantes usadas para completar até 2ⁿ.
        n_qubits: Número de qubits.

    Returns:
        Uma linha por combinação, com as colunas de ``AMPLITUDE_SWEEP_COLUMNS``.
    """
    rows: list[tuple[str, float, int, float, float]] = []
    for scaler in scalers:
        for padding_value in padding_values:
            for seed, (features, labels) in subsamples.items():
                amplitudes = build_amplitude_preprocessor(scaler, n_qubits, padding_value).fit_transform(features)
                rows.append((scaler, padding_value, seed, *_kernel_scores(amplitude_kernel_matrix(amplitudes), labels)))
    return pd.DataFrame(rows, columns=list(AMPLITUDE_SWEEP_COLUMNS))


def _weak_classifier_fold_scores(
    subsample: KernelSubsample,
    configuration: EncodingConfiguration,
    reducer: ReducerName,
    scaler: ScalerName,
    range_width: float,
    ansatz_reps: int,
    initialization_seed: int,
    max_iterations: int,
    n_splits: int,
    random_state: int,
) -> list[tuple[float, float, float]]:
    features, labels = subsample
    feature_map, feature_map_reps = configuration
    steps = quantum_encoding_steps(
        "angle",
        scaler=scaler,
        reducer=reducer,
        angle_range=(0.0, range_width),
        feature_map=feature_map,
        feature_map_reps=feature_map_reps,
        random_state=random_state,
    )
    pipeline = build_weak_quantum_classifier_pipeline(steps, ansatz_reps, max_iterations, initialization_seed)
    # Os mesmos folds para todas as combinações: as diferenças entre elas não vêm do sorteio dos folds
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(pipeline, features, labels, cv=folds, scoring=("roc_auc", "accuracy"), return_train_score=True)
    return list(zip(scores["test_roc_auc"], scores["test_accuracy"], scores["train_accuracy"], strict=True))


def weak_classifier_encoding_sweep(
    subsamples: Mapping[int, KernelSubsample],
    configurations: Mapping[str, EncodingConfiguration],
    reducers: Sequence[ReducerName],
    scalers: Sequence[ScalerName],
    range_widths: Sequence[float],
    ansatz_reps_options: Sequence[int],
    initialization_seeds: Sequence[int],
    max_iterations: int,
    n_splits: int,
    random_state: int,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compara codificações de rotação num classificador variacional sozinho, por validação cruzada.

    Para cada combinação, o pipeline inteiro (scaler, redutor, faixa de ângulos, codificação e
    classificador) é reajustado em cada fold, dentro da subamostra balanceada. Mede duas coisas
    diferentes: a ROC-AUC (o classificador **ordena** bem as linhas?) e a acurácia com o corte fixo
    em P(1) = 0,5 (ele consegue **posicionar** o corte?). É a segunda que o AdaBoost usa: o erro
    ponderado de cada rodada é calculado sobre as classes previstas, não sobre a ordem.

    Args:
        subsamples: Subamostras balanceadas (features, rótulos), identificadas pela semente que as gerou.
        configurations: Nome de exibição → (feature map, repetições do feature map).
        reducers: Redutores de dimensionalidade a comparar.
        scalers: Scalers de entrada a comparar.
        range_widths: Larguras da faixa de ângulos a comparar; a faixa é [0, largura].
        ansatz_reps_options: Repetições do ansatz a comparar.
        initialization_seeds: Sementes dos pesos iniciais do classificador.
        max_iterations: Avaliações da função de custo por ajuste (COBYLA).
        n_splits: Número de folds.
        random_state: Semente dos folds e do redutor.
        n_jobs: Processos paralelos; o resultado não depende deste valor.

    Returns:
        Uma linha por combinação e fold, com as colunas de ``WEAK_CLASSIFIER_SWEEP_COLUMNS``.
    """
    tasks = [
        (name, reducer, scaler, range_width, ansatz_reps, subsample_seed, initialization_seed)
        for name in configurations
        for reducer in reducers
        for scaler in scalers
        for range_width in range_widths
        for ansatz_reps in ansatz_reps_options
        for subsample_seed in subsamples
        for initialization_seed in initialization_seeds
    ]
    fold_scores = Parallel(n_jobs=n_jobs)(
        delayed(_weak_classifier_fold_scores)(
            subsamples[subsample_seed], configurations[name], reducer, scaler, range_width, ansatz_reps,
            initialization_seed, max_iterations, n_splits, random_state,
        )
        for name, reducer, scaler, range_width, ansatz_reps, subsample_seed, initialization_seed in tasks
    )
    rows = [
        (*task, fold, *scores)
        for task, task_scores in zip(tasks, fold_scores, strict=True)
        for fold, scores in enumerate(task_scores)
    ]
    return pd.DataFrame(rows, columns=list(WEAK_CLASSIFIER_SWEEP_COLUMNS))
