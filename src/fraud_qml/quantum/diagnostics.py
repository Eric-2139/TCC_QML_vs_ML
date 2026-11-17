"""Varreduras de diagnóstico: concentração e alinhamento dos kernels (KTA) e codificação no classificador fraco.

Existem como funções do pacote, e não como laços no notebook, para que o procedimento que
delimita as opções do grid (Seções 4 e 5 do notebook, só no treino) seja testado e rode em paralelo.
"""

from collections.abc import Mapping, Sequence
from typing import Final, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from sklearn.model_selection import StratifiedKFold, cross_validate

from fraud_qml.data import FRAUD_LABEL
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
from fraud_qml.quantum.variational import (
    ansatz_unitary,
    build_ansatz,
    readout_one_probability,
    weighted_cross_entropy,
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


# ---------------------------------------------------------------- barren plateau (Seção 8)

ReadoutObservable = Literal["local", "global"]
READOUT_OBSERVABLES: Final[tuple[ReadoutObservable, ...]] = ("local", "global")

# Regra do parameter-shift para uma rotação RY(θ) = exp(−iθY/2): a derivada exata de um valor esperado é a metade da
# diferença entre os valores com o peso deslocado de +π/2 e de −π/2
PARAMETER_SHIFT: Final[float] = np.pi / 2

# A mesma distância de 0 e de 1 que a entropia cruzada do classificador usa, para a derivada não explodir
_PROBABILITY_FLOOR: Final[float] = 1e-12

LANDSCAPE_COLUMNS: Final[tuple[str, ...]] = (
    "states",
    "n_qubits",
    "ansatz_reps",
    "observable",
    "sample",
    "cost",
    "gradient",
)


def parity_one_probability(states: NDArray[np.complex128], unitary: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Probabilidade de a paridade de **todos** os qubits ser ímpar depois do ansatz: (1 − ⟨Z⊗…⊗Z⟩) / 2.

    É a leitura **global**, o contrário da do classificador, que mede um qubit só (``readout_one_probability``). Serve de
    controle na medida de *barren plateau*: custos que dependem de todos os qubits são os que a teoria mais associa ao
    platô.

    Args:
        states: Estados codificados (n, 2ⁿ_qubits).
        unitary: Matriz do ansatz.

    Returns:
        P(paridade ímpar) para cada linha.
    """
    evolved = states @ unitary.T
    odd_parity = np.array([bin(index).count("1") % 2 == 1 for index in range(states.shape[1])])
    return np.sum(np.abs(evolved[:, odd_parity]) ** 2, axis=1)


def _one_probability(states: NDArray[np.complex128], unitary: NDArray[np.complex128], observable: ReadoutObservable) -> NDArray[np.float64]:
    if observable == "local":
        return readout_one_probability(states, unitary)
    return parity_one_probability(states, unitary)


def cost_and_gradient(
    states: NDArray[np.complex128],
    is_positive: NDArray[np.bool_],
    ansatz: QuantumCircuit,
    weights: NDArray[np.float64],
    parameter_index: int,
    observable: ReadoutObservable,
) -> tuple[float, float]:
    """Entropia cruzada (pesos iguais) e a derivada exata dela em relação a um peso do ansatz.

    A probabilidade de cada linha é linear num valor esperado, então a derivada dela sai exata do *parameter-shift*;
    a da entropia cruzada vem da regra da cadeia: dC/dθ = Σ (dC/dpᵢ)·(dpᵢ/dθ).

    Args:
        states: Estados codificados (n, 2ⁿ_qubits).
        is_positive: Se cada linha é da classe positiva.
        ansatz: Circuito ``RealAmplitudes`` (só rotações RY, cada peso numa porta).
        weights: Pesos do ansatz.
        parameter_index: Posição do peso derivado.
        observable: ``local`` (o qubit 0, como o classificador) ou ``global`` (a paridade de todos).

    Returns:
        O custo e a derivada.
    """
    probability = _one_probability(states, ansatz_unitary(ansatz, weights), observable)
    shifted = np.array(weights, dtype=float)
    shifted[parameter_index] += PARAMETER_SHIFT
    probability_plus = _one_probability(states, ansatz_unitary(ansatz, shifted), observable)
    shifted[parameter_index] -= 2 * PARAMETER_SHIFT
    probability_minus = _one_probability(states, ansatz_unitary(ansatz, shifted), observable)
    probability_slope = (probability_plus - probability_minus) / 2
    clipped = np.clip(probability, _PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)
    cost_slope = np.where(is_positive, -1.0 / clipped, 1.0 / (1.0 - clipped)) / len(probability)
    cost = weighted_cross_entropy(probability, is_positive, np.ones(len(probability)))
    return cost, float(np.sum(cost_slope * probability_slope))


def _landscape_samples(
    name: str,
    states: NDArray[np.complex128],
    labels: NDArray[np.int_],
    ansatz_reps: int,
    observables: Sequence[ReadoutObservable],
    n_samples: int,
    parameter_index: int,
    random_state: int,
) -> list[tuple[object, ...]]:
    n_qubits = int(np.log2(states.shape[1]))
    ansatz = build_ansatz(n_qubits, ansatz_reps)
    is_positive = labels == FRAUD_LABEL
    # Os mesmos pesos sorteados para os dois observáveis: a diferença entre eles não vem do sorteio
    rng = np.random.default_rng(random_state)
    rows: list[tuple[object, ...]] = []
    for sample in range(n_samples):
        # A mesma distribuição dos pesos iniciais do classificador variacional
        weights = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
        for observable in observables:
            cost, gradient = cost_and_gradient(states, is_positive, ansatz, weights, parameter_index, observable)
            rows.append((name, n_qubits, ansatz_reps, observable, sample, cost, gradient))
    return rows


def barren_plateau_sweep(
    state_sets: Mapping[str, tuple[NDArray[np.complex128], NDArray[np.int_]]],
    ansatz_reps_options: Sequence[int],
    observables: Sequence[ReadoutObservable],
    n_samples: int,
    random_state: int,
    parameter_index: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Custo e derivada em pesos sorteados, para cada conjunto de estados, profundidade do ansatz e observável.

    A variância da derivada (e do custo) entre os sorteios é a medida usual de *barren plateau*: se ela cai
    exponencialmente com o número de qubits, a superfície do custo fica plana em quase todo lugar, e o otimizador
    não tem para onde descer (com ou sem gradiente: o COBYLA compara valores do custo, que também deixam de variar).

    Args:
        state_sets: Nome → (estados codificados, rótulos); o número de qubits sai do tamanho dos estados.
        ansatz_reps_options: Profundidades do ansatz.
        observables: ``local`` e/ou ``global``.
        n_samples: Pesos sorteados por combinação, uniformes em [−π, π] (como os pesos iniciais do classificador).
        random_state: Semente dos sorteios (a mesma em cada combinação).
        parameter_index: Posição do peso derivado.
        n_jobs: Processos paralelos; o resultado não depende deste valor.

    Returns:
        Uma linha por combinação, sorteio e observável, com ``LANDSCAPE_COLUMNS``.
    """
    tasks = [(name, reps) for name in state_sets for reps in ansatz_reps_options]
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_landscape_samples)(name, *state_sets[name], reps, observables, n_samples, parameter_index, random_state)
        for name, reps in tasks
    )
    return pd.DataFrame([row for task_rows in outputs for row in task_rows], columns=list(LANDSCAPE_COLUMNS))


def readout_bloch_length(states: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Comprimento do vetor de Bloch do qubit de leitura (o qubit 0) em cada estado, antes do ansatz.

    O estado reduzido de um qubit é descrito por um vetor na esfera de Bloch: comprimento 1 quando o qubit tem estado
    próprio (não está emaranhado com os outros), e perto de 0 quando ele está muito emaranhado ("misturado"). Um qubit de
    leitura misturado dá P(1) perto de 0,5 para quase qualquer ansatz raso, o que achata o custo.

    Args:
        states: Estados (n, 2ⁿ_qubits); o qubit 0 é o bit menos significativo do índice (convenção do Qiskit).

    Returns:
        O comprimento, entre 0 e 1, para cada linha.
    """
    # Separa o índice do qubit 0 (última posição) dos outros qubits e traça os outros
    amplitudes = states.reshape(states.shape[0], -1, 2)
    reduced = np.einsum("nri,nrj->nij", amplitudes, amplitudes.conj())
    x = 2 * reduced[:, 0, 1].real
    y = -2 * reduced[:, 0, 1].imag
    z = (reduced[:, 0, 0] - reduced[:, 1, 1]).real
    return np.sqrt(x**2 + y**2 + z**2)
