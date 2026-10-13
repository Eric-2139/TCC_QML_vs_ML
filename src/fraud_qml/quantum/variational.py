"""Classificador variacional: o classificador fraco do Quantum AdaBoost.

Um circuito com pesos ajustáveis (ansatz) é aplicado ao estado codificado de cada linha, e mede-se
um qubit. A probabilidade de medir 1 é lida como P(classe positiva); o treino ajusta os pesos para
reduzir a entropia cruzada ponderada pelos pesos das amostras, que é como o boosting diz ao
classificador em quais linhas errar custa mais (Wang et al., 2024, Eq. 7 e Eq. 9).
"""

from typing import Final, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import Operator
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from fraud_qml.quantum.encodings import features_to_states

DEFAULT_ANSATZ_REPS: Final[int] = 1
DEFAULT_MAX_ITERATIONS: Final[int] = 60

# Qubit medido. No Qiskit o qubit 0 é o bit menos significativo do índice do estado: ele vale 1
# exatamente nos índices ímpares.
READOUT_QUBIT: Final[int] = 0

# log(0) não existe: a probabilidade é mantida a esta distância de 0 e de 1 na entropia cruzada
_PROBABILITY_FLOOR: Final[float] = 1e-12

_N_CLASSES: Final[int] = 2

# Avaliações que o COBYLA exige além de uma por peso (o ponto inicial e o primeiro passo)
_COBYLA_EXTRA_EVALUATIONS: Final[int] = 2


def build_ansatz(n_qubits: int, reps: int = DEFAULT_ANSATZ_REPS) -> QuantumCircuit:
    """Ansatz ``RealAmplitudes``: camadas de rotações RY intercaladas com portas CNOT.

    Tem ``n_qubits · (reps + 1)`` pesos. Só usa rotações em Y e CNOT, então a matriz do circuito é
    real: ele não mistura as partes real e imaginária do estado codificado (é por isso que a
    codificação de fase e a de rotação em Y podem se comportar de forma diferente aqui, CC14).

    Args:
        n_qubits: Número de qubits.
        reps: Número de blocos (CNOTs + camada de rotações) depois da primeira camada de rotações.

    Returns:
        Circuito parametrizado pelos pesos.
    """
    return real_amplitudes(n_qubits, reps=reps)


def ansatz_unitary(ansatz: QuantumCircuit, weights: ArrayLike) -> NDArray[np.complex128]:
    """Matriz 2ⁿ × 2ⁿ do ansatz para um vetor de pesos (calculada pelo Qiskit)."""
    return Operator(ansatz.assign_parameters(np.asarray(weights, dtype=float))).data


def readout_one_probability(states: NDArray[np.complex128], unitary: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Probabilidade de medir 1 no qubit de leitura depois de aplicar o ansatz a cada estado.

    Os estados codificados não dependem dos pesos, então aplicar o ansatz a todas as linhas é um
    único produto matricial. O valor é o mesmo que o simulador devolve para o circuito completo
    (feature map + ansatz), com ⟨Z⟩ = 1 − 2·P(1); há teste contra o ``EstimatorV2`` do Aer.

    Args:
        states: Estados codificados (n, 2ⁿ_qubits).
        unitary: Matriz do ansatz.

    Returns:
        P(1) no qubit de leitura, para cada linha.
    """
    evolved = states @ unitary.T
    readout_is_one = (np.arange(states.shape[1]) >> READOUT_QUBIT) & 1 == 1
    return np.sum(np.abs(evolved[:, readout_is_one]) ** 2, axis=1)


def _validated_sample_weight(sample_weight: ArrayLike | None, n_rows: int) -> NDArray[np.float64]:
    if sample_weight is None:
        return np.ones(n_rows)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (n_rows,):
        raise ValueError(f"sample_weight tem o formato {weights.shape}, esperado ({n_rows},)")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("sample_weight precisa ser não negativo e ter soma positiva")
    return weights


def weighted_cross_entropy(
    positive_probability: NDArray[np.float64], is_positive: NDArray[np.bool_], sample_weight: NDArray[np.float64]
) -> float:
    """Entropia cruzada média, ponderada pelos pesos das amostras.

    Cada linha contribui com −log da probabilidade que o modelo deu à classe **certa**, multiplicada
    pelo peso dela. Uma linha de peso zero não contribui: é como se não existisse.

    Args:
        positive_probability: P(classe positiva) prevista para cada linha.
        is_positive: Se cada linha é da classe positiva.
        sample_weight: Peso de cada linha (não negativo, soma positiva).

    Returns:
        Entropia cruzada ponderada, dividida pela soma dos pesos.
    """
    clipped = np.clip(positive_probability, _PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)
    row_losses = -np.where(is_positive, np.log(clipped), np.log1p(-clipped))
    return float(np.sum(sample_weight * row_losses) / np.sum(sample_weight))


class VariationalQuantumClassifier(ClassifierMixin, BaseEstimator):
    """Classificador binário: ansatz ``RealAmplitudes`` + medição de um qubit, treinado com COBYLA.

    Recebe os estados já codificados (saída do ``StatevectorEncoder``). Foi feito para ser o
    classificador **fraco** do ``AdaBoostClassifier``: aceita ``sample_weight`` e tem um orçamento
    curto de otimização (``max_iterations``).

    O COBYLA é um otimizador sem gradiente: a cada passo avalia a função de custo num ponto novo e
    ajusta uma aproximação linear dela. Cada avaliação equivale a uma execução do circuito para
    todas as linhas de treino.

    Args:
        ansatz_reps: Repetições do bloco do ansatz (profundidade do circuito treinável).
        max_iterations: Número máximo de avaliações da função de custo pelo COBYLA.
        random_state: Semente dos pesos iniciais, sorteados em [−π, π].
    """

    def __init__(
        self,
        ansatz_reps: int = DEFAULT_ANSATZ_REPS,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.ansatz_reps = ansatz_reps
        self.max_iterations = max_iterations
        self.random_state = random_state

    def fit(self, features: ArrayLike, labels: ArrayLike, sample_weight: ArrayLike | None = None) -> Self:
        """Ajusta os pesos do ansatz minimizando a entropia cruzada ponderada.

        Args:
            features: Estados de treino, no formato de ``states_to_features``.
            labels: Rótulos binários.
            sample_weight: Peso de cada linha; se ausente, todas pesam o mesmo.

        Returns:
            O próprio estimador.

        Raises:
            ValueError: Se os rótulos não tiverem exatamente duas classes, os pesos forem inválidos ou
                ``max_iterations`` for menor que o mínimo do COBYLA (número de pesos + 2).
        """
        array, targets = check_X_y(features, labels)
        self.classes_ = unique_labels(targets)
        if len(self.classes_) != _N_CLASSES:
            raise ValueError(f"o classificador é binário, mas os rótulos têm {len(self.classes_)} classes")
        self.n_features_in_ = array.shape[1]
        weights = _validated_sample_weight(sample_weight, array.shape[0])
        states = features_to_states(array)
        is_positive = targets == self.classes_[1]
        self.ansatz_ = build_ansatz(int(np.log2(states.shape[1])), self.ansatz_reps)
        # O COBYLA gasta uma avaliação no ponto inicial e uma por peso antes do primeiro passo; com menos
        # que isso ele aumentaria o orçamento por conta própria, e o custo deixaria de ser o declarado
        minimum_iterations = self.ansatz_.num_parameters + _COBYLA_EXTRA_EVALUATIONS
        if self.max_iterations < minimum_iterations:
            raise ValueError(
                f"max_iterations = {self.max_iterations} é menos que o mínimo do COBYLA para "
                f"{self.ansatz_.num_parameters} pesos ({minimum_iterations})"
            )

        loss_curve: list[float] = []

        def loss(ansatz_weights: NDArray[np.float64]) -> float:
            probability = readout_one_probability(states, ansatz_unitary(self.ansatz_, ansatz_weights))
            value = weighted_cross_entropy(probability, is_positive, weights)
            loss_curve.append(value)
            return value

        initial_weights = check_random_state(self.random_state).uniform(-np.pi, np.pi, self.ansatz_.num_parameters)
        result = minimize(loss, initial_weights, method="COBYLA", options={"maxiter": self.max_iterations})
        self.weights_ = np.asarray(result.x, dtype=float)
        self.unitary_ = ansatz_unitary(self.ansatz_, self.weights_)
        self.loss_curve_ = np.asarray(loss_curve)
        return self

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Probabilidades das duas classes: P(medir 0) e P(medir 1) no qubit de leitura."""
        check_is_fitted(self, "unitary_")
        positive = readout_one_probability(features_to_states(check_array(features)), self.unitary_)
        return np.column_stack([1.0 - positive, positive])

    def decision_function(self, features: ArrayLike) -> NDArray[np.float64]:
        """P(1) − P(0) = −⟨Z⟩ no qubit de leitura, em [−1, 1] (positivo = classe ``classes_[1]``)."""
        probabilities = self.predict_proba(features)
        return probabilities[:, 1] - probabilities[:, 0]

    def predict(self, features: ArrayLike) -> NDArray[np.int_]:
        """Classe prevista: a positiva quando P(1) > 0,5."""
        return self.classes_[(self.decision_function(features) > 0).astype(int)]
