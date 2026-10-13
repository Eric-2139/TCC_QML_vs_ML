"""Codificação: os circuitos que transformam uma linha de dados num estado quântico, e a simulação deles.

A codificação é uma etapa do pipeline (``StatevectorEncoder``), entre o pré-processamento e o modelo.
O estado codificado depende só dos dados e do feature map, nunca dos pesos do modelo: simulá-lo uma
vez por linha serve ao kernel do QSVM e a todos os classificadores fracos do Quantum AdaBoost.
"""

from typing import Final, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit_aer import AerSimulator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

AngleFeatureMapName = Literal["z", "zz", "ry"]
EncodingName = Literal["z", "zz", "ry", "amplitude"]
Entanglement = Literal["linear", "full"]

AMPLITUDE_ENCODING: Final[str] = "amplitude"
DEFAULT_FEATURE_MAP_REPS: Final[int] = 2

# O mesmo α = 2 dos mapas Z e ZZ do Qiskit: com ele, fase e rotação em Y têm o mesmo kernel e a mesma
# faixa útil de ângulos, e a comparação entre as duas não mistura codificação com largura de banda.
ROTATION_ANGLE_FACTOR: Final[float] = 2.0

# Linhas por execução do simulador: limita a memória do resultado (um objeto por linha) sem custo de tempo.
SIMULATION_BATCH_ROWS: Final[int] = 4_096

_UNIT_NORM_TOLERANCE: Final[float] = 1e-8


def _ry_feature_map(n_qubits: int) -> QuantumCircuit:
    inputs = ParameterVector("x", n_qubits)
    circuit = QuantumCircuit(n_qubits)
    for qubit in range(n_qubits):
        circuit.ry(ROTATION_ANGLE_FACTOR * inputs[qubit], qubit)
    return circuit


def build_angle_feature_map(
    name: AngleFeatureMapName,
    n_qubits: int,
    reps: int = DEFAULT_FEATURE_MAP_REPS,
    entanglement: Entanglement = "linear",
) -> QuantumCircuit:
    """Cria um feature map de rotação (uma feature por qubit).

    "Ângulo", nos nomes deste pacote, é a família das codificações em que a feature vira o ângulo
    de uma rotação, em oposição à amplitude. A rigor, os mapas ``z`` e ``zz`` são **codificação de
    fase**: Hadamard (leva o qubit ao equador da esfera de Bloch) seguida de porta de fase P(2x)
    (rotação em torno de Z). A codificação de ângulo propriamente dita é a rotação em Y (``ry``).

    Para um kernel, a diferença não existe com ``reps=1``: a fidelidade só depende do ângulo
    entre os dois estados, e percorrer o equador (fase) ou um meridiano (RY) dá o mesmo kernel,
    Π cos²(Δ). Com ``reps=2`` (o padrão) o kernel já é outro. Num modelo que **mede** o estado
    (classificador variacional), fase e RY não são equivalentes.

    - ``z``: rotações por qubit, **sem emaranhamento**. O estado é um produto de estados de
      1 qubit, então o kernel é o produto de kernels de 1 qubit: calculável classicamente.
    - ``zz``: acrescenta termos ZZ entre pares de qubits (emaranhamento), que codificam
      produtos de features (π − xᵢ)(π − xⱼ). É o mapa de Havlíček et al. (2019).
    - ``ry``: RY(2x) em cada qubit, sem Hadamard e sem emaranhamento. Tem uma camada só:
      repetir a mesma rotação apenas multiplicaria o ângulo.

    Todos usam α = 2 (o padrão do Qiskit nos mapas ``z`` e ``zz``): a rotação é por 2x, com período
    π em x. No mapa ``z`` a similaridade cai monotonicamente com a distância até π/2
    (``DEFAULT_ANGLE_RANGE``) e, em π, valores opostos viram o mesmo estado. No ``zz`` o termo de
    pares varia mais rápido e a similaridade deixa de ser monotônica antes disso (≈ 0,6 rad na
    configuração padrão): nele a largura da faixa é um hiperparâmetro a escolher por validação
    cruzada, não um limite teórico.

    Args:
        name: Tipo do mapa.
        n_qubits: Número de qubits (= número de features).
        reps: Quantas vezes o bloco de codificação se repete; ignorado no ``ry``.
        entanglement: Pares acoplados no ``zz`` (vizinhos ou todos); ignorado no ``z`` e no ``ry``.

    Returns:
        Circuito parametrizado com ``n_qubits`` parâmetros de dados.
    """
    if name == "ry":
        return _ry_feature_map(n_qubits)
    if name == "z":
        return z_feature_map(n_qubits, reps=reps)
    return zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)


def simulate_statevectors(feature_map: QuantumCircuit, parameter_rows: ArrayLike) -> NDArray[np.complex128]:
    """Simula o feature map para cada linha e devolve os estados finais (simulação exata, sem ruído).

    O circuito é compilado uma vez e o Aer recebe todas as linhas de um lote numa única execução.
    O resultado é o mesmo de simular um circuito por vez com o ``Statevector`` do Qiskit (há teste);
    só o tempo muda.

    Args:
        feature_map: Circuito com um parâmetro por coluna de ``parameter_rows``.
        parameter_rows: Valores dos parâmetros, uma linha por estado (n, n_parâmetros).

    Returns:
        Matriz (n, 2ⁿ_qubits) com as amplitudes de cada estado, na ordem de bits do Qiskit.

    Raises:
        ValueError: Se o número de colunas não for o número de parâmetros do circuito.
    """
    rows = np.asarray(parameter_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != feature_map.num_parameters:
        raise ValueError(f"esperadas {feature_map.num_parameters} colunas de parâmetros, recebido o formato {rows.shape}")
    simulator = AerSimulator(method="statevector")
    instrumented = feature_map.copy()
    instrumented.save_statevector()
    compiled = transpile(instrumented, simulator)
    states = np.empty((rows.shape[0], 2**feature_map.num_qubits), dtype=np.complex128)
    for start in range(0, rows.shape[0], SIMULATION_BATCH_ROWS):
        batch = rows[start : start + SIMULATION_BATCH_ROWS]
        binds = [{parameter: batch[:, column].tolist() for column, parameter in enumerate(feature_map.parameters)}]
        result = simulator.run(compiled, parameter_binds=binds).result()
        for offset in range(batch.shape[0]):
            states[start + offset] = np.asarray(result.data(offset)["statevector"])
    return states


def states_to_features(states: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Representa estados complexos como colunas reais: [partes reais | partes imaginárias].

    O scikit-learn não aceita matrizes complexas entre etapas de um pipeline; esta é a forma de
    entregar os estados ao modelo sem perder nada.
    """
    return np.hstack([states.real, states.imag])


def features_to_states(features: ArrayLike) -> NDArray[np.complex128]:
    """Inverso de ``states_to_features``.

    Raises:
        ValueError: Se o número de colunas não for 2 · 2ⁿ.
    """
    array = np.asarray(features, dtype=float)
    dimension = array.shape[1] // 2
    if array.shape[1] % 2 != 0 or dimension < 2 or dimension & (dimension - 1) != 0:
        raise ValueError(f"{array.shape[1]} colunas não representam estados de n qubits (esperado 2 · 2ⁿ)")
    return array[:, :dimension] + 1j * array[:, dimension:]


class StatevectorEncoder(TransformerMixin, BaseEstimator):
    """Etapa de pipeline que codifica cada linha num estado quântico (simulação exata).

    Recebe a saída do pré-processamento (ângulos, ou amplitudes com norma 1) e devolve o estado
    de cada linha no formato de ``states_to_features``. Não aprende nada com os dados: o ``fit``
    só registra o formato da entrada.

    Args:
        encoding: ``z``, ``zz`` ou ``ry`` (uma feature por qubit) ou ``amplitude`` (a linha já é o
            vetor de amplitudes, saída do ``AmplitudeEncodingPreparer``).
        feature_map_reps: Repetições do bloco de codificação nos mapas ``z`` e ``zz``.
        entanglement: Pares acoplados no ``zz``.
    """

    def __init__(
        self,
        encoding: EncodingName = "zz",
        feature_map_reps: int = DEFAULT_FEATURE_MAP_REPS,
        entanglement: Entanglement = "linear",
    ) -> None:
        self.encoding = encoding
        self.feature_map_reps = feature_map_reps
        self.entanglement = entanglement

    def fit(self, features: ArrayLike, labels: ArrayLike | None = None) -> Self:
        array = check_array(features)
        n_columns = array.shape[1]
        if self.encoding == AMPLITUDE_ENCODING:
            if n_columns < 2 or n_columns & (n_columns - 1) != 0:
                raise ValueError(f"a codificação de amplitude precisa de 2ⁿ colunas, recebidas {n_columns}")
            self.n_qubits_ = n_columns.bit_length() - 1
        else:
            self.n_qubits_ = n_columns
        self.n_features_in_ = n_columns
        return self

    def build_feature_map(self) -> QuantumCircuit:
        """Circuito de codificação desta etapa (só existe nas codificações de rotação).

        Raises:
            ValueError: Na codificação de amplitude, em que o estado é o próprio vetor e não há
                circuito parametrizado a simular.
        """
        check_is_fitted(self, "n_qubits_")
        if self.encoding == AMPLITUDE_ENCODING:
            raise ValueError("a codificação de amplitude não tem feature map parametrizado")
        return build_angle_feature_map(self.encoding, self.n_qubits_, self.feature_map_reps, self.entanglement)

    def transform(self, features: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "n_qubits_")
        array = check_array(features)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(f"esperadas {self.n_features_in_} features, recebidas {array.shape[1]}")
        if self.encoding == AMPLITUDE_ENCODING:
            if not np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=_UNIT_NORM_TOLERANCE):
                raise ValueError("as linhas precisam ter norma 1 (use o AmplitudeEncodingPreparer)")
            return states_to_features(array.astype(np.complex128))
        return states_to_features(simulate_statevectors(self.build_feature_map(), array))
