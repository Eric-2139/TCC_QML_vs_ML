"""Kernels quânticos de fidelidade e diagnósticos de qualidade de kernel.

O kernel de fidelidade mede a semelhança entre duas linhas como |⟨ψ(x)|ψ(x')⟩|²: 1 para
estados idênticos, 0 para estados ortogonais. É a matriz que o SVM usa no lugar do RBF.
"""

from typing import Final, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from fraud_qml.quantum.encodings import features_to_states

_UNIT_NORM_TOLERANCE = 1e-8

# Linhas por lote na predição do QSVM: limita a matriz de kernel em memória a (lote × treino)
PREDICTION_BATCH_ROWS: Final[int] = 4_096


def angle_kernel_matrix(
    feature_map: QuantumCircuit, features: ArrayLike, other_features: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Kernel de fidelidade exato (simulação por statevector) para um feature map de ângulo.

    Args:
        feature_map: Circuito parametrizado com uma entrada por feature.
        features: Linhas já no intervalo de ângulos (n, n_qubits).
        other_features: Segundo conjunto; se ausente, calcula a matriz de Gram de ``features``.

    Returns:
        Matriz (n, m) de similaridades em [0, 1].
    """
    kernel = FidelityStatevectorKernel(feature_map=feature_map)
    if other_features is None:
        return kernel.evaluate(x_vec=np.asarray(features))
    return kernel.evaluate(x_vec=np.asarray(features), y_vec=np.asarray(other_features))


def amplitude_kernel_matrix(
    amplitudes: ArrayLike, other_amplitudes: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Kernel de fidelidade da codificação de amplitude, calculado diretamente.

    Na codificação de amplitude o estado quântico **é** o próprio vetor normalizado, então a
    fidelidade é |⟨x|x'⟩|² = cos²(ângulo entre x e x'). Não há circuito a simular: o resultado
    é idêntico ao do Qiskit com ``raw_feature_vector`` (verificado nos testes) e milhares de
    vezes mais rápido. É também a prova de que este kernel é calculável classicamente.

    Args:
        amplitudes: Linhas com norma 1 (saída do ``AmplitudeEncodingPreparer``).
        other_amplitudes: Segundo conjunto; se ausente, usa ``amplitudes``.

    Returns:
        Matriz (n, m) de similaridades em [0, 1].

    Raises:
        ValueError: Se alguma linha não tiver norma 1.
    """
    left = np.asarray(amplitudes, dtype=float)
    right = left if other_amplitudes is None else np.asarray(other_amplitudes, dtype=float)
    for matrix in (left, right):
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=_UNIT_NORM_TOLERANCE):
            raise ValueError("as linhas precisam ter norma 1 (use o AmplitudeEncodingPreparer)")
    return np.abs(left @ right.T) ** 2


def fidelity_kernel_matrix(
    states: NDArray[np.complex128], other_states: NDArray[np.complex128] | None = None
) -> NDArray[np.float64]:
    """Kernel de fidelidade |⟨ψ(x)|ψ(x')⟩|² entre estados já simulados.

    Com um estado por linha, a matriz inteira sai de um produto matricial: o custo da simulação é
    O(n) (um circuito por linha), e não O(n²) (um circuito por par).

    Args:
        states: Estados (n, 2ⁿ_qubits), um por linha.
        other_states: Segundo conjunto (m, 2ⁿ_qubits); se ausente, calcula a matriz de Gram de ``states``.

    Returns:
        Matriz (n, m) de similaridades em [0, 1].
    """
    right = states if other_states is None else other_states
    return np.abs(states @ right.conj().T) ** 2


class QuantumKernelSVC(ClassifierMixin, BaseEstimator):
    """SVM com kernel quântico de fidelidade (QSVM).

    É o mesmo SVM do scikit-learn; a única diferença para o SVM clássico é a matriz de
    similaridades: em vez do RBF, a fidelidade entre os estados quânticos de cada par de linhas.
    Recebe os estados já codificados (saída do ``StatevectorEncoder``), então a codificação é
    escolhida numa etapa do pipeline, e não aqui.

    Args:
        C: Penalidade por erro de classificação no treino, como no ``SVC``.
    """

    def __init__(self, C: float = 1.0) -> None:  # noqa: N803  (mesmo nome do parâmetro no scikit-learn)
        self.C = C

    def fit(self, features: ArrayLike, labels: ArrayLike) -> Self:
        """Ajusta o SVM sobre a matriz de Gram dos estados de treino.

        Args:
            features: Estados de treino, no formato de ``states_to_features``.
            labels: Rótulos.

        Returns:
            O próprio estimador.
        """
        array, targets = check_X_y(features, labels)
        self.classes_ = unique_labels(targets)
        self.n_features_in_ = array.shape[1]
        # Os estados de treino ficam guardados: prever é medir a similaridade de cada linha nova com eles
        self.train_states_ = features_to_states(array)
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(fidelity_kernel_matrix(self.train_states_), targets)
        return self

    def decision_function(self, features: ArrayLike) -> NDArray[np.float64]:
        """Distância com sinal à fronteira de decisão (positiva = classe ``classes_[1]``)."""
        check_is_fitted(self, "svc_")
        states = features_to_states(check_array(features))
        decisions = np.empty(states.shape[0])
        # Em lotes: a matriz (linhas novas × treino) inteira ocuparia centenas de MiB na validação, um custo
        # de memória que seria só desta implementação, e não do modelo
        for start in range(0, states.shape[0], PREDICTION_BATCH_ROWS):
            batch = slice(start, start + PREDICTION_BATCH_ROWS)
            decisions[batch] = self.svc_.decision_function(fidelity_kernel_matrix(states[batch], self.train_states_))
        return decisions

    def predict(self, features: ArrayLike) -> NDArray[np.int_]:
        """Classe prevista para cada linha."""
        return self.classes_[(self.decision_function(features) > 0).astype(int)]


def off_diagonal_values(kernel_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Similaridades entre pares de linhas **diferentes** (a diagonal é sempre 1 e não informa nada)."""
    mask = ~np.eye(kernel_matrix.shape[0], dtype=bool)
    return kernel_matrix[mask]


def centered_kernel_target_alignment(kernel_matrix: NDArray[np.float64], labels: ArrayLike) -> float:
    """Alinhamento centrado entre o kernel e o rótulo (Cortes, Mohri e Rostamizadeh, 2012).

    Mede o quanto a similaridade do kernel concorda com o kernel "ideal" yyᵀ, que vale +1 para
    pares da mesma classe e −1 para pares de classes diferentes. Vai de −1 a 1: perto de 0, o
    kernel ignora o rótulo; quanto maior, mais "mesma classe" coincide com "parecidos".

    A centralização remove o efeito de uma similaridade de fundo igual para todos os pares: um
    kernel com todos os valores altos (concentrado em 1) não parece bom só por ser alto.

    Args:
        kernel_matrix: Matriz de Gram quadrada.
        labels: Rótulos binários (0/1) das mesmas linhas.

    Returns:
        Alinhamento centrado em [−1, 1].
    """
    signed_labels = np.where(np.asarray(labels) == 1, 1.0, -1.0)
    n_rows = kernel_matrix.shape[0]
    centering = np.eye(n_rows) - np.full((n_rows, n_rows), 1.0 / n_rows)
    centered_kernel = centering @ kernel_matrix @ centering
    centered_target = centering @ np.outer(signed_labels, signed_labels) @ centering
    numerator = np.sum(centered_kernel * centered_target)
    denominator = np.linalg.norm(centered_kernel) * np.linalg.norm(centered_target)
    return float(numerator / denominator)
