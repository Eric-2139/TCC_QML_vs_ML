"""Feature maps e kernels quânticos."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.circuit.library import raw_feature_vector
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from fraud_qml.preprocessing import DEFAULT_ANGLE_RANGE, AmplitudeEncodingPreparer
from fraud_qml.quantum.encodings import (
    StatevectorEncoder,
    build_angle_feature_map,
    features_to_states,
    simulate_statevectors,
)
from fraud_qml.quantum.kernels import (
    QuantumKernelSVC,
    amplitude_kernel_matrix,
    angle_kernel_matrix,
    centered_kernel_target_alignment,
    fidelity_kernel_matrix,
    off_diagonal_values,
)

RNG_SEED = 0


@pytest.fixture
def angles() -> np.ndarray:
    return np.random.default_rng(RNG_SEED).uniform(0, np.pi, size=(12, 3))


@pytest.mark.parametrize("map_name", ["z", "zz"])
def test_angle_kernel_is_a_valid_kernel(angles: np.ndarray, map_name: str) -> None:
    kernel = angle_kernel_matrix(build_angle_feature_map(map_name, n_qubits=3), angles)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-12)
    # Semidefinido positivo: exigência do SVM (autovalores ≥ 0, a menos de arredondamento)
    assert np.linalg.eigvalsh(kernel).min() > -1e-9
    assert kernel.min() >= -1e-12 and kernel.max() <= 1 + 1e-12


def test_z_kernel_is_product_of_single_qubit_kernels(angles: np.ndarray) -> None:
    # Sem emaranhamento, o kernel se fatora por qubit: é calculável classicamente
    full_kernel = angle_kernel_matrix(build_angle_feature_map("z", n_qubits=3), angles)
    single_qubit_map = build_angle_feature_map("z", n_qubits=1)
    product = np.ones_like(full_kernel)
    for column in range(angles.shape[1]):
        product *= angle_kernel_matrix(single_qubit_map, angles[:, [column]])
    np.testing.assert_allclose(full_kernel, product, atol=1e-10)


def test_zz_kernel_does_not_factorize(angles: np.ndarray) -> None:
    # Os termos de 1 qubit do ZZ são os mesmos do Z; se o kernel ZZ fosse o produto deles, o
    # emaranhamento não acrescentaria nada. (Um ZZ de 1 qubit não existe: não há par para acoplar.)
    full_kernel = angle_kernel_matrix(build_angle_feature_map("zz", n_qubits=3), angles)
    single_qubit_map = build_angle_feature_map("z", n_qubits=1)
    product = np.ones_like(full_kernel)
    for column in range(angles.shape[1]):
        product *= angle_kernel_matrix(single_qubit_map, angles[:, [column]])
    assert not np.allclose(full_kernel, product, atol=1e-3)


def _single_layer_map(gate: str, n_qubits: int) -> QuantumCircuit:
    """Codificação de fase (H e P(2x)) ou de ângulo (RY(2x)), uma camada, sem emaranhamento."""
    inputs = ParameterVector("x", n_qubits)
    circuit = QuantumCircuit(n_qubits)
    for qubit in range(n_qubits):
        if gate == "phase":
            circuit.h(qubit)
            circuit.p(2 * inputs[qubit], qubit)
        else:
            circuit.ry(2 * inputs[qubit], qubit)
    return circuit


def test_z_map_with_one_repetition_is_phase_encoding(angles: np.ndarray) -> None:
    # O mapa Z do Qiskit é Hadamard + porta de fase: "codificação de fase", não rotação em Y
    z_kernel = angle_kernel_matrix(build_angle_feature_map("z", n_qubits=3, reps=1), angles)
    np.testing.assert_allclose(z_kernel, angle_kernel_matrix(_single_layer_map("phase", 3), angles), atol=1e-12)


def test_phase_and_angle_encoding_give_the_same_kernel(angles: np.ndarray) -> None:
    # A fidelidade só depende do ângulo entre os estados: equador (fase) ou meridiano (RY) dão
    # o mesmo kernel, Π cos²(Δ). Para o QSVM, trocar uma codificação pela outra não muda nada
    phase_kernel = angle_kernel_matrix(_single_layer_map("phase", 3), angles)
    angle_kernel = angle_kernel_matrix(_single_layer_map("angle", 3), angles)
    differences = angles[:, None, :] - angles[None, :, :]
    np.testing.assert_allclose(phase_kernel, angle_kernel, atol=1e-12)
    np.testing.assert_allclose(phase_kernel, np.prod(np.cos(differences) ** 2, axis=2), atol=1e-12)


def test_second_repetition_of_the_z_map_changes_the_kernel(angles: np.ndarray) -> None:
    # Com reps=2 o mapa Z deixa de ser a codificação de fase pura: é por isso que reps entra no grid
    one_repetition = angle_kernel_matrix(build_angle_feature_map("z", n_qubits=3, reps=1), angles)
    two_repetitions = angle_kernel_matrix(build_angle_feature_map("z", n_qubits=3, reps=2), angles)
    assert np.abs(one_repetition - two_repetitions).max() > 0.05


def test_qiskit_z_map_has_period_pi() -> None:
    # alpha = 2: a rotação é P(2x). Valores a π de distância viram o mesmo estado, e é por isso
    # que a faixa de ângulos é limitada a π/2 (DEFAULT_ANGLE_RANGE)
    z_map = build_angle_feature_map("z", n_qubits=1)
    assert angle_kernel_matrix(z_map, [[0.0]], [[np.pi]])[0, 0] == pytest.approx(1.0)
    assert angle_kernel_matrix(z_map, [[0.0]], [[np.pi / 2]])[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_z_map_similarity_decreases_monotonically_up_to_half_pi() -> None:
    z_map = build_angle_feature_map("z", n_qubits=1)
    distances = np.linspace(0, np.pi / 2, 20).reshape(-1, 1)
    similarity = angle_kernel_matrix(z_map, [[0.0]], distances)[0]
    assert np.all(np.diff(similarity) <= 1e-12)


def test_zz_map_is_not_monotonic_within_half_pi() -> None:
    # O termo de pares (π − xᵢ)(π − xⱼ) varia mais rápido que os termos de 1 qubit: no ZZ, a
    # similaridade cai e volta a subir antes de π/2 (achado 5 da auditoria da Fase 2b)
    zz_map = build_angle_feature_map("zz", n_qubits=5)
    base = np.full((1, 5), 0.5)
    shifted = np.repeat(base, 20, axis=0)
    shifted[:, -1] = 0.5 + np.linspace(0, np.pi / 2, 20)
    similarity = angle_kernel_matrix(zz_map, base, shifted)[0]
    assert np.any(np.diff(similarity) > 0.01)


def test_default_angle_range_stays_within_monotonic_limit() -> None:
    low, high = DEFAULT_ANGLE_RANGE
    assert high - low <= np.pi / 2


def test_amplitude_kernel_matches_qiskit_simulation() -> None:
    rng = np.random.default_rng(RNG_SEED)
    amplitudes = AmplitudeEncodingPreparer(n_qubits=3).fit_transform(rng.normal(size=(6, 7)))
    simulated = FidelityStatevectorKernel(feature_map=raw_feature_vector(8)).evaluate(x_vec=amplitudes)
    np.testing.assert_allclose(amplitude_kernel_matrix(amplitudes), simulated, atol=1e-10)


def test_amplitude_kernel_rejects_unnormalized_rows() -> None:
    with pytest.raises(ValueError, match="norma 1"):
        amplitude_kernel_matrix(np.array([[1.0, 1.0]]))


@pytest.mark.parametrize(("encoding", "reps"), [("z", 1), ("z", 2), ("zz", 1), ("zz", 2), ("ry", 1)])
def test_fidelity_of_batch_states_matches_the_qiskit_kernel(angles: np.ndarray, encoding: str, reps: int) -> None:
    # O produto matricial sobre os estados simulados em lote é o kernel do qiskit-machine-learning
    feature_map = build_angle_feature_map(encoding, n_qubits=3, reps=reps)
    states = simulate_statevectors(feature_map, angles)
    kernel = fidelity_kernel_matrix(states)
    np.testing.assert_allclose(kernel, angle_kernel_matrix(feature_map, angles), atol=1e-10)
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(kernel).min() > -1e-9
    assert kernel.min() >= -1e-12 and kernel.max() <= 1 + 1e-12


def test_fidelity_of_amplitude_states_is_the_classical_cosine_squared() -> None:
    rng = np.random.default_rng(RNG_SEED)
    amplitudes = AmplitudeEncodingPreparer(n_qubits=3).fit_transform(rng.normal(size=(6, 7)))
    states = features_to_states(StatevectorEncoder(encoding="amplitude").fit_transform(amplitudes))
    np.testing.assert_allclose(fidelity_kernel_matrix(states), amplitude_kernel_matrix(amplitudes), atol=1e-12)
    np.testing.assert_allclose(
        fidelity_kernel_matrix(states[:2], states), amplitude_kernel_matrix(amplitudes[:2], amplitudes), atol=1e-12
    )


def _labeled_angles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    labels = np.array([0, 1] * 20)
    # As duas classes ocupam metades diferentes da faixa de ângulos: problema fácil, mas não trivial
    train = rng.uniform(0, np.pi / 4, size=(40, 3)) + labels[:, None] * np.pi / 4
    return train, labels, rng.uniform(0, np.pi / 2, size=(15, 3))


def test_qsvm_matches_svc_with_the_qiskit_kernel_as_callable() -> None:
    # O desenho original do plano era SVC(kernel=callable); o estimador precisa dar as mesmas decisões
    train, labels, new_rows = _labeled_angles()
    feature_map = build_angle_feature_map("zz", n_qubits=3)
    reference = SVC(kernel=lambda left, right: angle_kernel_matrix(feature_map, left, right), C=2.0).fit(train, labels)
    encoder = StatevectorEncoder(encoding="zz").fit(train)
    qsvm = QuantumKernelSVC(C=2.0).fit(encoder.transform(train), labels)
    np.testing.assert_allclose(
        qsvm.decision_function(encoder.transform(new_rows)), reference.decision_function(new_rows), atol=1e-8
    )
    np.testing.assert_array_equal(qsvm.predict(encoder.transform(new_rows)), reference.predict(new_rows))


def test_qsvm_prediction_does_not_depend_on_the_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    train, labels, new_rows = _labeled_angles()
    encoder = StatevectorEncoder(encoding="zz").fit(train)
    qsvm = QuantumKernelSVC().fit(encoder.transform(train), labels)
    encoded_rows = encoder.transform(new_rows)
    in_one_batch = qsvm.decision_function(encoded_rows)
    monkeypatch.setattr("fraud_qml.quantum.kernels.PREDICTION_BATCH_ROWS", 4)
    np.testing.assert_allclose(qsvm.decision_function(encoded_rows), in_one_batch, atol=1e-12)


def test_qsvm_separates_an_easy_problem_and_keeps_the_original_labels() -> None:
    train, labels, _ = _labeled_angles()
    named_labels = np.where(labels == 1, 7, 3)
    encoded = StatevectorEncoder(encoding="z", feature_map_reps=1).fit_transform(train)
    qsvm = QuantumKernelSVC().fit(encoded, named_labels)
    assert qsvm.classes_.tolist() == [3, 7]
    assert (qsvm.predict(encoded) == named_labels).mean() > 0.9


def test_qsvm_works_inside_a_grid_search() -> None:
    train, labels, _ = _labeled_angles()
    pipeline = Pipeline([("encoder", StatevectorEncoder()), ("model", QuantumKernelSVC())])
    search = GridSearchCV(pipeline, {"encoder__encoding": ["z", "zz"], "model__C": [0.1, 1.0]}, cv=2, scoring="roc_auc")
    search.fit(train, labels)
    assert set(search.best_params_) == {"encoder__encoding", "model__C"}
    assert clone(QuantumKernelSVC(C=3.0)).get_params() == {"C": 3.0}


def test_off_diagonal_excludes_the_diagonal() -> None:
    kernel = np.array([[1.0, 0.2], [0.3, 1.0]])
    assert sorted(off_diagonal_values(kernel).tolist()) == [0.2, 0.3]


def test_alignment_is_one_for_the_ideal_kernel() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0])
    signed = np.where(labels == 1, 1.0, -1.0)
    assert centered_kernel_target_alignment(np.outer(signed, signed), labels) == pytest.approx(1.0)


def test_alignment_is_low_for_a_memorizing_kernel() -> None:
    # A identidade (cada linha só parece consigo mesma) não carrega informação de classe
    labels = np.array([0, 1] * 25)
    assert centered_kernel_target_alignment(np.eye(50), labels) < 0.2
