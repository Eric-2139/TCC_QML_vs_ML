"""Feature maps, simulação em lote e a etapa de codificação do pipeline."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from sklearn.base import clone

from fraud_qml.preprocessing import AmplitudeEncodingPreparer
from fraud_qml.quantum.encodings import (
    StatevectorEncoder,
    build_angle_feature_map,
    features_to_states,
    simulate_statevectors,
    states_to_features,
)
from fraud_qml.quantum.kernels import angle_kernel_matrix

RNG_SEED = 0
N_QUBITS = 3


@pytest.fixture
def angles() -> np.ndarray:
    return np.random.default_rng(RNG_SEED).uniform(0, np.pi / 2, size=(10, N_QUBITS))


@pytest.mark.parametrize(
    ("name", "reps", "entanglement"),
    [("z", 1, "linear"), ("z", 2, "linear"), ("zz", 1, "linear"), ("zz", 2, "linear"), ("zz", 2, "full"), ("ry", 1, "linear")],
)
def test_batch_simulation_matches_one_circuit_at_a_time(angles: np.ndarray, name: str, reps: int, entanglement: str) -> None:
    # A simulação em lote no Aer só pode mudar o tempo: os estados são os do Statevector do Qiskit
    feature_map = build_angle_feature_map(name, N_QUBITS, reps, entanglement)
    one_at_a_time = np.array([Statevector(feature_map.assign_parameters(row)).data for row in angles])
    np.testing.assert_allclose(simulate_statevectors(feature_map, angles), one_at_a_time, atol=1e-12)


def test_batch_simulation_crosses_batch_boundaries(angles: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    feature_map = build_angle_feature_map("zz", N_QUBITS)
    whole = simulate_statevectors(feature_map, angles)
    monkeypatch.setattr("fraud_qml.quantum.encodings.SIMULATION_BATCH_ROWS", 3)
    np.testing.assert_allclose(simulate_statevectors(feature_map, angles), whole, atol=1e-12)


def test_batch_simulation_rejects_wrong_number_of_columns(angles: np.ndarray) -> None:
    with pytest.raises(ValueError, match="colunas de parâmetros"):
        simulate_statevectors(build_angle_feature_map("z", N_QUBITS), angles[:, :2])


def test_ry_map_has_the_kernel_of_the_single_layer_z_map(angles: np.ndarray) -> None:
    # Fase (mapa Z com uma camada) e rotação em Y dão o mesmo kernel: para o QSVM a troca não muda nada
    ry_kernel = angle_kernel_matrix(build_angle_feature_map("ry", N_QUBITS), angles)
    z_kernel = angle_kernel_matrix(build_angle_feature_map("z", N_QUBITS, reps=1), angles)
    np.testing.assert_allclose(ry_kernel, z_kernel, atol=1e-12)


def test_ry_map_ignores_repetitions(angles: np.ndarray) -> None:
    one_repetition = simulate_statevectors(build_angle_feature_map("ry", N_QUBITS, reps=1), angles)
    two_repetitions = simulate_statevectors(build_angle_feature_map("ry", N_QUBITS, reps=2), angles)
    np.testing.assert_array_equal(one_repetition, two_repetitions)


def test_states_round_trip_through_real_columns(angles: np.ndarray) -> None:
    states = simulate_statevectors(build_angle_feature_map("zz", N_QUBITS), angles)
    np.testing.assert_array_equal(features_to_states(states_to_features(states)), states)


def test_features_to_states_rejects_impossible_widths() -> None:
    with pytest.raises(ValueError, match="2 · 2ⁿ"):
        features_to_states(np.ones((2, 12)))


@pytest.mark.parametrize("encoding", ["z", "zz", "ry"])
def test_encoder_outputs_normalized_states(angles: np.ndarray, encoding: str) -> None:
    encoded = StatevectorEncoder(encoding=encoding).fit_transform(angles)
    assert encoded.shape == (10, 2 * 2**N_QUBITS)
    np.testing.assert_allclose(np.linalg.norm(features_to_states(encoded), axis=1), 1.0)


def test_encoder_matches_the_feature_map_it_reports(angles: np.ndarray) -> None:
    encoder = StatevectorEncoder(encoding="zz", feature_map_reps=1, entanglement="full").fit(angles)
    expected = simulate_statevectors(build_angle_feature_map("zz", N_QUBITS, 1, "full"), angles)
    np.testing.assert_allclose(features_to_states(encoder.transform(angles)), expected, atol=1e-12)
    assert encoder.build_feature_map().num_qubits == N_QUBITS


def test_amplitude_encoder_passes_the_amplitudes_through() -> None:
    rng = np.random.default_rng(RNG_SEED)
    amplitudes = AmplitudeEncodingPreparer(n_qubits=3).fit_transform(rng.normal(size=(6, 7)))
    encoder = StatevectorEncoder(encoding="amplitude").fit(amplitudes)
    assert encoder.n_qubits_ == 3
    np.testing.assert_array_equal(features_to_states(encoder.transform(amplitudes)), amplitudes.astype(complex))


def test_amplitude_encoder_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="2ⁿ colunas"):
        StatevectorEncoder(encoding="amplitude").fit(np.ones((2, 6)))
    encoder = StatevectorEncoder(encoding="amplitude").fit(np.full((2, 4), 0.5))
    with pytest.raises(ValueError, match="norma 1"):
        encoder.transform(np.ones((2, 4)))
    with pytest.raises(ValueError, match="feature map"):
        encoder.build_feature_map()


def test_encoder_is_cloneable() -> None:
    encoder = StatevectorEncoder(encoding="z", feature_map_reps=1, entanglement="full")
    assert clone(encoder).get_params() == {"encoding": "z", "entanglement": "full", "feature_map_reps": 1}
