"""A medida de *barren plateau* (Seção 8): leitura global, derivada exata e a queda da variância com os qubits."""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

from fraud_qml.quantum.diagnostics import (
    LANDSCAPE_COLUMNS,
    barren_plateau_sweep,
    cost_and_gradient,
    parity_one_probability,
    readout_bloch_length,
)
from fraud_qml.quantum.variational import ansatz_unitary, build_ansatz, readout_one_probability, weighted_cross_entropy

RNG_SEED = 0
N_ROWS = 40
FINITE_DIFFERENCE_STEP = 1e-6
SAMPLES = 150


def _random_states(n_qubits: int, n_rows: int = N_ROWS, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n_rows, 2**n_qubits)) + 1j * rng.normal(size=(n_rows, 2**n_qubits))
    return states / np.linalg.norm(states, axis=1, keepdims=True)


def _labels(n_rows: int = N_ROWS) -> np.ndarray:
    return np.repeat([0, 1], n_rows // 2)


def test_parity_probability_matches_the_qiskit_expectation_value() -> None:
    n_qubits = 3
    states = _random_states(n_qubits, n_rows=5)
    ansatz = build_ansatz(n_qubits, 2)
    unitary = ansatz_unitary(ansatz, np.random.default_rng(RNG_SEED).uniform(-np.pi, np.pi, ansatz.num_parameters))
    parity = SparsePauliOp("Z" * n_qubits)
    expected = [(1 - Statevector(unitary @ state).expectation_value(parity).real) / 2 for state in states]
    np.testing.assert_allclose(parity_one_probability(states, unitary), expected, atol=1e-12)


@pytest.mark.parametrize("observable", ["local", "global"])
@pytest.mark.parametrize("parameter_index", [0, 5])
def test_parameter_shift_gradient_equals_the_finite_difference(observable: str, parameter_index: int) -> None:
    # A regra do parameter-shift é exata para rotações RY; a diferença finita central confere o valor e a regra da cadeia
    n_qubits = 3
    states = _random_states(n_qubits)
    is_positive = _labels() == 1
    ansatz = build_ansatz(n_qubits, 2)
    weights = np.random.default_rng(RNG_SEED).uniform(-np.pi, np.pi, ansatz.num_parameters)
    cost, gradient = cost_and_gradient(states, is_positive, ansatz, weights, parameter_index, observable)

    def cost_at(values: np.ndarray) -> float:
        unitary = ansatz_unitary(ansatz, values)
        probability = readout_one_probability(states, unitary) if observable == "local" else parity_one_probability(states, unitary)
        return weighted_cross_entropy(probability, is_positive, np.ones(len(states)))

    step = np.zeros_like(weights)
    step[parameter_index] = FINITE_DIFFERENCE_STEP
    numeric = (cost_at(weights + step) - cost_at(weights - step)) / (2 * FINITE_DIFFERENCE_STEP)
    assert cost == pytest.approx(cost_at(weights), abs=1e-12)
    assert gradient == pytest.approx(numeric, abs=1e-6)


def test_random_states_flatten_the_cost_as_qubits_grow() -> None:
    # Com estados aleatórios, os valores esperados se concentram em torno da média, mais a cada qubit: de 2 para 6
    # qubits a variância da derivada cai cerca de 20 vezes (um fator 2⁴ = 16 seria a queda exponencial pura). A margem
    # de 4 vezes deixa folga para o ruído de 150 sorteios
    state_sets = {f"{n} qubits": (_random_states(n), _labels()) for n in (2, 6)}
    sweep = barren_plateau_sweep(state_sets, (1,), ("global",), SAMPLES, RNG_SEED)
    variance = sweep.groupby("n_qubits")["gradient"].var()
    assert variance[6] < variance[2] / 4


def test_sweep_has_one_row_per_sample_and_observable() -> None:
    state_sets = {"amostra": (_random_states(2, n_rows=8), _labels(8)), "maior": (_random_states(3, n_rows=8), _labels(8))}
    sweep = barren_plateau_sweep(state_sets, (1, 2), ("local", "global"), 3, RNG_SEED)
    assert list(sweep.columns) == list(LANDSCAPE_COLUMNS)
    assert len(sweep) == 2 * 2 * 2 * 3
    assert sweep.groupby("states")["n_qubits"].unique().to_dict() == {"amostra": [2], "maior": [3]}


def test_sweep_does_not_depend_on_the_number_of_processes() -> None:
    state_sets = {"amostra": (_random_states(2, n_rows=8), _labels(8))}
    serial = barren_plateau_sweep(state_sets, (1, 2), ("local",), 4, RNG_SEED, n_jobs=1)
    parallel = barren_plateau_sweep(state_sets, (1, 2), ("local",), 4, RNG_SEED, n_jobs=2)
    np.testing.assert_allclose(serial["gradient"], parallel["gradient"])


def test_bloch_length_is_one_for_a_product_state_and_zero_for_a_bell_pair() -> None:
    plus_and_zero = np.kron(np.array([1, 0]), np.array([1, 1]) / np.sqrt(2)).astype(complex)  # qubit 0 em |+⟩
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    lengths = readout_bloch_length(np.vstack([plus_and_zero, bell]))
    np.testing.assert_allclose(lengths, [1.0, 0.0], atol=1e-12)


def test_bloch_length_matches_the_qiskit_reduced_state() -> None:
    from qiskit.quantum_info import partial_trace

    states = _random_states(3, n_rows=4)
    for state, length in zip(states, readout_bloch_length(states), strict=True):
        reduced = partial_trace(Statevector(state), [1, 2]).data
        pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
        vector = [np.trace(reduced @ matrix).real for matrix in pauli]
        assert length == pytest.approx(np.linalg.norm(vector), abs=1e-12)
