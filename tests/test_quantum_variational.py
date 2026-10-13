"""Classificador variacional (classificador fraco do Quantum AdaBoost)."""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer.primitives import EstimatorV2
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from fraud_qml.preprocessing import AmplitudeEncodingPreparer
from fraud_qml.quantum.encodings import (
    StatevectorEncoder,
    build_angle_feature_map,
    features_to_states,
    simulate_statevectors,
)
from fraud_qml.quantum.variational import (
    VariationalQuantumClassifier,
    ansatz_unitary,
    build_ansatz,
    readout_one_probability,
    weighted_cross_entropy,
)

RNG_SEED = 0
N_QUBITS = 3
READOUT_Z = SparsePauliOp("I" * (N_QUBITS - 1) + "Z")


@pytest.fixture
def angles() -> np.ndarray:
    return np.random.default_rng(RNG_SEED).uniform(0, np.pi / 2, size=(8, N_QUBITS))


def _separable_problem(n_rows: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Ângulos em que a classe depende só da primeira feature (a do qubit medido)."""
    rng = np.random.default_rng(RNG_SEED)
    labels = np.array([0, 1] * (n_rows // 2))
    rows = rng.uniform(0, np.pi / 2, size=(n_rows, N_QUBITS))
    rows[:, 0] = rng.uniform(0, np.pi / 5, size=n_rows) + labels * 0.3 * np.pi
    return StatevectorEncoder(encoding="ry").fit_transform(rows), labels


@pytest.mark.parametrize(("encoding", "ansatz_reps", "n_qubits"), [("zz", 1, 3), ("z", 2, 3), ("ry", 1, 3), ("zz", 1, 5)])
def test_readout_matches_the_aer_estimator_on_the_full_circuit(encoding: str, ansatz_reps: int, n_qubits: int) -> None:
    # O produto matricial (estado codificado × matriz do ansatz) é o circuito feature map + ansatz.
    # O caso de 5 qubits é o tamanho usado no experimento
    angles = np.random.default_rng(RNG_SEED).uniform(0, np.pi / 2, size=(8, n_qubits))
    feature_map = build_angle_feature_map(encoding, n_qubits)
    ansatz = build_ansatz(n_qubits, ansatz_reps)
    weights = np.random.default_rng(RNG_SEED).uniform(-np.pi, np.pi, ansatz.num_parameters)
    circuit = feature_map.compose(ansatz.assign_parameters(weights))
    readout = SparsePauliOp("I" * (n_qubits - 1) + "Z")
    expectations = EstimatorV2().run([(circuit, readout, angles)]).result()[0].data.evs
    probability = readout_one_probability(simulate_statevectors(feature_map, angles), ansatz_unitary(ansatz, weights))
    np.testing.assert_allclose(1.0 - 2.0 * probability, expectations, atol=1e-9)


def test_readout_matches_qiskit_for_amplitude_encoding() -> None:
    rng = np.random.default_rng(RNG_SEED)
    amplitudes = AmplitudeEncodingPreparer(n_qubits=N_QUBITS).fit_transform(rng.normal(size=(5, 7)))
    ansatz = build_ansatz(N_QUBITS)
    weights = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
    bound_ansatz = ansatz.assign_parameters(weights)
    expectations = [Statevector(row).evolve(bound_ansatz).expectation_value(READOUT_Z).real for row in amplitudes]
    states = features_to_states(StatevectorEncoder(encoding="amplitude").fit_transform(amplitudes))
    probability = readout_one_probability(states, ansatz_unitary(ansatz, weights))
    np.testing.assert_allclose(1.0 - 2.0 * probability, expectations, atol=1e-12)


def test_ansatz_matrix_is_real() -> None:
    # RY e CNOT têm matrizes reais: o ansatz não mistura a parte real com a imaginária do estado (CC14)
    ansatz = build_ansatz(N_QUBITS, reps=2)
    assert ansatz.num_parameters == N_QUBITS * 3
    weights = np.random.default_rng(RNG_SEED).uniform(-np.pi, np.pi, ansatz.num_parameters)
    assert np.abs(ansatz_unitary(ansatz, weights).imag).max() == 0.0


def test_weighted_cross_entropy_with_equal_weights_is_the_log_loss() -> None:
    rng = np.random.default_rng(RNG_SEED)
    probability = rng.uniform(0.05, 0.95, size=30)
    labels = rng.integers(0, 2, size=30)
    assert weighted_cross_entropy(probability, labels == 1, np.ones(30)) == pytest.approx(log_loss(labels, probability))


def test_weighted_cross_entropy_ignores_zero_weight_rows_and_survives_certainty() -> None:
    probability = np.array([0.9, 0.2, 0.0])
    is_positive = np.array([True, False, True])
    # A terceira linha (probabilidade 0 para a classe certa) teria perda enorme, mas pesa zero
    with_zero = weighted_cross_entropy(probability, is_positive, np.array([1.0, 3.0, 0.0]))
    assert with_zero == pytest.approx(weighted_cross_entropy(probability[:2], is_positive[:2], np.array([1.0, 3.0])))
    assert np.isfinite(weighted_cross_entropy(probability, is_positive, np.ones(3)))


def test_zero_weight_is_the_same_as_removing_the_row() -> None:
    features, labels = _separable_problem()
    weights = np.random.default_rng(RNG_SEED).uniform(0.5, 2.0, size=len(labels))
    weights[[3, 10, 11, 40]] = 0.0
    kept = weights > 0
    classifier = VariationalQuantumClassifier(max_iterations=40, random_state=RNG_SEED)
    with_zeros = clone(classifier).fit(features, labels, sample_weight=weights)
    without_rows = clone(classifier).fit(features[kept], labels[kept], sample_weight=weights[kept])
    # A função de custo é a mesma nos dois casos: o otimizador percorre o mesmo caminho, e o custo coincide
    # em todas as avaliações menos a última (o passo final já é tão pequeno que o arredondamento o desvia)
    np.testing.assert_allclose(with_zeros.loss_curve_[:-1], without_rows.loss_curve_[:-1], rtol=0, atol=1e-12)
    # Diferenças medidas no ponto final: ≈ 1e-3 nos pesos e ≈ 3e-4 nas probabilidades; as tolerâncias ficam
    # mais de 10 vezes acima
    np.testing.assert_allclose(with_zeros.weights_, without_rows.weights_, atol=2e-2)
    np.testing.assert_allclose(with_zeros.predict_proba(features), without_rows.predict_proba(features), atol=5e-3)


def test_sample_weights_change_what_is_learned() -> None:
    features, labels = _separable_problem()
    classifier = VariationalQuantumClassifier(max_iterations=40, random_state=RNG_SEED)
    uniform = clone(classifier).fit(features, labels)
    # Quase todo o peso na classe 0: o classificador deve passar a prever menos a classe 1
    skewed = clone(classifier).fit(features, labels, sample_weight=np.where(labels == 0, 50.0, 1.0))
    assert skewed.predict_proba(features)[:, 1].mean() < uniform.predict_proba(features)[:, 1].mean() - 0.05


def test_training_reduces_the_loss_and_respects_the_budget() -> None:
    features, labels = _separable_problem()
    classifier = VariationalQuantumClassifier(max_iterations=25, random_state=RNG_SEED).fit(features, labels)
    assert len(classifier.loss_curve_) <= 25
    assert classifier.loss_curve_.min() < classifier.loss_curve_[0]


def test_learns_a_separable_problem() -> None:
    features, labels = _separable_problem()
    classifier = VariationalQuantumClassifier(max_iterations=200, random_state=RNG_SEED).fit(features, labels)
    assert (classifier.predict(features) == labels).mean() > 0.9


def test_outputs_are_consistent_and_keep_the_original_labels() -> None:
    features, labels = _separable_problem()
    named_labels = np.where(labels == 1, 7, 3)
    classifier = VariationalQuantumClassifier(max_iterations=30, random_state=RNG_SEED).fit(features, named_labels)
    probabilities = classifier.predict_proba(features)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    decision = classifier.decision_function(features)
    assert decision.min() >= -1.0 and decision.max() <= 1.0
    np.testing.assert_array_equal(classifier.predict(features), np.where(decision > 0, 7, 3))


def test_rejects_invalid_labels_and_weights() -> None:
    features, labels = _separable_problem()
    with pytest.raises(ValueError, match="binário"):
        VariationalQuantumClassifier().fit(features, np.arange(len(labels)) % 3)
    with pytest.raises(ValueError, match="não negativo"):
        VariationalQuantumClassifier().fit(features, labels, sample_weight=-np.ones(len(labels)))
    with pytest.raises(ValueError, match="formato"):
        VariationalQuantumClassifier().fit(features, labels, sample_weight=np.ones(3))
    # 3 qubits e 1 repetição = 6 pesos: o COBYLA precisa de pelo menos 8 avaliações
    with pytest.raises(ValueError, match="mínimo do COBYLA"):
        VariationalQuantumClassifier(max_iterations=7).fit(features, labels)
    VariationalQuantumClassifier(max_iterations=8).fit(features, labels)


def test_is_a_valid_weak_learner_for_adaboost() -> None:
    features, labels = _separable_problem()
    weak_learner = VariationalQuantumClassifier(max_iterations=30)
    boosted = AdaBoostClassifier(estimator=weak_learner, n_estimators=4, random_state=RNG_SEED).fit(features, labels)
    assert 1 <= len(boosted.estimators_) <= 4
    # O AdaBoost dá uma semente diferente a cada rodada: os classificadores fracos partem de pesos diferentes
    assert len({estimator.random_state for estimator in boosted.estimators_}) == len(boosted.estimators_)
    assert boosted.decision_function(features).shape == (len(labels),)


def test_works_inside_a_grid_search() -> None:
    rng = np.random.default_rng(RNG_SEED)
    labels = np.array([0, 1] * 20)
    rows = rng.uniform(0, np.pi / 4, size=(40, N_QUBITS)) + labels[:, None] * np.pi / 4
    pipeline = Pipeline(
        [("encoder", StatevectorEncoder(encoding="ry")), ("model", VariationalQuantumClassifier(max_iterations=15, random_state=RNG_SEED))]
    )
    search = GridSearchCV(pipeline, {"model__ansatz_reps": [1, 2]}, cv=2, scoring="roc_auc").fit(rows, labels)
    assert search.best_params_["model__ansatz_reps"] in (1, 2)
    params = clone(VariationalQuantumClassifier(ansatz_reps=2, max_iterations=7, random_state=3)).get_params()
    assert params == {"ansatz_reps": 2, "max_iterations": 7, "random_state": 3}


def test_phase_is_invisible_to_a_direct_measurement_until_the_second_repetition() -> None:
    # Num qubit, medido logo depois da codificação (sem ansatz): a fase pura dá P(1) = 0,5 para qualquer x;
    # com 2 repetições do mapa Z, a segunda Hadamard traz a fase para o eixo medido e P(1) = sen²(x), o
    # mesmo da rotação em Y (CC14)
    values = np.array([[0.2], [0.7], [1.3]])
    identity = np.eye(2, dtype=complex)

    def measured(name: str, reps: int) -> np.ndarray:
        return readout_one_probability(simulate_statevectors(build_angle_feature_map(name, 1, reps), values), identity)

    np.testing.assert_allclose(measured("z", 1), 0.5, atol=1e-12)
    np.testing.assert_allclose(measured("z", 2), np.sin(values[:, 0]) ** 2, atol=1e-12)
    np.testing.assert_allclose(measured("ry", 1), np.sin(values[:, 0]) ** 2, atol=1e-12)
