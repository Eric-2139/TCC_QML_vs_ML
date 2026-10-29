"""Fábrica dos pipelines e medição de um ajuste."""

import numpy as np
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

from fraud_qml import config
from fraud_qml.models import (
    InputEncoding,
    ModelSettings,
    build_adaboost_pipeline,
    build_pipeline,
    build_dummy_pipeline,
    build_qsvm_pipeline,
    build_quantum_adaboost_pipeline,
    build_svm_pipeline,
    build_weak_quantum_classifier_pipeline,
    fit_and_score,
    quantum_encoding_steps,
    replay_boosting,
)
from fraud_qml.quantum.encodings import StatevectorEncoder
from fraud_qml.quantum.variational import VariationalQuantumClassifier

RNG_SEED = 0
N_FEATURES = 31
N_FRAUDS = 30


def _imbalanced_data() -> tuple[np.ndarray, np.ndarray]:
    """31 features (como no projeto), 30 fraudes em 400 linhas; as fraudes ficam deslocadas em 4 features."""
    rng = np.random.default_rng(RNG_SEED)
    labels = np.array([1] * N_FRAUDS + [0] * 370)
    features = rng.normal(size=(400, N_FEATURES))
    features[:N_FRAUDS, :4] += 2.5
    return features, labels


def _all_pipelines() -> dict[str, ImbPipeline]:
    return {
        "dummy": build_dummy_pipeline(),
        "svm": build_svm_pipeline(),
        "svm_like_for_like": build_svm_pipeline("angle_components"),
        "adaboost": build_adaboost_pipeline(n_estimators=5),
        "adaboost_like_for_like": build_adaboost_pipeline("angle_components", n_estimators=5),
        "qsvm_angle": build_qsvm_pipeline(quantum_encoding_steps("angle")),
        "qsvm_amplitude": build_qsvm_pipeline(quantum_encoding_steps("amplitude")),
        "quantum_adaboost_angle": build_quantum_adaboost_pipeline(
            quantum_encoding_steps("angle"), n_estimators=2, max_iterations=15
        ),
        "quantum_adaboost_amplitude": build_quantum_adaboost_pipeline(
            quantum_encoding_steps("amplitude"), n_estimators=2, max_iterations=15
        ),
    }


@pytest.mark.parametrize("name", list(_all_pipelines()))
def test_every_pipeline_undersamples_first_and_scores_every_row(name: str) -> None:
    features, labels = _imbalanced_data()
    pipeline = _all_pipelines()[name]
    assert [step_name for step_name, _ in pipeline.steps][0] == "undersampler"
    assert pipeline.steps[-1][0] == "model"
    run = fit_and_score(pipeline, features, labels, features)
    # Na avaliação nenhuma linha é descartada, e as classes previstas são as do próprio pipeline
    assert run.scores.shape == run.predictions.shape == (400,)
    np.testing.assert_array_equal(run.predictions, pipeline.predict(features))
    assert run.fit_cost.wall_seconds > 0 and run.predict_cost.wall_seconds >= 0
    assert np.isnan(run.fit_cost.python_peak_mib)


@pytest.mark.parametrize("name", ["svm", "adaboost", "qsvm_angle", "qsvm_amplitude", "quantum_adaboost_amplitude"])
def test_pipelines_rank_frauds_above_legitimate_rows_on_an_easy_problem(name: str) -> None:
    features, labels = _imbalanced_data()
    run = fit_and_score(_all_pipelines()[name], features, labels, features)
    assert run.scores[labels == 1].mean() > run.scores[labels == 0].mean()


def test_quantum_steps_have_the_documented_names() -> None:
    angle_names = [name for name, _ in quantum_encoding_steps("angle")]
    amplitude_names = [name for name, _ in quantum_encoding_steps("amplitude")]
    assert angle_names == ["scaler", "reducer", "angle_scaler", "encoder"]
    assert amplitude_names == ["scaler", "amplitude", "encoder"]


def test_quantum_variations_use_the_same_qubit_budget() -> None:
    features, labels = _imbalanced_data()
    for variation in ("angle", "amplitude"):
        pipeline = build_qsvm_pipeline(quantum_encoding_steps(variation)).fit(features, labels)
        assert pipeline.named_steps["encoder"].n_qubits_ == config.N_QUBITS


def test_classical_like_for_like_input_is_the_quantum_angle_input() -> None:
    # "Entrada idêntica": o SVM clássico recebe exatamente os 5 valores que, no quântico, viram ângulos
    features, labels = _imbalanced_data()
    classical = build_svm_pipeline("angle_components").fit(features, labels)
    quantum = build_qsvm_pipeline(quantum_encoding_steps("angle")).fit(features, labels)
    classical_input = features
    quantum_input = features
    for name in ("scaler", "reducer", "angle_scaler"):
        classical_input = classical.named_steps[name].transform(classical_input)
        quantum_input = quantum.named_steps[name].transform(quantum_input)
    np.testing.assert_array_equal(classical_input, quantum_input)
    assert classical_input.shape[1] == config.N_QUBITS


def test_both_adaboosts_are_the_same_booster_with_a_different_weak_learner() -> None:
    classical = build_adaboost_pipeline().named_steps["model"]
    quantum = build_quantum_adaboost_pipeline(quantum_encoding_steps("angle")).named_steps["model"]
    assert type(classical) is type(quantum) is AdaBoostClassifier
    assert isinstance(quantum.estimator, VariationalQuantumClassifier)
    shared = ("n_estimators", "learning_rate", "random_state")
    assert {name: classical.get_params()[name] for name in shared} == {name: quantum.get_params()[name] for name in shared}


def test_adaboost_with_stumps_ignores_the_scaler() -> None:
    # Um stump só compara uma feature com um limiar; reescalar a feature reescala o limiar junto
    rng = np.random.default_rng(RNG_SEED)
    features, labels = _imbalanced_data()
    features = features * rng.uniform(0.01, 1000, size=N_FEATURES)
    booster = AdaBoostClassifier(n_estimators=20, random_state=RNG_SEED)
    raw = clone(booster).fit(features, labels)
    scaler = StandardScaler().fit(features)
    scaled = clone(booster).fit(scaler.transform(features), labels)
    np.testing.assert_array_equal(raw.predict(features), scaled.predict(scaler.transform(features)))
    np.testing.assert_allclose(
        raw.decision_function(features), scaled.decision_function(scaler.transform(features)), atol=1e-12
    )


def _default_descriptions() -> dict[str, tuple[InputEncoding, ModelSettings]]:
    return {
        "svm": (InputEncoding("all_features"), ModelSettings("svm")),
        "svm_like_for_like": (InputEncoding("angle_components"), ModelSettings("svm")),
        "adaboost": (InputEncoding("all_features"), ModelSettings("adaboost", n_estimators=5)),
        "adaboost_like_for_like": (InputEncoding("angle_components"), ModelSettings("adaboost", n_estimators=5)),
        "qsvm_angle": (InputEncoding("angle"), ModelSettings("qsvm")),
        "qsvm_amplitude": (InputEncoding("amplitude"), ModelSettings("qsvm")),
        "quantum_adaboost_angle": (InputEncoding("angle"), ModelSettings("quantum_adaboost", n_estimators=2, max_iterations=15)),
        "quantum_adaboost_amplitude": (
            InputEncoding("amplitude"), ModelSettings("quantum_adaboost", n_estimators=2, max_iterations=15),
        ),
    }


@pytest.mark.parametrize("name", list(_default_descriptions()))
def test_generic_factory_with_defaults_gives_the_section_5_pipelines(name: str) -> None:
    # O grid e os modelos finais usam build_pipeline; com os padrões, ele precisa dar os mesmos pipelines da Seção 5
    features, labels = _imbalanced_data()
    generic = build_pipeline(*_default_descriptions()[name]).fit(features, labels)
    specific = _all_pipelines()[name].fit(features, labels)
    assert [step for step, _ in generic.steps] == [step for step, _ in specific.steps]
    np.testing.assert_allclose(generic.decision_function(features), specific.decision_function(features), atol=1e-12)


def test_classical_reducer_is_a_parameter() -> None:
    features, labels = _imbalanced_data()
    pipeline = build_svm_pipeline("angle_components", reducer="mutual_info").fit(features, labels)
    assert type(pipeline.named_steps["reducer"]).__name__ == "SelectKBest"
    assert pipeline.named_steps["reducer"].transform(features).shape[1] == config.N_QUBITS


def test_generic_factory_rejects_a_quantum_model_on_a_classical_input() -> None:
    with pytest.raises(ValueError, match="não aceita"):
        build_pipeline(InputEncoding("all_features"), ModelSettings("qsvm"))
    with pytest.raises(ValueError, match="não aceita"):
        build_pipeline(InputEncoding("amplitude"), ModelSettings("svm"))


def test_weak_classifier_pipeline_has_no_undersampler() -> None:
    features, labels = _imbalanced_data()
    pipeline = build_weak_quantum_classifier_pipeline(quantum_encoding_steps("angle", feature_map="ry"), max_iterations=15)
    assert "undersampler" not in pipeline.named_steps
    pipeline.fit(features, labels)
    assert pipeline.named_steps["encoder"].encoding == "ry"
    assert pipeline.predict_proba(features).shape == (400, 2)


def test_pipelines_are_cloneable_with_grid_parameter_names() -> None:
    pipeline = build_quantum_adaboost_pipeline(quantum_encoding_steps("angle"))
    parameters = clone(pipeline).get_params()
    for name in ("scaler", "angle_scaler__feature_range", "encoder__encoding", "encoder__feature_map_reps",
                 "model__n_estimators", "model__learning_rate", "model__estimator__ansatz_reps"):
        assert name in parameters
    assert "model__C" in build_qsvm_pipeline(quantum_encoding_steps("amplitude")).get_params()


@pytest.mark.parametrize("weak_learner", ["stump", "variational"])
def test_boosting_replay_reproduces_what_the_booster_did(weak_learner: str) -> None:
    rng = np.random.default_rng(RNG_SEED)
    labels = np.array([0, 1] * 40)
    features = rng.uniform(0, np.pi / 2, size=(80, 3))
    features[:, 0] = rng.uniform(0, np.pi / 4, size=80) + labels * np.pi / 5
    estimator = None
    if weak_learner == "variational":
        features = StatevectorEncoder(encoding="ry").fit_transform(features)
        estimator = VariationalQuantumClassifier(max_iterations=20)
    rounds = 4
    shorter = AdaBoostClassifier(estimator=estimator, n_estimators=rounds, random_state=RNG_SEED).fit(features, labels)
    longer = AdaBoostClassifier(estimator=estimator, n_estimators=rounds + 1, random_state=RNG_SEED).fit(features, labels)
    assert len(shorter.estimators_) == rounds and len(longer.estimators_) == rounds + 1, "o exemplo não pode parar cedo"
    replay = replay_boosting(shorter, features, labels)
    # Os erros refeitos são os que o booster guardou, e o "classificador seguinte" é a rodada a mais do booster maior
    np.testing.assert_allclose(replay.errors, shorter.estimator_errors_[:rounds], atol=1e-10)
    assert replay.next_error == pytest.approx(longer.estimator_errors_[rounds], abs=1e-10)
    assert replay.next_sample_weight.sum() == pytest.approx(1.0)


def test_boosting_replay_refuses_an_unexpected_seed_sequence() -> None:
    features, labels = _imbalanced_data()
    booster = AdaBoostClassifier(n_estimators=3, random_state=RNG_SEED).fit(features, labels)
    booster.estimators_[1].set_params(random_state=123)
    with pytest.raises(ValueError, match="sementes"):
        replay_boosting(booster, features, labels)
