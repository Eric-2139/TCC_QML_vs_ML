"""Escalonamento, redução, preparo de amplitude e undersampling."""

import numpy as np
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from fraud_qml.preprocessing import (
    AmplitudeEncodingPreparer,
    build_amplitude_preprocessor,
    build_angle_preprocessor,
    make_undersampler,
)

RNG_SEED = 0


@pytest.fixture
def features() -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    # Escalas bem diferentes por coluna, como Time/Amount contra as V
    return rng.normal(size=(200, 6)) * np.array([1, 5, 50, 0.1, 2, 1000])


def test_standard_then_minmax_equals_minmax_alone(features: np.ndarray) -> None:
    # Duas transformações afins por coluna em sequência: a segunda anula a primeira
    chained = MinMaxScaler().fit_transform(StandardScaler().fit_transform(features))
    direct = MinMaxScaler().fit_transform(features)
    np.testing.assert_allclose(chained, direct, atol=1e-12)


def test_robust_then_minmax_equals_minmax_alone(features: np.ndarray) -> None:
    # O RobustScaler também é afim por coluna (mediana e IQR): o MinMax desfaz o efeito dele
    chained = MinMaxScaler().fit_transform(RobustScaler().fit_transform(features))
    np.testing.assert_allclose(chained, MinMaxScaler().fit_transform(features), atol=1e-12)


def test_scaler_is_irrelevant_on_the_mutual_information_path(features: np.ndarray) -> None:
    # A informação mútua é invariante a esticar e deslocar cada coluna (o estimador reescala por
    # conta própria) e o MinMax final desfaz o scaler: Standard e Robust dão o mesmo resultado
    labels = (features[:, 0] > 0).astype(int)
    outputs = [
        build_angle_preprocessor(scaler=name, reducer="mutual_info", n_components=3, random_state=RNG_SEED)
        .fit_transform(features, labels)
        for name in ("standard", "robust")
    ]
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-12)


def test_standard_matters_when_pca_is_in_between(features: np.ndarray) -> None:
    with_standard = build_angle_preprocessor(scaler="standard", n_components=2).fit_transform(features)
    without_standard = (
        build_angle_preprocessor(scaler="standard", n_components=2)
        .set_params(scaler="passthrough")
        .fit_transform(features)
    )
    assert not np.allclose(with_standard, without_standard)


def test_angle_preprocessor_output_shape_and_range(features: np.ndarray) -> None:
    angles = build_angle_preprocessor(n_components=3, angle_range=(0.0, np.pi)).fit_transform(features)
    assert angles.shape == (200, 3)
    assert angles.min() >= 0.0 and angles.max() <= np.pi


def test_angle_preprocessor_clips_values_beyond_training_range(features: np.ndarray) -> None:
    preprocessor = build_angle_preprocessor(n_components=2, angle_range=(0.0, np.pi)).fit(features)
    # Uma linha muito mais extrema que qualquer uma do treino não pode "dar a volta" no círculo
    extreme_row = features[:1] * 100
    angles = preprocessor.transform(extreme_row)
    assert angles.min() >= 0.0 and angles.max() <= np.pi


def test_mutual_info_reducer_keeps_the_informative_column() -> None:
    rng = np.random.default_rng(RNG_SEED)
    labels = rng.integers(0, 2, size=400)
    noise = rng.normal(size=(400, 4))
    informative = labels + rng.normal(scale=0.1, size=400)
    features = np.column_stack([noise[:, :2], informative, noise[:, 2:]])
    preprocessor = build_angle_preprocessor(reducer="mutual_info", n_components=1).fit(features, labels)
    assert preprocessor.named_steps["reducer"].get_support().nonzero()[0].tolist() == [2]


def test_amplitude_output_has_unit_norm_and_power_of_two_width(features: np.ndarray) -> None:
    amplitudes = build_amplitude_preprocessor(n_qubits=3).fit_transform(features)
    assert amplitudes.shape == (200, 8)
    np.testing.assert_allclose(np.linalg.norm(amplitudes, axis=1), 1.0)


def test_constant_padding_preserves_norm_information() -> None:
    vector = np.array([[1.0, 2.0, 0.5]])
    doubled = 2 * vector
    with_constant = AmplitudeEncodingPreparer(n_qubits=2, padding_value=1.0).fit(vector)
    with_zero = AmplitudeEncodingPreparer(n_qubits=2, padding_value=0.0).fit(vector)
    # Completando com zero, x e 2x viram o mesmo estado: a norma se perde
    np.testing.assert_allclose(with_zero.transform(vector), with_zero.transform(doubled))
    # Completando com constante, os estados diferem
    assert not np.allclose(with_constant.transform(vector), with_constant.transform(doubled))


def test_amplitude_rejects_too_many_features() -> None:
    with pytest.raises(ValueError, match="não cabem"):
        AmplitudeEncodingPreparer(n_qubits=2).fit(np.ones((3, 5)))


def test_preprocessors_are_cloneable() -> None:
    # GridSearchCV clona cada etapa; parâmetros precisam sobreviver ao clone
    angle = build_angle_preprocessor(scaler="robust", reducer="mutual_info", n_components=4)
    assert clone(angle).get_params()["reducer__k"] == 4
    amplitude = build_amplitude_preprocessor(padding_value=0.5)
    assert clone(amplitude).get_params()["amplitude__padding_value"] == 0.5


class _RowCountingClassifier(ClassifierMixin, BaseEstimator):
    """Classificador falso que registra quantas linhas de cada classe viu no fit e no predict."""

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_RowCountingClassifier":
        self.fit_class_counts_ = np.bincount(labels)
        self.classes_ = np.unique(labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.predicted_rows_ = len(features)
        return np.zeros(len(features), dtype=int)


def test_undersampler_balances_training_and_skips_prediction() -> None:
    rng = np.random.default_rng(RNG_SEED)
    labels = np.array([1] * 20 + [0] * 980)
    features = rng.normal(size=(1000, 3))
    pipeline = ImbPipeline([("undersampler", make_undersampler(RNG_SEED)), ("model", _RowCountingClassifier())])
    pipeline.fit(features, labels)
    assert pipeline.named_steps["model"].fit_class_counts_.tolist() == [20, 20]
    # Na predição nada é descartado: a proporção real é preservada
    pipeline.predict(features)
    assert pipeline.named_steps["model"].predicted_rows_ == 1000
