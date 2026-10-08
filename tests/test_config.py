"""Invariantes do protocolo experimental definido em ``config``."""

import math

from fraud_qml import config
from fraud_qml.features import MODEL_FEATURE_COLUMNS


def test_split_fractions_sum_to_one() -> None:
    total = config.TRAIN_SIZE + config.VALIDATION_SIZE + config.TEST_SIZE
    assert math.isclose(total, 1.0)


def test_main_seed_is_first_robustness_seed() -> None:
    # A análise de robustez precisa conter o experimento principal como uma das repetições
    assert config.UNDERSAMPLING_SEEDS[0] == config.RANDOM_STATE


def test_robustness_seeds_are_unique() -> None:
    # Semente repetida faria duas "repetições" idênticas e subestimaria a variância
    assert len(set(config.UNDERSAMPLING_SEEDS)) == len(config.UNDERSAMPLING_SEEDS)


def test_amplitude_budget_fits_all_features_plus_constant() -> None:
    # 28 componentes V + log(Amount) + seno e cosseno da hora = 31 features, +1 constante
    assert 2**config.N_QUBITS == len(MODEL_FEATURE_COLUMNS) + 1


def test_angle_widths_respect_the_period_of_qiskit_feature_maps() -> None:
    # Com α = 2 o período em x é π: acima de π/2, extremos opostos voltam a se parecer (CC11)
    assert all(0 < width <= math.pi / 2 for width in config.ANGLE_RANGE_WIDTH_GRID)


def test_amplitude_padding_never_discards_the_norm() -> None:
    # Completar com zero faz x e 2x virarem o mesmo estado (CC4)
    assert all(value != 0 for value in config.AMPLITUDE_PADDING_GRID)


def test_feature_map_grid_includes_the_pure_phase_encoding() -> None:
    # reps=1 é o único caso em que o mapa Z equivale à codificação de fase (e à de ângulo) pura
    assert 1 in config.FEATURE_MAP_REPS_GRID and all(reps >= 1 for reps in config.FEATURE_MAP_REPS_GRID)


def test_project_root_contains_pyproject() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
