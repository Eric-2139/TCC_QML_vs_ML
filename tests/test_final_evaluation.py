"""Avaliação final (Seção 9): pontuar sem reajustar e a robustez nas sementes."""

import numpy as np
import pandas as pd
import pytest

from fraud_qml.final_evaluation import (
    ROBUSTNESS_COLUMNS,
    robustness_point,
    robustness_scores,
    score_fitted_pipeline,
)
from fraud_qml.models import InputEncoding, ModelSettings, build_pipeline, fit_and_score
from fraud_qml.thresholds import best_f1_threshold, predictions_at_threshold
from fraud_qml.tuning import package_dependencies

RNG_SEED = 0
N_FEATURES = 31
N_FRAUDS = 60
N_ROWS = 600
SMALL_BUDGET = 15

CONFIGURATIONS: dict[str, tuple[InputEncoding, ModelSettings]] = {
    "SVM": (InputEncoding("all_features"), ModelSettings("svm", C=10.0, gamma=0.01)),
    "AdaBoost": (InputEncoding("angle_components", reducer="mutual_info"), ModelSettings("adaboost", n_estimators=5)),
    "QSVM B": (InputEncoding("amplitude", padding_value=3.0), ModelSettings("qsvm")),
    "Quantum AdaBoost B": (
        InputEncoding("amplitude", padding_value=3.0),
        ModelSettings("quantum_adaboost", n_estimators=3, max_iterations=SMALL_BUDGET, learning_rate=0.5),
    ),
}


def _imbalanced_table(seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """31 features, 60 fraudes em 600 linhas, como tabela do pandas; as fraudes ficam deslocadas em 4 features."""
    rng = np.random.default_rng(seed)
    labels = np.array([1] * N_FRAUDS + [0] * (N_ROWS - N_FRAUDS))
    features = rng.normal(size=(N_ROWS, N_FEATURES))
    features[:N_FRAUDS, :4] += 1.5
    columns = [f"x{index}" for index in range(N_FEATURES)]
    return pd.DataFrame(features, columns=columns), pd.Series(labels, name="Class")


TRAIN = _imbalanced_table(RNG_SEED)
VALIDATION = _imbalanced_table(RNG_SEED + 1)
TEST = _imbalanced_table(RNG_SEED + 2)


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_scoring_a_fitted_pipeline_equals_fit_and_score(name: str) -> None:
    reference_pipeline = build_pipeline(*CONFIGURATIONS[name], random_state=7)
    reference = fit_and_score(reference_pipeline, *TRAIN, VALIDATION[0])
    fitted = build_pipeline(*CONFIGURATIONS[name], random_state=7).fit(*TRAIN)
    rescored = score_fitted_pipeline(fitted, VALIDATION[0])
    assert np.array_equal(rescored.scores, reference.scores)
    assert np.array_equal(rescored.predictions, reference.predictions)
    # Pontuar não reajusta: pontuar de novo com o mesmo pipeline dá o mesmo resultado
    assert np.array_equal(score_fitted_pipeline(fitted, VALIDATION[0]).scores, rescored.scores)


def test_robustness_point_chooses_the_threshold_on_validation_and_measures_on_test() -> None:
    input_encoding, model = CONFIGURATIONS["SVM"]
    point = robustness_point(input_encoding, model, 7, *TRAIN, *VALIDATION, *TEST)
    fitted = build_pipeline(input_encoding, model, 7).fit(*TRAIN)
    validation_scores = score_fitted_pipeline(fitted, VALIDATION[0]).scores
    test_scores = score_fitted_pipeline(fitted, TEST[0]).scores
    choice = best_f1_threshold(VALIDATION[1].to_numpy(), validation_scores)
    test_alerts = predictions_at_threshold(test_scores, choice.threshold) == 1
    assert point["threshold"] == choice.threshold
    assert point["validation_f1"] == choice.f1
    assert point["test_alerts"] == int(test_alerts.sum())
    assert point["test_true_positives"] == int((test_alerts & (TEST[1].to_numpy() == 1)).sum())
    assert np.isnan(point["kept_estimators"])


def test_robustness_point_counts_the_kept_weak_learners() -> None:
    point = robustness_point(*CONFIGURATIONS["AdaBoost"], 3, *TRAIN, *VALIDATION, *TEST)
    assert 1 <= point["kept_estimators"] <= 5


def test_robustness_has_one_row_per_pipeline_and_seed_and_does_not_depend_on_parallelism() -> None:
    configurations = {name: CONFIGURATIONS[name] for name in ("SVM", "Quantum AdaBoost B")}
    seeds = (7, 11)
    sequential = robustness_scores(configurations, seeds, *TRAIN, *VALIDATION, *TEST, n_jobs=1)
    parallel = robustness_scores(configurations, seeds, *TRAIN, *VALIDATION, *TEST, n_jobs=2)
    assert list(sequential.columns) == list(ROBUSTNESS_COLUMNS)
    assert len(sequential) == len(configurations) * len(seeds)
    assert set(zip(sequential["pipeline"], sequential["seed"])) == {(name, seed) for name in configurations for seed in seeds}
    timing = ["fit_seconds", "score_seconds"]
    pd.testing.assert_frame_equal(sequential.drop(columns=timing), parallel.drop(columns=timing))


def test_seeds_change_the_model() -> None:
    first = robustness_point(*CONFIGURATIONS["SVM"], 7, *TRAIN, *VALIDATION, *TEST)
    second = robustness_point(*CONFIGURATIONS["SVM"], 11, *TRAIN, *VALIDATION, *TEST)
    assert first["validation_pr_auc"] != second["validation_pr_auc"]


def test_final_evaluation_modules_stay_out_of_the_grid_cache_key() -> None:
    names = {path.name for path in package_dependencies("fraud_qml.tuning")}
    assert {"final_evaluation.py", "significance.py"}.isdisjoint(names)
