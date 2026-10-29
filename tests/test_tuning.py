"""Busca de hiperparâmetros da Seção 6: o grid em dois níveis equivale ao pipeline inteiro."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import average_precision_score

from fraud_qml import config
from fraud_qml.models import InputEncoding, ModelSettings, build_pipeline
from fraud_qml.tuning import (
    CONFIGURATION_COLUMNS,
    SearchSpace,
    amplitude_encoding_grid,
    angle_encoding_grid,
    angle_input_options,
    cached_grid_search,
    classical_input_grid,
    configuration_from_row,
    cv_folds,
    evaluate_input_on_fold,
    project_grid_seconds,
    project_search_spaces,
    run_grid_search,
    summarize_grid,
    write_grid_report,
)

RNG_SEED = 0
N_FEATURES = 31
N_FRAUDS = 60
N_ROWS = 600
N_SPLITS = 3
SMALL_BUDGET = 15


def _imbalanced_data() -> tuple[np.ndarray, np.ndarray]:
    """31 features, 60 fraudes em 600 linhas; as fraudes ficam deslocadas em 4 features."""
    rng = np.random.default_rng(RNG_SEED)
    labels = np.array([1] * N_FRAUDS + [0] * (N_ROWS - N_FRAUDS))
    features = rng.normal(size=(N_ROWS, N_FEATURES))
    features[:N_FRAUDS, :4] += 1.5
    return features, labels


def _small_spaces() -> dict[str, SearchSpace]:
    angle = InputEncoding("angle", range_width=1.0, feature_map="ry")
    entangled = InputEncoding("angle", reducer="mutual_info", range_width=0.25, feature_map="zz", entanglement="full")
    amplitude = InputEncoding("amplitude", padding_value=3.0)
    return {
        "svm": SearchSpace(
            "svm", (InputEncoding("all_features"), InputEncoding("angle_components", reducer="mutual_info")),
            (ModelSettings("svm", C=1.0), ModelSettings("svm", C=10.0, gamma=0.1)),
        ),
        "qsvm": SearchSpace(
            "qsvm", (angle, entangled, amplitude), (ModelSettings("qsvm", C=1.0), ModelSettings("qsvm", C=10.0))
        ),
        "adaboost": SearchSpace(
            "adaboost", (InputEncoding("all_features"),),
            (ModelSettings("adaboost", n_estimators=8, learning_rate=0.5),), boosting_stages=(2, 5, 8),
        ),
        "quantum_adaboost": SearchSpace(
            "quantum_adaboost", (angle, amplitude),
            (ModelSettings("quantum_adaboost", n_estimators=4, max_iterations=SMALL_BUDGET),), boosting_stages=(1, 4),
        ),
    }


@pytest.mark.parametrize(
    ("name", "input_position"),
    [(name, position) for name, space in _small_spaces().items() for position in range(len(space.input_encodings))],
)
def test_two_level_search_scores_like_the_full_pipeline(name: str, input_position: int) -> None:
    # Reaproveitar a entrada entre os modelos não pode mudar nada: cada linha do grid tem de ser a PR-AUC que
    # o pipeline completo daquela combinação teria no fold (em todas as entradas: informação mútua, ZZ, amplitude...)
    features, labels = _imbalanced_data()
    space = _small_spaces()[name]
    train, validation = cv_folds(labels, N_SPLITS, config.RANDOM_STATE)[0]
    input_encoding = space.input_encodings[input_position]
    rows, seconds = evaluate_input_on_fold(space, input_encoding, 0, train, validation, features, labels, config.RANDOM_STATE)
    assert seconds > 0
    for row in rows:
        input_encoding, model = configuration_from_row(row)
        pipeline = build_pipeline(input_encoding, model, config.RANDOM_STATE).fit(features[train], labels[train])
        scores = pipeline.decision_function(features[validation])
        expected = average_precision_score(labels[validation], scores)
        assert row["pr_auc"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("weak_learner", ["stump", "variational"])
def test_reading_a_round_equals_fitting_with_that_ceiling(weak_learner: str) -> None:
    features, labels = _imbalanced_data()
    if weak_learner == "stump":
        input_encoding, ceiling = InputEncoding("all_features"), 12
        model = ModelSettings("adaboost", n_estimators=ceiling, learning_rate=0.5)
    else:
        input_encoding, ceiling = InputEncoding("amplitude", padding_value=3.0), 4
        model = ModelSettings("quantum_adaboost", n_estimators=ceiling, max_iterations=SMALL_BUDGET)
    stages = (1, ceiling // 2, ceiling)
    space = SearchSpace("boosting", (input_encoding,), (model,), boosting_stages=stages)
    train, validation = cv_folds(labels, N_SPLITS, config.RANDOM_STATE)[0]
    rows, _ = evaluate_input_on_fold(space, input_encoding, 0, train, validation, features, labels, config.RANDOM_STATE)
    for row in rows:
        refit = build_pipeline(input_encoding, replace(model, n_estimators=int(row["n_estimators"])), config.RANDOM_STATE)
        refit.fit(features[train], labels[train])
        assert row["pr_auc"] == pytest.approx(average_precision_score(labels[validation], refit.decision_function(features[validation])), abs=1e-12)
        assert row["kept_estimators"] == len(refit.named_steps["model"].estimators_)


def test_a_booster_worse_than_chance_is_recorded_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    # O scikit-learn não ajusta nada quando o primeiro classificador fraco já é pior que o acaso; o grid
    # registra a combinação como falha em vez de interromper a busca
    def failing_fit(self: AdaBoostClassifier, *args: object, **kwargs: object) -> None:
        raise ValueError("BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble can not be fit.")

    monkeypatch.setattr(AdaBoostClassifier, "fit", failing_fit)
    features, labels = _imbalanced_data()
    space = _small_spaces()["adaboost"]
    train, validation = cv_folds(labels, N_SPLITS, config.RANDOM_STATE)[0]
    rows, _ = evaluate_input_on_fold(space, space.input_encodings[0], 0, train, validation, features, labels, config.RANDOM_STATE)
    assert len(rows) == len(space.boosting_stages)
    assert all(row["failed"] and math.isnan(row["pr_auc"]) and row["kept_estimators"] == 0 for row in rows)


def test_other_fit_errors_are_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_fit(self: AdaBoostClassifier, *args: object, **kwargs: object) -> None:
        raise ValueError("outro erro")

    monkeypatch.setattr(AdaBoostClassifier, "fit", broken_fit)
    features, labels = _imbalanced_data()
    space = _small_spaces()["adaboost"]
    train, validation = cv_folds(labels, N_SPLITS, config.RANDOM_STATE)[0]
    with pytest.raises(ValueError, match="outro erro"):
        evaluate_input_on_fold(space, space.input_encodings[0], 0, train, validation, features, labels, config.RANDOM_STATE)


def test_project_grids_have_the_planned_sizes() -> None:
    # PLAN, decisão de arquitetura 4: 72 codificações de ângulo no QSVM, 84 no Quantum AdaBoost (com a RY) e 6 de
    # amplitude; o caminho com informação mútua entra com um scaler só (CC8)
    scalers = ("standard", "robust")
    reducers = ("pca", "mutual_info")
    assert angle_input_options(scalers, reducers) == (("standard", "pca"), ("robust", "pca"), ("standard", "mutual_info"))
    common = (scalers, reducers, config.ANGLE_RANGE_WIDTH_GRID)
    qsvm = angle_encoding_grid(*common, ("z", "zz"), config.FEATURE_MAP_REPS_GRID, config.ENTANGLEMENT_GRID)
    quantum_adaboost = angle_encoding_grid(*common, ("z", "zz", "ry"), config.FEATURE_MAP_REPS_GRID, config.ENTANGLEMENT_GRID)
    assert len(qsvm) == 72 and len(set(qsvm)) == 72
    assert len(quantum_adaboost) == 84 and len(set(quantum_adaboost)) == 84
    assert sum(option.feature_map == "ry" for option in quantum_adaboost) == 3 * len(config.ANGLE_RANGE_WIDTH_GRID)
    assert len(amplitude_encoding_grid(scalers, config.AMPLITUDE_PADDING_GRID)) == 6
    classical = classical_input_grid(scalers, scalers, reducers)
    assert [option.kind for option in classical].count("all_features") == 2 and len(classical) == 5


def test_the_four_project_grids_follow_the_plan() -> None:
    spaces = project_search_spaces()
    assert list(spaces) == ["svm", "adaboost", "qsvm", "quantum_adaboost"]
    n_c, n_gamma = len(config.SVM_C_GRID), len(config.SVM_GAMMA_GRID)
    n_rates, n_stages = len(config.BOOSTING_LEARNING_RATE_GRID), len(config.BOOSTING_N_ESTIMATORS_GRID)
    assert spaces["svm"].n_combinations == 5 * n_c * n_gamma
    assert spaces["adaboost"].n_combinations == 4 * n_rates * n_stages
    assert spaces["qsvm"].n_combinations == (72 + 6) * n_c
    assert spaces["quantum_adaboost"].n_combinations == (84 + 6) * n_rates * len(config.ANSATZ_REPS_GRID) * n_stages
    # Os dois lados de cada par: entradas quânticas só nos quânticos, a RY só no Quantum AdaBoost
    assert not any(option.is_quantum for option in spaces["svm"].input_encodings + spaces["adaboost"].input_encodings)
    assert all(option.is_quantum for option in spaces["qsvm"].input_encodings + spaces["quantum_adaboost"].input_encodings)
    assert not any(option.feature_map == "ry" for option in spaces["qsvm"].input_encodings if option.kind == "angle")
    # Os dois AdaBoost têm o mesmo grid de modelo; só o classificador fraco (e as repetições do ansatz) muda
    assert {model.learning_rate for model in spaces["adaboost"].models} == {model.learning_rate for model in spaces["quantum_adaboost"].models}
    assert spaces["adaboost"].boosting_stages == spaces["quantum_adaboost"].boosting_stages
    # As 31 features entram sempre inteiras: a amplitude não tem opção de conjunto de features (ponto ③ do PLAN)
    assert {option.kind for option in spaces["qsvm"].input_encodings} == {"angle", "amplitude"}
    assert project_search_spaces() == spaces


def test_search_space_rejects_inconsistent_boosting_settings() -> None:
    adaboost = ModelSettings("adaboost", n_estimators=10)
    with pytest.raises(ValueError, match="tetos"):
        SearchSpace("x", (InputEncoding("all_features"),), (adaboost,))
    with pytest.raises(ValueError, match="maior teto"):
        SearchSpace("x", (InputEncoding("all_features"),), (adaboost,), boosting_stages=(5, 20))
    with pytest.raises(ValueError, match="rodadas"):
        SearchSpace("x", (InputEncoding("all_features"),), (ModelSettings("svm"),), boosting_stages=(5,))
    with pytest.raises(ValueError, match="mistura"):
        SearchSpace("x", (InputEncoding("all_features"),), (adaboost, ModelSettings("svm")), boosting_stages=(10,))
    assert SearchSpace("x", (InputEncoding("all_features"),), (adaboost,), boosting_stages=(5, 10)).n_combinations == 2


def test_summary_averages_folds_and_puts_failures_last() -> None:
    features, labels = _imbalanced_data()
    result = run_grid_search(_small_spaces()["svm"], features, labels, n_splits=N_SPLITS, n_jobs=1)
    assert len(result.fold_scores) == 2 * 2 * N_SPLITS
    assert len(result.task_seconds) == 2 * N_SPLITS
    failed = result.fold_scores.iloc[:1].assign(pr_auc=np.nan, failed=True, kind="amplitude")
    summary = summarize_grid(pd.concat([result.fold_scores, failed], ignore_index=True))
    assert list(summary["rank"]) == list(range(1, len(summary) + 1))
    assert summary["pr_auc_mean"].iloc[:-1].is_monotonic_decreasing
    assert np.isnan(summary["pr_auc_mean"].iloc[-1]) and summary["failed_folds"].iloc[-1] == 1
    first = summary.iloc[0]
    mask = np.ones(len(result.fold_scores), dtype=bool)
    for column in CONFIGURATION_COLUMNS:
        value = first[column]
        column_values = result.fold_scores[column]
        mask &= column_values.isna().to_numpy() if pd.isna(value) else (column_values == value).to_numpy()
    assert first["pr_auc_mean"] == pytest.approx(result.fold_scores.loc[mask, "pr_auc"].mean())


def test_configuration_round_trips_through_a_result_row() -> None:
    features, labels = _imbalanced_data()
    space = _small_spaces()["quantum_adaboost"]
    summary = summarize_grid(run_grid_search(space, features, labels, n_splits=N_SPLITS, n_jobs=1).fold_scores)
    for _, row in summary.iterrows():
        input_encoding, model = configuration_from_row(row)
        assert input_encoding in space.input_encodings
        assert model.n_estimators in space.boosting_stages
        assert replace(model, n_estimators=space.models[0].n_estimators) == space.models[0]


def test_cache_returns_the_same_result_and_depends_on_the_grid(tmp_path: Path) -> None:
    features, labels = _imbalanced_data()
    space = _small_spaces()["svm"]
    first = cached_grid_search(space, features, labels, cache_dir=tmp_path, n_splits=N_SPLITS, n_jobs=1)
    second = cached_grid_search(space, features, labels, cache_dir=tmp_path, n_splits=N_SPLITS, n_jobs=1)
    assert not first.from_cache and second.from_cache
    pd.testing.assert_frame_equal(first.fold_scores, second.fold_scores)
    other = replace(space, models=space.models[:1])
    assert not cached_grid_search(other, features, labels, cache_dir=tmp_path, n_splits=N_SPLITS, n_jobs=1).from_cache


def test_projection_assumes_every_task_as_slow_as_the_slowest() -> None:
    assert project_grid_seconds([1.0, 3.0, 2.0], n_tasks=25, n_jobs=12) == 3.0 * 3


def test_report_is_valid_json_with_the_best_combinations(tmp_path: Path) -> None:
    features, labels = _imbalanced_data()
    result = run_grid_search(_small_spaces()["svm"], features, labels, n_splits=N_SPLITS, n_jobs=1)
    target = tmp_path / "grid_svm.json"
    write_grid_report(target, result, summarize_grid(result.fold_scores), {"commit": "abc"}, top=2)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["model"] == "svm" and payload["commit"] == "abc" and len(payload["best"]) == 2
    assert payload["best"][0]["rank"] == 1
