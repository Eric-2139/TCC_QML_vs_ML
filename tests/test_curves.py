"""Curvas de aprendizado e de validação da Seção 8: nota no treino ponderada e coerência com a busca da Seção 6."""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_qml import config
from fraud_qml.curves import (
    CEILING_PARAMETER,
    CURVE_STAGES,
    CURVE_VALUES,
    LEARNING_CURVE_COLUMNS,
    curve_parameters,
    curve_points,
    learning_curve_point,
    learning_curve_scores,
    nested_fraction_rows,
    prevalence_weights,
    training_metrics,
    validation_curve_scores,
    validation_curve_table,
)
from fraud_qml.evaluation import weighted_average_precision
from fraud_qml.models import InputEncoding, ModelSettings, build_pipeline
from fraud_qml.tuning import SearchSpace, cv_folds, evaluate_input_on_fold

RNG_SEED = 0
N_FEATURES = 31
N_FRAUDS = 60
N_ROWS = 600
N_SPLITS = 3
SMALL_BUDGET = 15
SMALL_CEILING = 6


def _imbalanced_data(seed: int = RNG_SEED) -> tuple[np.ndarray, np.ndarray]:
    """31 features, 60 fraudes em 600 linhas; as fraudes ficam deslocadas em 4 features."""
    rng = np.random.default_rng(seed)
    labels = np.array([1] * N_FRAUDS + [0] * (N_ROWS - N_FRAUDS))
    features = rng.normal(size=(N_ROWS, N_FEATURES))
    features[:N_FRAUDS, :4] += 1.5
    return features, labels


def test_prevalence_weights_restore_the_pool_proportion() -> None:
    pool = np.array([1] * 10 + [0] * 990)
    sampled = np.array([1] * 10 + [0] * 10)
    weights = prevalence_weights(sampled, pool)
    assert np.all(weights[sampled == 1] == 1.0)
    assert np.all(weights[sampled == 0] == 99.0)
    assert np.sum(weights[sampled == 1]) / np.sum(weights) == pytest.approx(pool.mean())


def test_prevalence_weights_reject_a_class_missing_from_the_pool() -> None:
    with pytest.raises(ValueError, match="não existe"):
        prevalence_weights(np.array([0, 1]), np.array([0, 0, 0]))


def test_training_metrics_use_the_whole_fit_set_and_the_seen_rows() -> None:
    rng = np.random.default_rng(RNG_SEED)
    pool_labels = np.array([1] * 20 + [0] * 380)
    pool_scores = rng.normal(size=400) + 2 * pool_labels
    seen = np.concatenate([np.arange(20), np.arange(20, 40)])
    metrics = training_metrics(pool_labels[seen], pool_scores[seen], pool_labels, pool_scores)
    assert metrics["train_pr_auc"] == pytest.approx(average_precision_score(pool_labels, pool_scores), abs=1e-12)
    assert metrics["train_roc_auc"] == pytest.approx(roc_auc_score(pool_labels[seen], pool_scores[seen]), abs=1e-12)


def test_weighted_pr_auc_of_the_seen_rows_jumps_with_a_single_legitimate_row() -> None:
    # Por que a PR-AUC ponderada nas linhas vistas não serve de nota de treino: com cada legítima valendo 598, uma só
    # legítima acima das fraudes derruba a nota, embora a ordem das outras 567 linhas seja perfeita
    labels = np.repeat([1, 0], 284)
    pool = np.concatenate([np.ones(284, dtype=int), np.zeros(169_951, dtype=int)])
    perfect = np.where(labels == 1, 1.0, 0.0)
    one_on_top = perfect.copy()
    one_on_top[284] = 2.0
    weights = prevalence_weights(labels, pool)
    assert weighted_average_precision(labels, perfect, weights) == pytest.approx(1.0)
    assert weighted_average_precision(labels, one_on_top, weights) < 0.4
    assert roc_auc_score(labels, one_on_top) > 0.99


def test_nested_fractions_are_stratified_and_nested() -> None:
    _, labels = _imbalanced_data()
    smaller = nested_fraction_rows(labels, 0.25, RNG_SEED)
    larger = nested_fraction_rows(labels, 0.8, RNG_SEED)
    assert int(labels[smaller].sum()) == round(0.25 * N_FRAUDS)
    assert int((labels[larger] == 0).sum()) == round(0.8 * (N_ROWS - N_FRAUDS))
    assert set(smaller) <= set(larger)
    assert np.all(np.diff(larger) > 0)


def test_whole_fraction_keeps_every_row_in_order() -> None:
    _, labels = _imbalanced_data()
    np.testing.assert_array_equal(nested_fraction_rows(labels, 1.0, RNG_SEED), np.arange(N_ROWS))


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.2])
def test_nested_fraction_rejects_values_outside_the_unit_interval(fraction: float) -> None:
    with pytest.raises(ValueError, match="fração"):
        nested_fraction_rows(np.array([0, 1]), fraction, RNG_SEED)


def test_whole_training_point_is_the_pipeline_fitted_on_everything() -> None:
    # O ponto de 100% com a semente do experimento precisa ser o modelo da Seção 7: o mesmo pipeline, nas mesmas linhas
    features, labels = _imbalanced_data()
    evaluation_features, evaluation_labels = _imbalanced_data(seed=1)
    input_encoding, model = InputEncoding("angle_components", reducer="mutual_info"), ModelSettings("svm", C=10.0, gamma=0.1)
    point = learning_curve_point(input_encoding, model, 1.0, config.RANDOM_STATE, features, labels, evaluation_features, evaluation_labels)
    pipeline = build_pipeline(input_encoding, model, config.RANDOM_STATE).fit(features, labels)
    expected = average_precision_score(evaluation_labels, pipeline.decision_function(evaluation_features))
    assert point["pr_auc"] == pytest.approx(expected, abs=1e-12)
    assert point["frauds_seen"] == N_FRAUDS
    assert point["rows_seen"] == 2 * N_FRAUDS
    assert 0.0 <= point["train_pr_auc"] <= 1.0


def test_learning_curve_has_one_row_per_pipeline_fraction_and_seed() -> None:
    features, labels = _imbalanced_data()
    evaluation_features, evaluation_labels = _imbalanced_data(seed=1)
    configurations = {
        "SVM": (InputEncoding("all_features"), ModelSettings("svm")),
        "AdaBoost": (InputEncoding("all_features"), ModelSettings("adaboost", n_estimators=5)),
    }
    scores = learning_curve_scores(configurations, (0.5, 1.0), (0, 1), features, labels, evaluation_features, evaluation_labels, n_jobs=2)
    assert list(scores.columns) == list(LEARNING_CURVE_COLUMNS)
    assert len(scores) == 2 * 2 * 2
    assert set(scores.loc[scores["fraction"] == 0.5, "frauds_seen"]) == {N_FRAUDS // 2}
    assert scores["kept_estimators"].isna().sum() == 4


@pytest.mark.parametrize(
    ("input_encoding", "model", "expected"),
    [
        (InputEncoding("all_features"), ModelSettings("svm"), ("C", "gamma")),
        (InputEncoding("angle"), ModelSettings("qsvm"), ("C", "range_width")),
        (InputEncoding("amplitude"), ModelSettings("qsvm"), ("C", "padding_value")),
        (InputEncoding("all_features"), ModelSettings("adaboost"), ("learning_rate", CEILING_PARAMETER)),
        (InputEncoding("amplitude"), ModelSettings("quantum_adaboost"), ("learning_rate", CEILING_PARAMETER, "padding_value")),
    ],
)
def test_each_family_gets_its_numeric_hyperparameters(input_encoding: InputEncoding, model: ModelSettings, expected: tuple[str, ...]) -> None:
    assert curve_parameters(input_encoding, model) == expected


def test_curve_points_pass_through_the_chosen_configuration() -> None:
    input_encoding = InputEncoding("angle", range_width=0.25)
    model = ModelSettings("quantum_adaboost", n_estimators=100, learning_rate=0.1)
    points = curve_points(input_encoding, model)
    for parameter in ("learning_rate", "range_width"):
        curve = [point for point in points if point.parameter == parameter]
        assert [point.value for point in curve] == list(CURVE_VALUES[parameter])
        assert all(point.model.n_estimators == 100 and point.stages == () for point in curve)
        assert any(point.input_encoding == input_encoding and point.model == model for point in curve)
    (ceiling,) = [point for point in points if point.parameter == CEILING_PARAMETER]
    assert ceiling.model == replace(model, n_estimators=200)
    assert ceiling.stages == CURVE_STAGES


def test_curve_points_reject_a_chosen_value_outside_the_curve() -> None:
    with pytest.raises(ValueError, match="não está na curva"):
        curve_points(InputEncoding("all_features"), ModelSettings("svm", C=3.0, gamma=0.01))


def test_curve_points_reject_a_ceiling_that_is_not_read() -> None:
    with pytest.raises(ValueError, match="rodadas lidas"):
        curve_points(InputEncoding("all_features"), ModelSettings("adaboost", n_estimators=33))


def test_curve_values_include_the_steps_beyond_the_remaining_borders() -> None:
    # As três bordas que sobraram na Seção 6.5 (CC19): γ = 0,001, taxa 0,05 e constante 10, um passo além de cada
    assert min(CURVE_VALUES["gamma"]) < 0.001
    assert min(CURVE_VALUES["learning_rate"]) < 0.05
    assert max(CURVE_VALUES["padding_value"]) > 10.0
    assert max(CURVE_VALUES["range_width"]) == pytest.approx(math.pi / 2)
    assert {10, 25, 50, 100, 200} <= set(CURVE_STAGES)


@pytest.mark.parametrize(
    ("input_encoding", "model"),
    [
        (InputEncoding("angle_components", reducer="mutual_info"), ModelSettings("svm", C=10.0, gamma=0.1)),
        (InputEncoding("all_features"), ModelSettings("adaboost", n_estimators=SMALL_CEILING, learning_rate=0.5)),
        (InputEncoding("amplitude", padding_value=3.0), ModelSettings("quantum_adaboost", n_estimators=4, max_iterations=SMALL_BUDGET, learning_rate=0.5)),
    ],
)
def test_validation_curve_scores_match_the_search_of_section_six(input_encoding: InputEncoding, model: ModelSettings) -> None:
    # Mesmos folds, mesmo sorteio e mesmas operações: a nota de avaliação de cada ponto é a da busca, inclusive as
    # rodadas lidas do ajuste longo do teto
    features, labels = _imbalanced_data()
    stages = (1, 2, model.n_estimators) if model.is_boosting else ()
    points = curve_points(input_encoding, model, max_rounds=model.n_estimators, stages=stages)
    chosen = [point for point in points if point.input_encoding == input_encoding and point.model.n_estimators == model.n_estimators
              and (point.stages or not model.is_boosting)][:1]
    curve = validation_curve_scores(chosen, features, labels, n_splits=N_SPLITS, n_jobs=1)
    space = SearchSpace("conferência", (input_encoding,), (chosen[0].model,), boosting_stages=stages)
    train, evaluation = cv_folds(labels, N_SPLITS, config.RANDOM_STATE)[0]
    grid_rows, _ = evaluate_input_on_fold(space, input_encoding, 0, train, evaluation, features, labels, config.RANDOM_STATE)
    fold_zero = curve[curve["fold"] == 0].reset_index(drop=True)
    assert len(fold_zero) == len(grid_rows)
    for grid_row, (_, row) in zip(grid_rows, fold_zero.iterrows(), strict=True):
        assert row["pr_auc"] == pytest.approx(grid_row["pr_auc"], abs=1e-12)
        assert 0.0 <= row["train_pr_auc"] <= 1.0 and 0.0 <= row["train_roc_auc"] <= 1.0


def test_validation_curve_table_reads_one_value_per_fold() -> None:
    features, labels = _imbalanced_data()
    input_encoding, model = InputEncoding("all_features"), ModelSettings("adaboost", n_estimators=4, learning_rate=0.5)
    points = curve_points(input_encoding, model, max_rounds=SMALL_CEILING, stages=(1, 2, 4, SMALL_CEILING))
    scores = validation_curve_scores(points, features, labels, n_splits=N_SPLITS, n_jobs=2)
    rates = validation_curve_table(scores, input_encoding, model, "learning_rate")
    assert list(rates["value"].unique()) == list(CURVE_VALUES["learning_rate"])
    assert rates.groupby("value")["fold"].nunique().eq(N_SPLITS).all()
    ceiling = validation_curve_table(scores, input_encoding, model, CEILING_PARAMETER)
    assert list(ceiling["value"].unique()) == [1, 2, 4, SMALL_CEILING]
    # O teto escolhido aparece nos dois ajustes (com 4 rodadas e lido do de 6): as notas precisam ser iguais
    chosen = scores[(scores["learning_rate"] == 0.5) & (scores["n_estimators"] == 4)]
    assert chosen.groupby("fold")["pr_auc"].nunique().eq(1).all()
    assert chosen.groupby("fold")["train_pr_auc"].nunique().eq(1).all()
    np.testing.assert_allclose(
        rates.loc[rates["value"] == 0.5, "pr_auc"].to_numpy(), ceiling.loc[ceiling["value"] == 4, "pr_auc"].to_numpy()
    )


def test_learning_curve_point_keeps_the_table_type() -> None:
    # Com uma tabela do pandas, o ponto de 100% recebe a mesma tabela que a Seção 7 passou ao pipeline
    features, labels = _imbalanced_data()
    frame, target = pd.DataFrame(features), pd.Series(labels)
    input_encoding, model = InputEncoding("all_features"), ModelSettings("adaboost", n_estimators=5)
    from_frame = learning_curve_point(input_encoding, model, 0.5, RNG_SEED, frame, target, features, labels)
    from_array = learning_curve_point(input_encoding, model, 0.5, RNG_SEED, features, labels, features, labels)
    assert from_frame["frauds_seen"] == from_array["frauds_seen"] == N_FRAUDS // 2
    assert from_frame["pr_auc"] == pytest.approx(from_array["pr_auc"], abs=1e-12)


def test_fraction_rows_keep_the_pandas_table() -> None:
    # O tipo do dado muda o Quantum AdaBoost (CC21): as frações abaixo de 1 também precisam continuar tabelas
    from fraud_qml.curves import _take_rows

    features, labels = _imbalanced_data()
    rows = nested_fraction_rows(labels, 0.5, RNG_SEED)
    taken = _take_rows(pd.DataFrame(features), rows)
    assert isinstance(taken, pd.DataFrame)
    np.testing.assert_array_equal(taken.to_numpy(), features[rows])
    assert isinstance(_take_rows(features, rows), np.ndarray)
