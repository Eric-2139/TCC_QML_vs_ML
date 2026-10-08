"""Duplicatas e engenharia de features."""

import numpy as np
import pandas as pd
import pytest

from fraud_qml.data import EXPECTED_COLUMNS, RAW_FEATURE_COLUMNS, TARGET_COLUMN, remove_exact_duplicates, summarize_duplicates
from fraud_qml.features import (
    AUXILIARY_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    engineer_features,
)


def make_frame(rows: list[tuple[float, int]]) -> pd.DataFrame:
    """Dataset sintético: cada linha é (valor repetido em todas as features, rótulo)."""
    features = np.array([[value] * len(RAW_FEATURE_COLUMNS) for value, _ in rows])
    frame = pd.DataFrame(features, columns=RAW_FEATURE_COLUMNS)
    frame[TARGET_COLUMN] = [label for _, label in rows]
    return frame[list(EXPECTED_COLUMNS)]


def test_duplicate_summary_counts_groups_copies_and_frauds() -> None:
    # 3 legítimas iguais, 2 fraudes iguais e 1 linha única
    frame = make_frame([(1.0, 0), (1.0, 0), (1.0, 0), (2.0, 1), (2.0, 1), (3.0, 0)])
    summary = summarize_duplicates(frame)
    assert summary.rows_in_duplicate_groups == 5
    assert summary.extra_copies == 3
    assert summary.fraud_extra_copies == 1
    assert summary.conflicting_label_groups == 0


def test_duplicate_summary_detects_conflicting_labels() -> None:
    frame = make_frame([(1.0, 0), (1.0, 1), (2.0, 0)])
    summary = summarize_duplicates(frame)
    # Não são duplicatas exatas (o rótulo difere), mas as features são idênticas
    assert summary.extra_copies == 0
    assert summary.conflicting_label_groups == 1


def test_remove_duplicates_keeps_first_occurrence_and_original_index() -> None:
    frame = make_frame([(1.0, 0), (2.0, 1), (1.0, 0)])
    deduplicated = remove_exact_duplicates(frame)
    assert deduplicated.index.tolist() == [0, 1]


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(4, len(RAW_FEATURE_COLUMNS))), columns=RAW_FEATURE_COLUMNS)
    frame["Time"] = [0.0, 13.5 * SECONDS_PER_HOUR, SECONDS_PER_DAY - 1, SECONDS_PER_DAY]
    frame["Amount"] = [0.0, 1.0, 100.0, 25_000.0]
    frame[TARGET_COLUMN] = [0, 1, 0, 1]
    return frame


def test_engineered_columns_and_order(raw_frame: pd.DataFrame) -> None:
    engineered = engineer_features(raw_frame)
    assert tuple(engineered.columns) == (*MODEL_FEATURE_COLUMNS, *AUXILIARY_COLUMNS, TARGET_COLUMN)
    assert len(MODEL_FEATURE_COLUMNS) == 31
    assert "Time" not in MODEL_FEATURE_COLUMNS
    assert not set(AUXILIARY_COLUMNS) & set(MODEL_FEATURE_COLUMNS)


def test_amount_log_handles_zero_and_preserves_order(raw_frame: pd.DataFrame) -> None:
    amount_log = engineer_features(raw_frame)["amount_log"]
    assert amount_log.iloc[0] == 0.0
    assert amount_log.is_monotonic_increasing


def test_hour_encoding_is_on_unit_circle(raw_frame: pd.DataFrame) -> None:
    engineered = engineer_features(raw_frame)
    radius = engineered["hour_sin"] ** 2 + engineered["hour_cos"] ** 2
    np.testing.assert_allclose(radius, 1.0)
    assert engineered["hour_of_day"].iloc[1] == pytest.approx(13.5)


def test_hour_encoding_is_cyclic(raw_frame: pd.DataFrame) -> None:
    engineered = engineer_features(raw_frame)
    point = engineered[["hour_sin", "hour_cos"]].to_numpy()
    # 23h59m59s fica colado à meia-noite no círculo, embora a hora crua difira em ~24
    assert np.linalg.norm(point[2] - point[0]) < 1e-3
    # Um dia inteiro depois volta exatamente ao mesmo ponto
    np.testing.assert_allclose(point[3], point[0], atol=1e-12)


def test_engineering_does_not_mutate_input(raw_frame: pd.DataFrame) -> None:
    snapshot = raw_frame.copy()
    engineer_features(raw_frame)
    pd.testing.assert_frame_equal(raw_frame, snapshot)


def test_engineering_a_part_equals_the_part_of_the_engineered_whole(raw_frame: pd.DataFrame) -> None:
    # Cada linha é transformada só com os próprios valores: aplicar a função a uma parte do split dá
    # exatamente as mesmas linhas que aplicá-la à base inteira e depois separar. É o que permite
    # aplicar a transformação à validação e ao teste sem vazamento
    part = raw_frame.iloc[::2]
    pd.testing.assert_frame_equal(engineer_features(part), engineer_features(raw_frame).loc[part.index])
