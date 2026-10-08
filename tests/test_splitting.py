"""Divisão treino/validação/teste e suas verificações."""

import numpy as np
import pandas as pd
import pytest

from fraud_qml.splitting import (
    TEST_NAME,
    TRAIN_NAME,
    VALIDATION_NAME,
    DatasetSplit,
    SplitError,
    find_cross_split_groups,
    find_split_violations,
    stratified_three_way_split,
    validate_split,
)

RNG_SEED = 0
N_ROWS = 10_000
N_POSITIVES = 50  # 0,5%: raro como a fraude, mas com contagem suficiente para as frações


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    labels = np.zeros(N_ROWS, dtype=int)
    labels[rng.choice(N_ROWS, N_POSITIVES, replace=False)] = 1
    # Índice que não começa em 0 nem é contíguo, como o da base deduplicada
    index = pd.Index(np.arange(N_ROWS) * 3 + 7)
    return pd.DataFrame({"feature": rng.normal(size=N_ROWS), "Class": labels}, index=index)


@pytest.fixture
def split(frame: pd.DataFrame) -> DatasetSplit:
    return stratified_three_way_split(frame, random_state=RNG_SEED)


def test_split_is_valid(frame: pd.DataFrame, split: DatasetSplit) -> None:
    assert find_split_violations(frame, split) == []
    validate_split(frame, split)


def test_parts_have_the_requested_fractions(split: DatasetSplit) -> None:
    sizes = {name: len(part) for name, part in split.parts().items()}
    assert sizes == {TRAIN_NAME: 6_000, VALIDATION_NAME: 2_000, TEST_NAME: 2_000}


def test_rare_class_is_stratified_in_every_part(split: DatasetSplit) -> None:
    # 50 positivos em 60/20/20 = 30/10/10: sem estratificação, a validação poderia ter 5 ou 15
    positives = [int(part["Class"].sum()) for part in split.parts().values()]
    assert positives == [30, 10, 10]


def test_original_index_is_preserved(frame: pd.DataFrame, split: DatasetSplit) -> None:
    # O índice identifica a transação no arquivo bruto; resetá-lo quebraria o rastreio
    assert set(split.train.index) <= set(frame.index)
    pd.testing.assert_frame_equal(split.train, frame.loc[split.train.index])


def test_same_seed_gives_same_split_and_other_seed_does_not(frame: pd.DataFrame, split: DatasetSplit) -> None:
    again = stratified_three_way_split(frame, random_state=RNG_SEED)
    other = stratified_three_way_split(frame, random_state=RNG_SEED + 1)
    assert again.fingerprint() == split.fingerprint()
    assert other.fingerprint() != split.fingerprint()


def test_membership_names_each_row_once(frame: pd.DataFrame, split: DatasetSplit) -> None:
    membership = split.membership()
    assert membership.index.sort_values().equals(frame.index.sort_values())
    assert membership.value_counts().to_dict() == {TRAIN_NAME: 6_000, VALIDATION_NAME: 2_000, TEST_NAME: 2_000}


def test_fractions_must_sum_to_one(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="somam"):
        stratified_three_way_split(frame, train_size=0.7, validation_size=0.2, test_size=0.2)


def test_overlap_between_parts_is_detected(frame: pd.DataFrame, split: DatasetSplit) -> None:
    leaked = DatasetSplit(
        train=split.train, validation=split.validation, test=pd.concat([split.test, split.train.iloc[:5]])
    )
    violations = find_split_violations(frame, leaked)
    assert any("5 linhas em 'train' e 'test'" in violation for violation in violations)
    with pytest.raises(SplitError):
        validate_split(frame, leaked)


def test_missing_rows_are_detected(frame: pd.DataFrame, split: DatasetSplit) -> None:
    incomplete = DatasetSplit(train=split.train.iloc[10:], validation=split.validation, test=split.test)
    violations = find_split_violations(frame, incomplete)
    assert any("10 linhas ausentes" in violation for violation in violations)


def test_unstratified_part_is_detected(frame: pd.DataFrame, split: DatasetSplit) -> None:
    # Troca 5 positivos da validação por 5 negativos do treino: tamanhos iguais, proporção errada
    moved_positives = split.validation[split.validation["Class"] == 1].iloc[:5]
    moved_negatives = split.train[split.train["Class"] == 0].iloc[:5]
    unbalanced = DatasetSplit(
        train=pd.concat([split.train.drop(moved_negatives.index), moved_positives]),
        validation=pd.concat([split.validation.drop(moved_positives.index), moved_negatives]),
        test=split.test,
    )
    violations = find_split_violations(frame, unbalanced)
    assert any("'validation' tem 5 linhas da classe 1" in violation for violation in violations)


def test_cross_split_groups_counts_near_duplicates() -> None:
    # Duas cópias da mesma transação (mesma chave) em partes diferentes, e duas na mesma parte
    frame = pd.DataFrame(
        {"key": [1.0, 1.0, 2.0, 2.0, 3.0], "hour": [1, 9, 2, 5, 4], "Class": [0, 0, 1, 1, 0]},
        index=[10, 11, 12, 13, 14],
    )
    split = DatasetSplit(train=frame.loc[[10, 12, 13]], validation=frame.loc[[11]], test=frame.loc[[14]])
    crossing = find_cross_split_groups(split, key_columns=["key"])
    assert (crossing.groups, crossing.rows, crossing.rows_by_label) == (1, 2, {0: 2})
    assert crossing.rows_by_split == {TRAIN_NAME: 1, VALIDATION_NAME: 1, TEST_NAME: 0}


@pytest.mark.parametrize("n_rows", [997, 1_003, 1_009, 1_013])
@pytest.mark.parametrize("n_positives", [7, 11, 19, 23])
def test_tolerance_covers_rounding_in_both_stages(n_rows: int, n_positives: int) -> None:
    # Tamanhos que não dividem exatamente em 60/20/20 forçam arredondamento nos dois estágios;
    # a tolerância de < 2 linhas por classe precisa aceitar todos os splits legítimos
    labels = np.zeros(n_rows, dtype=int)
    labels[:n_positives] = 1
    frame = pd.DataFrame({"feature": np.arange(n_rows, dtype=float), "Class": labels})
    for seed in range(5):
        assert find_split_violations(frame, stratified_three_way_split(frame, random_state=seed)) == []
