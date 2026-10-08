"""Carregamento, integridade e contrato do dataset."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_qml import config
from fraud_qml.data import (
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
    DataContractError,
    csv_sha256,
    find_contract_violations,
    load_raw_dataset,
    validate_raw_contract,
    verify_dataset_integrity,
)

SMALL_N_ROWS = 10
SMALL_N_FRAUDS = 2


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    """Dataset sintético pequeno que cumpre o contrato (com contagens parametrizadas)."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(SMALL_N_ROWS, len(EXPECTED_COLUMNS) - 1)), columns=EXPECTED_COLUMNS[:-1])
    frame[TARGET_COLUMN] = [1] * SMALL_N_FRAUDS + [0] * (SMALL_N_ROWS - SMALL_N_FRAUDS)
    return frame


def small_contract_violations(frame: pd.DataFrame) -> list[str]:
    return find_contract_violations(frame, expected_n_rows=SMALL_N_ROWS, expected_n_frauds=SMALL_N_FRAUDS)


def test_valid_frame_has_no_violations(valid_frame: pd.DataFrame) -> None:
    assert small_contract_violations(valid_frame) == []


def test_detects_reordered_columns(valid_frame: pd.DataFrame) -> None:
    reordered = valid_frame[list(reversed(valid_frame.columns))]
    assert any("colunas" in violation for violation in small_contract_violations(reordered))


def test_detects_missing_values(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, "V1"] = np.nan
    assert any("ausentes em" in violation for violation in small_contract_violations(valid_frame))


def test_detects_unexpected_labels(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, TARGET_COLUMN] = 2
    assert any("rótulos" in violation for violation in small_contract_violations(valid_frame))


def test_reports_all_violations_together(valid_frame: pd.DataFrame) -> None:
    broken = valid_frame.drop(columns="Amount").iloc[:-1]
    # Coluna ausente e contagem de linhas errada devem aparecer na mesma mensagem
    with pytest.raises(DataContractError) as error:
        validate_raw_contract(broken, expected_n_rows=SMALL_N_ROWS, expected_n_frauds=SMALL_N_FRAUDS)
    assert "colunas" in str(error.value)
    assert "linhas" in str(error.value)


def test_hash_ignores_zip_compression(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    csv_path = tmp_path / "data.csv"
    valid_frame.to_csv(csv_path, index=False)
    zip_path = tmp_path / "data.csv.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="data.csv")
    assert csv_sha256(zip_path) == csv_sha256(csv_path)


def test_integrity_check_rejects_other_file(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    csv_path = tmp_path / "data.csv"
    valid_frame.to_csv(csv_path, index=False)
    with pytest.raises(DataContractError):
        verify_dataset_integrity(csv_path)


def test_missing_file_explains_where_to_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_raw_dataset(tmp_path / "inexistente.csv.zip")


@pytest.mark.skipif(not config.DATA_PATH.is_file(), reason="dataset real ausente")
def test_real_dataset_honours_contract_and_hash() -> None:
    verify_dataset_integrity()
    validate_raw_contract(load_raw_dataset())
