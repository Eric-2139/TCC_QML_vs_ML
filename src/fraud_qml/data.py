"""Carregamento e contrato de dados do dataset Credit Card Fraud Detection (ULB / Kaggle).

O contrato é a lista de propriedades que o arquivo precisa ter para que os resultados do
TCC valham: se alguém rodar o projeto com outra versão do dataset (outra contagem de
linhas, colunas em outra ordem, rótulos diferentes), o código deve parar com uma mensagem
clara, em vez de produzir números silenciosamente diferentes.
"""

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from pandas.api.types import is_numeric_dtype

from fraud_qml import config

DATASET_URL: Final[str] = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

# Hash do CSV original do Kaggle (150.828.752 bytes), o mesmo usado no trabalho anterior.
# É calculado sobre o CSV descompactado para não depender de como o .zip foi gerado.
EXPECTED_CSV_SHA256: Final[str] = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"

EXPECTED_N_ROWS: Final[int] = 284_807
EXPECTED_N_FRAUDS: Final[int] = 492

PCA_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(f"V{index}" for index in range(1, 29))
RAW_FEATURE_COLUMNS: Final[tuple[str, ...]] = ("Time", *PCA_FEATURE_COLUMNS, "Amount")
TARGET_COLUMN: Final[str] = "Class"
EXPECTED_COLUMNS: Final[tuple[str, ...]] = (*RAW_FEATURE_COLUMNS, TARGET_COLUMN)

LEGITIMATE_LABEL: Final[int] = 0
FRAUD_LABEL: Final[int] = 1

_HASH_CHUNK_BYTES: Final[int] = 1 << 20


class DataContractError(ValueError):
    """O dataset carregado viola uma ou mais propriedades esperadas."""


def load_raw_dataset(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Lê o dataset (``.csv`` ou ``.csv.zip``) sem nenhuma transformação.

    O pandas descompacta o ``.zip`` em memória; não é preciso extrair o arquivo no disco.

    Args:
        path: Caminho do ``creditcard.csv`` ou ``creditcard.csv.zip``.

    Returns:
        DataFrame bruto, com as colunas na ordem original do arquivo.

    Raises:
        FileNotFoundError: Se o arquivo não existir; a mensagem indica onde baixá-lo.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. Baixe o creditcard.csv.zip em {DATASET_URL} "
            "e coloque-o na raiz do projeto (não é preciso descompactar)."
        )
    return pd.read_csv(path)


def csv_sha256(path: Path) -> str:
    """Calcula o SHA-256 do conteúdo CSV, descompactando o ``.zip`` quando necessário.

    Args:
        path: Arquivo ``.csv`` ou ``.zip`` contendo um único CSV.

    Returns:
        Hash hexadecimal do CSV.

    Raises:
        DataContractError: Se o ``.zip`` não contiver exatamente um arquivo.
    """
    digest = hashlib.sha256()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != 1:
                raise DataContractError(f"O .zip deveria conter 1 CSV, mas contém {len(members)} arquivos.")
            with archive.open(members[0]) as stream:
                for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
                    digest.update(chunk)
    else:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_integrity(path: Path = config.DATA_PATH, expected_sha256: str = EXPECTED_CSV_SHA256) -> str:
    """Confirma que o arquivo é byte a byte o CSV original usado no TCC.

    Args:
        path: Arquivo do dataset.
        expected_sha256: Hash esperado do CSV.

    Returns:
        O hash calculado (para registro nos metadados dos resultados).

    Raises:
        DataContractError: Se o hash divergir.
    """
    actual_sha256 = csv_sha256(path)
    if actual_sha256 != expected_sha256:
        raise DataContractError(
            f"O CSV não é o arquivo original do TCC (SHA-256 {actual_sha256[:12]}…, "
            f"esperado {expected_sha256[:12]}…). Baixe novamente em {DATASET_URL}."
        )
    return actual_sha256


def find_contract_violations(
    frame: pd.DataFrame,
    expected_n_rows: int = EXPECTED_N_ROWS,
    expected_n_frauds: int = EXPECTED_N_FRAUDS,
) -> list[str]:
    """Lista todas as propriedades do contrato que o DataFrame viola.

    Todas as verificações rodam antes de qualquer erro ser levantado: corrigir um problema
    por vez, rodando de novo a cada descoberta, é lento e esconde a causa comum.

    Args:
        frame: Dataset bruto.
        expected_n_rows: Número de linhas esperado (parametrizável para testes).
        expected_n_frauds: Número de fraudes esperado (parametrizável para testes).

    Returns:
        Descrições das violações; lista vazia quando o contrato é cumprido.
    """
    violations: list[str] = []

    if tuple(frame.columns) != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
        violations.append(
            f"colunas diferentes do esperado (ausentes: {missing}; inesperadas: {unexpected}; "
            "ou ordem trocada)"
        )

    if len(frame) != expected_n_rows:
        violations.append(f"{len(frame):,} linhas, esperado {expected_n_rows:,}")

    non_numeric = [column for column in frame.columns if not is_numeric_dtype(frame[column])]
    if non_numeric:
        violations.append(f"colunas não numéricas: {non_numeric}")

    columns_with_missing = frame.columns[frame.isna().any()].tolist()
    if columns_with_missing:
        violations.append(f"valores ausentes em: {columns_with_missing}")

    # As verificações do rótulo só fazem sentido se a coluna existir
    if TARGET_COLUMN in frame.columns:
        labels = set(frame[TARGET_COLUMN].dropna().unique().tolist())
        if labels != {LEGITIMATE_LABEL, FRAUD_LABEL}:
            violations.append(f"rótulos {sorted(labels)} em '{TARGET_COLUMN}', esperado [0, 1]")
        n_frauds = int((frame[TARGET_COLUMN] == FRAUD_LABEL).sum())
        if n_frauds != expected_n_frauds:
            violations.append(f"{n_frauds} fraudes, esperado {expected_n_frauds}")

    return violations


def validate_raw_contract(
    frame: pd.DataFrame,
    expected_n_rows: int = EXPECTED_N_ROWS,
    expected_n_frauds: int = EXPECTED_N_FRAUDS,
) -> None:
    """Levanta ``DataContractError`` com todas as violações, se houver alguma.

    Args:
        frame: Dataset bruto.
        expected_n_rows: Número de linhas esperado.
        expected_n_frauds: Número de fraudes esperado.

    Raises:
        DataContractError: Se o contrato for violado.
    """
    violations = find_contract_violations(frame, expected_n_rows, expected_n_frauds)
    if violations:
        bullet_list = "\n".join(f"  - {violation}" for violation in violations)
        raise DataContractError(f"O dataset viola o contrato:\n{bullet_list}")


@dataclass(frozen=True)
class DuplicateSummary:
    """Diagnóstico de linhas repetidas no dataset.

    Attributes:
        rows_in_duplicate_groups: Linhas que têm pelo menos uma cópia idêntica (contando todas as cópias).
        extra_copies: Linhas que a remoção eliminaria (cada grupo mantém um representante).
        fraud_extra_copies: Quantas dessas cópias extras são fraudes.
        conflicting_label_groups: Grupos com features idênticas, mas rótulos diferentes. A remoção
            de duplicatas exatas não os resolve, porque as linhas diferem justamente no rótulo.
    """

    rows_in_duplicate_groups: int
    extra_copies: int
    fraud_extra_copies: int
    conflicting_label_groups: int


def summarize_duplicates(frame: pd.DataFrame) -> DuplicateSummary:
    """Conta duplicatas exatas e conflitos de rótulo entre linhas com as mesmas features.

    Args:
        frame: Dataset bruto (features originais + rótulo).

    Returns:
        Resumo das duplicatas.
    """
    extra_copy_mask = frame.duplicated(keep="first")
    same_features_mask = frame.duplicated(subset=list(RAW_FEATURE_COLUMNS), keep=False)
    labels_per_feature_group = frame[same_features_mask].groupby(list(RAW_FEATURE_COLUMNS))[TARGET_COLUMN].nunique()

    return DuplicateSummary(
        rows_in_duplicate_groups=int(frame.duplicated(keep=False).sum()),
        extra_copies=int(extra_copy_mask.sum()),
        fraud_extra_copies=int((frame.loc[extra_copy_mask, TARGET_COLUMN] == FRAUD_LABEL).sum()),
        conflicting_label_groups=int((labels_per_feature_group > 1).sum()),
    )


def remove_exact_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas idênticas em todas as colunas, mantendo a primeira ocorrência.

    O índice original é preservado de propósito: ele identifica a transação no arquivo
    bruto e permite rastrear qualquer linha até a origem.

    Args:
        frame: Dataset bruto.

    Returns:
        Nova tabela sem duplicatas exatas.
    """
    return frame.drop_duplicates(keep="first")
