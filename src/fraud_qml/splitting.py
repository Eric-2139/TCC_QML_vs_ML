"""Divisão treino/validação/teste e as verificações que tornam o split confiável.

O split acontece na base inteira (deduplicada), **antes** de qualquer balanceamento: validação e
teste precisam ter a proporção real de fraudes, porque é nela que o modelo vai operar. O
undersampling só existe dentro do pipeline, aplicado ao treino.
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_qml import config
from fraud_qml.data import TARGET_COLUMN

TRAIN_NAME: Final[str] = "train"
VALIDATION_NAME: Final[str] = "validation"
TEST_NAME: Final[str] = "test"

# O split é feito em dois estágios (teste; depois validação dentro do restante), e cada estágio
# arredonda a contagem de cada classe em no máximo 1 linha. Somados, os dois arredondamentos
# afastam a contagem de uma parte do valor exato em menos de 2 linhas.
STRATIFICATION_TOLERANCE_ROWS: Final[float] = 2.0


class SplitError(ValueError):
    """O split viola uma ou mais propriedades esperadas."""


@dataclass(frozen=True)
class DatasetSplit:
    """As três partes do dataset, com o índice original preservado.

    O índice é o da linha no arquivo bruto: qualquer transação pode ser rastreada até a origem, e
    as verificações de disjunção comparam índices, não conteúdos.

    Attributes:
        train: Linhas de treino (o undersampling é aplicado só a elas, dentro do pipeline).
        validation: Linhas de validação, com a proporção real de fraudes.
        test: Linhas de teste, usadas uma única vez na avaliação final.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def parts(self) -> dict[str, pd.DataFrame]:
        """As três partes, na ordem treino, validação, teste."""
        return {TRAIN_NAME: self.train, VALIDATION_NAME: self.validation, TEST_NAME: self.test}

    def membership(self) -> pd.Series:
        """Para cada índice original, o nome da parte em que a linha caiu."""
        return pd.concat(
            [pd.Series(name, index=part.index) for name, part in self.parts().items()]
        ).rename("split")

    def fingerprint(self) -> str:
        """Hash dos índices de cada parte: identifica o split nos resultados salvos.

        Dois splits com o mesmo fingerprint têm exatamente as mesmas linhas em cada parte, então
        resultados de sessões diferentes só são comparáveis quando o fingerprint coincide.
        """
        digest = hashlib.sha256()
        for name, part in self.parts().items():
            digest.update(name.encode())
            digest.update(",".join(map(str, sorted(part.index))).encode())
        return digest.hexdigest()


def stratified_three_way_split(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    train_size: float = config.TRAIN_SIZE,
    validation_size: float = config.VALIDATION_SIZE,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> DatasetSplit:
    """Divide em treino, validação e teste, preservando a proporção de cada classe em todas as partes.

    O scikit-learn só divide em duas partes por chamada. Primeiro o teste é separado; depois a
    validação sai do restante, com a fração reescalada para que, no fim, cada parte tenha a
    fração pedida **da base inteira** (0,2 / 0,8 = 0,25 do restante).

    Args:
        frame: Base deduplicada, com features e rótulo.
        target_column: Coluna usada na estratificação.
        train_size: Fração do treino.
        validation_size: Fração da validação.
        test_size: Fração do teste.
        random_state: Semente do sorteio.

    Returns:
        As três partes, com o índice original.

    Raises:
        ValueError: Se as frações não somarem 1.
    """
    if not math.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError(f"as frações somam {train_size + validation_size + test_size}, e não 1")

    remainder, test = train_test_split(
        frame, test_size=test_size, stratify=frame[target_column], random_state=random_state
    )
    train, validation = train_test_split(
        remainder,
        test_size=validation_size / (train_size + validation_size),
        stratify=remainder[target_column],
        random_state=random_state,
    )
    return DatasetSplit(train=train, validation=validation, test=test)


def find_split_violations(
    frame: pd.DataFrame,
    split: DatasetSplit,
    target_column: str = TARGET_COLUMN,
    sizes: tuple[float, float, float] = (config.TRAIN_SIZE, config.VALIDATION_SIZE, config.TEST_SIZE),
) -> list[str]:
    """Lista as propriedades que o split viola.

    Verifica, sem parar na primeira falha:
    - **disjunção**: nenhuma linha em duas partes (uma linha no treino e no teste é vazamento);
    - **cobertura**: a união das partes é exatamente a base (nenhuma linha perdida ou inventada);
    - **tamanho** de cada parte;
    - **estratificação**: a contagem de cada classe em cada parte.

    Args:
        frame: Base que foi dividida.
        split: Resultado do split.
        target_column: Coluna do rótulo.
        sizes: Frações esperadas de treino, validação e teste.

    Returns:
        Descrições das violações; lista vazia quando o split é válido.
    """
    violations: list[str] = []
    parts = split.parts()

    names = list(parts)
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            shared = parts[first].index.intersection(parts[second].index)
            if len(shared) > 0:
                violations.append(f"{len(shared)} linhas em '{first}' e '{second}' ao mesmo tempo")

    union = pd.Index([]).append([part.index for part in parts.values()])
    if union.has_duplicates or not union.sort_values().equals(frame.index.sort_values()):
        missing = frame.index.difference(union)
        extra = union.difference(frame.index)
        violations.append(f"as partes não cobrem a base ({len(missing)} linhas ausentes, {len(extra)} inexistentes)")

    class_counts = frame[target_column].value_counts()
    for (name, part), fraction in zip(parts.items(), sizes, strict=True):
        expected_rows = fraction * len(frame)
        if abs(len(part) - expected_rows) >= STRATIFICATION_TOLERANCE_ROWS:
            violations.append(f"'{name}' tem {len(part)} linhas, esperado ≈ {expected_rows:.1f}")
        part_counts = part[target_column].value_counts()
        for label, total in class_counts.items():
            expected = fraction * total
            actual = int(part_counts.get(label, 0))
            if abs(actual - expected) >= STRATIFICATION_TOLERANCE_ROWS:
                violations.append(f"'{name}' tem {actual} linhas da classe {label}, esperado ≈ {expected:.1f}")

    return violations


def validate_split(
    frame: pd.DataFrame,
    split: DatasetSplit,
    target_column: str = TARGET_COLUMN,
    sizes: tuple[float, float, float] = (config.TRAIN_SIZE, config.VALIDATION_SIZE, config.TEST_SIZE),
) -> None:
    """Levanta ``SplitError`` com todas as violações, se houver alguma.

    Args:
        frame: Base que foi dividida.
        split: Resultado do split.
        target_column: Coluna do rótulo.
        sizes: Frações esperadas de treino, validação e teste.

    Raises:
        SplitError: Se o split for inválido.
    """
    violations = find_split_violations(frame, split, target_column, sizes)
    if violations:
        bullet_list = "\n".join(f"  - {violation}" for violation in violations)
        raise SplitError(f"O split é inválido:\n{bullet_list}")


@dataclass(frozen=True)
class CrossSplitGroups:
    """Grupos de linhas com os mesmos valores numa lista de colunas, espalhados por mais de uma parte.

    Attributes:
        groups: Grupos com linhas em pelo menos duas partes.
        rows: Linhas pertencentes a esses grupos.
        rows_by_label: Quantas dessas linhas há em cada classe.
        rows_by_split: Quantas dessas linhas há em cada parte.
    """

    groups: int
    rows: int
    rows_by_label: dict[int, int]
    rows_by_split: dict[str, int]


def find_cross_split_groups(
    split: DatasetSplit, key_columns: list[str], target_column: str = TARGET_COLUMN
) -> CrossSplitGroups:
    """Conta quase-duplicatas que atravessam o split.

    A remoção de duplicatas exatas não pega linhas iguais em quase tudo (por exemplo, a mesma
    compra repetida em outro horário). Se uma cópia fica no treino e outra na validação, o modelo
    é avaliado numa transação praticamente já vista.

    Args:
        split: Resultado do split.
        key_columns: Colunas que definem "a mesma transação".
        target_column: Coluna do rótulo.

    Returns:
        Quantos grupos e linhas atravessam o split, por classe.
    """
    # membership() concatena as partes na mesma ordem, então as posições coincidem
    frame = pd.concat(split.parts().values()).assign(split=split.membership().to_numpy())
    parts_per_row = frame.groupby(key_columns, sort=False)["split"].transform("nunique")
    crossing = frame[parts_per_row > 1]
    split_counts = crossing["split"].value_counts()
    return CrossSplitGroups(
        groups=int(crossing.groupby(key_columns, sort=False).ngroups),
        rows=len(crossing),
        rows_by_label={int(label): int(count) for label, count in crossing[target_column].value_counts().items()},
        rows_by_split={name: int(split_counts.get(name, 0)) for name in split.parts()},
    )
