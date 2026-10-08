"""Engenharia de features linha a linha (sem parâmetros aprendidos).

Tudo aqui transforma cada transação usando só os valores dela mesma: nada é estimado a
partir de outras linhas. Por isso pode ser aplicado antes do split sem vazamento. O que
aprende com os dados (scaler, PCA, seleção, undersampling) mora no pipeline e é ajustado
só no treino.
"""

from typing import Final

import numpy as np
import pandas as pd

from fraud_qml.data import PCA_FEATURE_COLUMNS, TARGET_COLUMN

SECONDS_PER_HOUR: Final[int] = 3_600
HOURS_PER_DAY: Final[int] = 24
SECONDS_PER_DAY: Final[int] = SECONDS_PER_HOUR * HOURS_PER_DAY

ENGINEERED_COLUMNS: Final[tuple[str, ...]] = ("amount_log", "hour_sin", "hour_cos")

# Entrada dos modelos: 28 componentes V + 3 features derivadas = 31 colunas
MODEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (*PCA_FEATURE_COLUMNS, *ENGINEERED_COLUMNS)

# Mantidas para análise (gráficos, modelo de custo do limiar), mas NUNCA entregues aos modelos
AUXILIARY_COLUMNS: Final[tuple[str, ...]] = ("Amount", "hour_of_day")


def hour_of_day(time_seconds: pd.Series) -> pd.Series:
    """Converte segundos corridos em hora do dia contínua, no intervalo [0, 24).

    Supõe que ``Time = 0`` corresponde à meia-noite. O dataset não informa a hora da
    primeira transação; a suposição é checada na EDA pelo vale de volume de madrugada.

    Args:
        time_seconds: Coluna ``Time`` (segundos desde a primeira transação).

    Returns:
        Hora do dia com fração (ex.: 13,5 = 13h30).
    """
    return (time_seconds % SECONDS_PER_DAY) / SECONDS_PER_HOUR


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Cria as features dos modelos a partir do dataset limpo.

    - ``amount_log = log(1 + Amount)``: comprime a cauda longa dos valores. Usa-se
      ``log1p`` porque existem transações de valor zero, e log(0) não existe.
    - ``hour_sin``/``hour_cos``: hora do dia como ponto num círculo, para que 23h59 e 00h00
      fiquem próximas (na hora "crua" elas distam 24 unidades).
    - ``Time`` bruto é descartado: segundos corridos em dois dias descrevem *quando* a
      coleta aconteceu, não um padrão que se repita no futuro.

    Args:
        frame: Dataset com as colunas originais (``Time``, ``V1``–``V28``, ``Amount``, ``Class``).

    Returns:
        Nova tabela com ``MODEL_FEATURE_COLUMNS``, ``AUXILIARY_COLUMNS`` e o rótulo, no mesmo índice.
    """
    hours = hour_of_day(frame["Time"])
    hour_angle = 2 * np.pi * hours / HOURS_PER_DAY

    engineered = pd.DataFrame(
        {
            "amount_log": np.log1p(frame["Amount"]),
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "Amount": frame["Amount"],
            "hour_of_day": hours,
        },
        index=frame.index,
    )
    result = pd.concat([frame[list(PCA_FEATURE_COLUMNS)], engineered], axis="columns")
    result[TARGET_COLUMN] = frame[TARGET_COLUMN]
    return result[[*MODEL_FEATURE_COLUMNS, *AUXILIARY_COLUMNS, TARGET_COLUMN]]
