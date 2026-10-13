"""O notebook cumpre a regra de vazamento de dados na exploração (CC7 e NC12 em methodological_issues.md).

A regra (CC7): na base inteira, só o contrato dos dados e a remoção de duplicatas; depois do split,
a exploração (Seção 4) lê apenas linhas de treino. Este teste varre o código da Seção 4 e falha se
alguma célula tocar na base inteira, na validação ou no teste fora das exceções declaradas.
"""

import re
from typing import Final

import pytest

from tools import notebook_tools as nbt

EXPLORATION_SECTION_MARKER: Final[str] = "## 4. Análise exploratória e pré-processamento"
NEXT_SECTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^## 5\. ", re.MULTILINE)

# Nomes que dão acesso a linhas fora do treino (o \b impede que train_raw_df case com raw_df)
FORBIDDEN_NAMES: Final[tuple[str, ...]] = (
    "raw_df",
    "clean_df",
    "validation_raw_df",
    "test_raw_df",
    "validation_df",
    "test_df",
    "X_validation",
    "y_validation",
)

# Células que usam validação ou teste de propósito, cada uma com o motivo
ALLOWED_CELLS: Final[dict[str, str]] = {
    "engineer_features(part) for part in": "aplica às três partes a transformação linha a linha, sem ajustar nada",
    "prior_probe = ImbPipeline(": "prevê na validação para provar que o undersampling é pulado na predição",
    "composition = pd.DataFrame(": "conta linhas por classe em cada parte",
}


def _exploration_code_cells() -> list[str]:
    notebook = nbt.load()
    start = nbt.find_cell_index(notebook, EXPLORATION_SECTION_MARKER)
    sources: list[str] = []
    for cell in notebook.cells[start + 1 :]:
        if cell.cell_type == "markdown" and NEXT_SECTION_PATTERN.search(cell.source):
            break
        if cell.cell_type == "code":
            sources.append(cell.source)
    return sources


def test_exploration_section_has_code_to_check() -> None:
    assert len(_exploration_code_cells()) > 20


@pytest.mark.parametrize("name", FORBIDDEN_NAMES)
def test_exploration_only_reads_training_rows(name: str) -> None:
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    offenders = [
        source.splitlines()[0]
        for source in _exploration_code_cells()
        if pattern.search(source) and not any(marker in source for marker in ALLOWED_CELLS)
    ]
    assert offenders == [], f"células da Seção 4 que usam {name} fora das exceções declaradas: {offenders}"


def test_exploration_has_no_hand_picked_columns() -> None:
    # Escolher a dedo uma coluna "porque já sabíamos que ela importa" é conhecimento prévio da base inteira
    # (NC12): a seleção precisa sair de uma saída calculada no treino, ou a análise usa todas as colunas
    hand_picked = re.compile(r"[\"']V\d{1,2}[\"']")
    offenders = [source.splitlines()[0] for source in _exploration_code_cells() if hand_picked.search(source)]
    assert offenders == [], f"células da Seção 4 com coluna escolhida a dedo: {offenders}"


# Células da Seção 5 que definem opção do grid ou se declaram "só treino": cada uma identificada por um trecho
# único do código. A 5.4 e a 5.5 avaliam na validação de propósito e ficam de fora.
TRAIN_ONLY_TRAINING_CELLS: Final[dict[str, str]] = {
    "WEAK_LEARNER_CURVE_ITERATIONS: int": "5.2, curva de custo do classificador fraco",
    "ENCODING_CONFIGURATIONS: dict[str, EncodingConfiguration]": "5.3, fase × rotação em Y",
    "WIDTH_SWEEP_SCALERS: dict[ScalerName, str]": "5.3.1, largura e scaler",
    "MECHANISM_ENCODINGS: tuple[str, ...]": "5.3.2, mecanismo",
    "INNER_HOLDOUT_FRACTION: float": "5.4.1, variação entre sorteios",
    "BOOSTER_SEEDS: tuple[int, ...]": "5.5.1, parada do boosting",
}


@pytest.mark.parametrize("marker", list(TRAIN_ONLY_TRAINING_CELLS))
def test_train_only_cells_of_the_training_section_do_not_read_validation_or_test(marker: str) -> None:
    # As decisões e os diagnósticos da Seção 5 que dizem usar só o treino não podem tocar em validação ou teste
    notebook = nbt.load()
    source = notebook.cells[nbt.find_cell_index(notebook, marker)].source
    used = [name for name in FORBIDDEN_NAMES if re.search(rf"\b{re.escape(name)}\b", source)]
    assert used == [], f"{TRAIN_ONLY_TRAINING_CELLS[marker]} usa {used}"


def test_every_declared_exception_still_exists() -> None:
    # Uma exceção que não corresponde a nenhuma célula é uma porta aberta sem dono
    sources = _exploration_code_cells()
    for marker in ALLOWED_CELLS:
        assert sum(marker in source for source in sources) == 1, marker
