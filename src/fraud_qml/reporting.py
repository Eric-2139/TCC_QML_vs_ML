"""Relatórios em JSON dos resultados que não são buscas do grid (a avaliação da Seção 7 em diante).

Fica fora do que a busca do grid importa (``tuning.package_dependencies``): mudá-lo não invalida o cache.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def json_ready(value: object) -> object:
    """Converte para tipos que o ``json`` aceita: tipos do NumPy viram nativos, tuplas viram listas e NaN vira null.

    Args:
        value: Valor, dicionário ou sequência (aninhados).

    Returns:
        O mesmo conteúdo, só com tipos nativos.
    """
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_json_report(path: Path, payload: Mapping[str, object]) -> None:
    """Grava um relatório em JSON (UTF-8, indentado, acentos legíveis), criando a pasta se preciso.

    Args:
        path: Arquivo de destino.
        payload: Conteúdo do relatório.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
