"""Cache em disco das medidas das seções depois da escolha (Seção 8 em diante).

A busca da Seção 6 tem o próprio cache (``tuning.cached_grid_search``), com a chave limitada ao código que ela
importa. Este módulo faz o mesmo para qualquer outra medida cara: a chave junta o que define a medida (os
parâmetros e os dados), o código do pacote que ela executa e as versões das bibliotecas. Ele **não** é importado
pelo ``tuning``: mudá-lo não invalida as buscas da Seção 6.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import joblib

from fraud_qml.environment import collect_versions
from fraud_qml.tuning import package_source_fingerprint

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class CachedResult(Generic[ResultT]):
    """Um resultado e de onde ele veio.

    Attributes:
        value: O resultado da medida.
        from_cache: Se foi lido do disco (calculado antes com a mesma chave).
        cache_file: Arquivo do cache.
    """

    value: ResultT
    from_cache: bool
    cache_file: Path


def computation_cache_key(name: str, definition: object, root_module: str) -> str:
    """Identidade de uma medida: o nome, o que a define, o código que ela executa e as versões das bibliotecas.

    Args:
        name: Nome da medida (entra também no nome do arquivo).
        definition: Tudo o que muda o resultado: parâmetros, configurações e os próprios dados (o ``joblib.hash``
            lê o conteúdo de arrays e tabelas).
        root_module: Módulo do pacote cujo código (com tudo o que ele importa do pacote) entra na chave.

    Returns:
        Hash hexadecimal.
    """
    return joblib.hash((name, definition, package_source_fingerprint(root_module), collect_versions()))


def cached_computation(
    name: str, definition: object, root_module: str, compute: Callable[[], ResultT], cache_dir: Path
) -> CachedResult[ResultT]:
    """Lê do disco o resultado de uma medida, ou a calcula e grava.

    Args:
        name: Nome da medida.
        definition: O que define a medida (ver ``computation_cache_key``).
        root_module: Módulo do pacote que a medida executa.
        compute: Função sem argumentos que faz a medida.
        cache_dir: Pasta do cache.

    Returns:
        O resultado e se ele veio do disco.
    """
    cache_file = cache_dir / f"{name}_{computation_cache_key(name, definition, root_module)}.joblib"
    if cache_file.is_file():
        return CachedResult(value=joblib.load(cache_file), from_cache=True, cache_file=cache_file)
    value = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(value, cache_file)
    return CachedResult(value=value, from_cache=False, cache_file=cache_file)
