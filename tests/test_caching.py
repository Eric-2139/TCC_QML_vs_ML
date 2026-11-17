"""Cache em disco das medidas da Seção 8 em diante."""

from pathlib import Path

import numpy as np

from fraud_qml.caching import cached_computation, computation_cache_key
from fraud_qml.tuning import package_dependencies

ROOT_MODULE = "fraud_qml.curves"


def test_second_call_reads_the_disk_instead_of_computing(tmp_path: Path) -> None:
    calls: list[int] = []

    def compute() -> np.ndarray:
        calls.append(1)
        return np.arange(3)

    first = cached_computation("medida", {"a": 1}, ROOT_MODULE, compute, tmp_path)
    second = cached_computation("medida", {"a": 1}, ROOT_MODULE, compute, tmp_path)
    assert not first.from_cache and second.from_cache
    assert len(calls) == 1
    np.testing.assert_array_equal(second.value, np.arange(3))


def test_a_different_definition_or_data_gives_a_different_key() -> None:
    base = computation_cache_key("medida", (np.zeros(3),), ROOT_MODULE)
    assert computation_cache_key("medida", (np.ones(3),), ROOT_MODULE) != base
    assert computation_cache_key("outra", (np.zeros(3),), ROOT_MODULE) != base


def test_the_key_follows_the_code_the_measure_runs() -> None:
    # A chave muda com o módulo que a medida executa (e o que ele importa), e não com módulos que ela não usa
    assert computation_cache_key("medida", 1, ROOT_MODULE) != computation_cache_key("medida", 1, "fraud_qml.quantum.diagnostics")
    assert any(path.name == "curves.py" for path in package_dependencies(ROOT_MODULE))


def test_the_search_of_section_six_does_not_import_the_new_modules() -> None:
    # Mudar o código da Seção 8 não pode invalidar as 18 buscas da Seção 6 (183 min sem cache)
    names = {path.name for path in package_dependencies("fraud_qml.tuning")}
    assert {"curves.py", "caching.py", "diagnostics.py"}.isdisjoint(names)
