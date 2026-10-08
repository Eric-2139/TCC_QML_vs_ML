"""Coleta de metadados do ambiente."""

from pathlib import Path

from fraud_qml.environment import collect_versions, current_git_commit


def test_collect_versions_includes_python_and_tracked_packages() -> None:
    versions = collect_versions(("numpy", "qiskit"))
    assert set(versions) == {"python", "numpy", "qiskit"}
    assert versions["numpy"] != "não instalado"


def test_collect_versions_marks_missing_package() -> None:
    versions = collect_versions(("pacote-que-nao-existe-xyz",))
    assert versions["pacote-que-nao-existe-xyz"] == "não instalado"


def test_current_git_commit_returns_none_outside_repository(tmp_path: Path) -> None:
    assert current_git_commit(tmp_path) is None
