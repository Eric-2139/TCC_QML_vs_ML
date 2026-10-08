"""Metadados do ambiente de execução, anexados a cada resultado salvo.

Um número no TCC só é reproduzível se soubermos com quais versões e com qual commit ele
foi gerado; por isso esses metadados viajam junto com cada arquivo de resultados.
"""

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "imbalanced-learn",
    "qiskit",
    "qiskit-aer",
    "qiskit-machine-learning",
    "matplotlib",
)


def collect_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Retorna a versão do Python e de cada pacote rastreado.

    Usa ``importlib.metadata`` em vez de ``modulo.__version__`` porque o nome de
    distribuição (ex.: ``scikit-learn``) nem sempre coincide com o nome de importação
    (``sklearn``), e alguns pacotes não expõem ``__version__``.

    Args:
        packages: Nomes de distribuição (como no pip) a consultar.

    Returns:
        Dicionário ``{pacote: versão}``; pacotes ausentes aparecem como ``"não instalado"``.
    """
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "não instalado"
    return versions


def current_git_commit(repo_dir: Path) -> str | None:
    """Retorna o hash do commit atual, ou ``None`` se não houver repositório/commit.

    Args:
        repo_dir: Diretório dentro do repositório git.

    Returns:
        Hash completo do ``HEAD`` ou ``None`` quando o git falha (sem commits, sem git).
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()
