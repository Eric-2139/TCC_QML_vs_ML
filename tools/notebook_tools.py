"""Operações sobre o notebook do experimento: editar células, executar, inspecionar saídas.

Substitui os scripts improvisados de edição, que já produziram erro (um ``\\n`` escapado no
shell virou quebra de linha dentro de uma string). Três princípios:

- células são localizadas por **marcador de texto** único, nunca por índice, porque os índices
  mudam a cada inserção;
- toda alteração no código de uma célula **apaga as saídas dela**, para que "editei e não
  executei" apareça como célula não executada (o Stop hook usa isso);
- a verificação de saúde é determinística e barata, para rodar em hook.

Uso pela linha de comando (a partir da raiz do projeto)::

    .venv/Scripts/python.exe tools/notebook_tools.py health
    .venv/Scripts/python.exe tools/notebook_tools.py dump --from "### 3.6 "
    .venv/Scripts/python.exe tools/notebook_tools.py figures <pasta> --from "### 3.6 "
    .venv/Scripts/python.exe tools/notebook_tools.py execute
"""

import argparse
import base64
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Final

import nbformat
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK: Final[Path] = PROJECT_ROOT / "notebooks" / "tcc_ml_vs_qml.ipynb"
NOTEBOOK_VERSION: Final[int] = 4
DEFAULT_EXECUTION_TIMEOUT_SECONDS: Final[int] = 3600

# Reexportados para que os scripts de construção de células importem tudo de um lugar só
code_cell = new_code_cell
markdown_cell = new_markdown_cell


class CellNotFoundError(LookupError):
    """O marcador não identifica exatamente uma célula."""


def load(path: Path = DEFAULT_NOTEBOOK) -> NotebookNode:
    """Lê o notebook."""
    return nbformat.read(path, as_version=NOTEBOOK_VERSION)


def save(notebook: NotebookNode, path: Path = DEFAULT_NOTEBOOK) -> None:
    """Grava o notebook, validando o formato antes."""
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def find_cell_index(notebook: NotebookNode, marker: str) -> int:
    """Devolve o índice da **única** célula cujo texto contém ``marker``.

    Args:
        notebook: Notebook carregado.
        marker: Trecho de texto que identifica a célula.

    Returns:
        Índice da célula.

    Raises:
        CellNotFoundError: Se nenhuma ou mais de uma célula contiver o marcador. Um marcador
            ambíguo editaria a célula errada sem aviso.
    """
    matches = [index for index, cell in enumerate(notebook.cells) if marker in cell.source]
    if len(matches) != 1:
        raise CellNotFoundError(f"marcador {marker!r} encontrado em {len(matches)} células (esperado: 1)")
    return matches[0]


def _clear_outputs(cell: NotebookNode) -> None:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None


def append_cells(cells: list[NotebookNode], path: Path = DEFAULT_NOTEBOOK) -> None:
    """Acrescenta células ao fim do notebook."""
    notebook = load(path)
    notebook.cells.extend(cells)
    save(notebook, path)


def insert_after(marker: str, cells: list[NotebookNode], path: Path = DEFAULT_NOTEBOOK) -> None:
    """Insere células logo depois da célula identificada por ``marker``."""
    notebook = load(path)
    position = find_cell_index(notebook, marker) + 1
    notebook.cells[position:position] = cells
    save(notebook, path)


def delete_cell(marker: str, path: Path = DEFAULT_NOTEBOOK) -> NotebookNode:
    """Remove a célula identificada por ``marker`` e a devolve.

    Raises:
        CellNotFoundError: Se o marcador for ambíguo ou ausente.
    """
    notebook = load(path)
    removed = notebook.cells.pop(find_cell_index(notebook, marker))
    save(notebook, path)
    return removed


def truncate_from(marker: str, path: Path = DEFAULT_NOTEBOOK) -> list[NotebookNode]:
    """Remove a célula identificada por ``marker`` e todas as seguintes; devolve as células removidas.

    Serve para reconstruir o fim do notebook quando a ordem das seções muda: corta-se a partir da
    primeira célula afetada e as células novas entram com ``append_cells``. As removidas são
    devolvidas para que o script de reconstrução reaproveite o código delas sem redigitar.

    Raises:
        CellNotFoundError: Se o marcador for ambíguo ou ausente.
    """
    notebook = load(path)
    position = find_cell_index(notebook, marker)
    removed = list(notebook.cells[position:])
    del notebook.cells[position:]
    save(notebook, path)
    return removed


def replace_in_cell(marker: str, old: str, new: str, path: Path = DEFAULT_NOTEBOOK) -> None:
    """Troca ``old`` por ``new`` na célula identificada por ``marker`` e apaga as saídas dela.

    Raises:
        CellNotFoundError: Se o marcador for ambíguo ou ausente.
        ValueError: Se ``old`` não aparecer exatamente uma vez na célula.
    """
    notebook = load(path)
    cell = notebook.cells[find_cell_index(notebook, marker)]
    occurrences = cell.source.count(old)
    if occurrences != 1:
        raise ValueError(f"o trecho a substituir aparece {occurrences} vezes na célula (esperado: 1)")
    cell.source = cell.source.replace(old, new)
    _clear_outputs(cell)
    save(notebook, path)


def set_cell_source(marker: str, new_source: str, path: Path = DEFAULT_NOTEBOOK) -> None:
    """Substitui todo o conteúdo da célula identificada por ``marker`` e apaga as saídas dela.

    Raises:
        CellNotFoundError: Se o marcador for ambíguo ou ausente.
    """
    notebook = load(path)
    cell = notebook.cells[find_cell_index(notebook, marker)]
    cell.source = new_source
    _clear_outputs(cell)
    save(notebook, path)


def execute(path: Path = DEFAULT_NOTEBOOK, timeout_seconds: int = DEFAULT_EXECUTION_TIMEOUT_SECONDS) -> None:
    """Executa o notebook inteiro no kernel do ambiente atual e grava as saídas no próprio arquivo.

    Raises:
        subprocess.CalledProcessError: Se alguma célula falhar.
    """
    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={timeout_seconds}", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _start_index(notebook: NotebookNode, from_marker: str | None) -> int:
    return 0 if from_marker is None else find_cell_index(notebook, from_marker)


class _HtmlTableText(HTMLParser):
    """Converte as tabelas de um HTML em linhas de texto separadas por " | " (legenda incluída)."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._caption: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []
        elif tag == "caption":
            self._caption = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.lines.append(" | ".join(self._row))
            self._row = None
        elif tag == "caption" and self._caption is not None:
            self.lines.append(f"[tabela] {' '.join(''.join(self._caption).split())}")
            self._caption = None

    def handle_data(self, data: str) -> None:
        for buffer in (self._cell, self._caption):
            if buffer is not None:
                buffer.append(data)


def html_tables_to_text(html: str) -> str:
    """Texto das tabelas de um HTML (usado para ler saídas de ``DataFrame.style``, cujo texto puro é só o repr)."""
    parser = _HtmlTableText()
    parser.feed(html)
    return "\n".join(parser.lines)


def _output_text(output: NotebookNode) -> str:
    if output.output_type == "stream":
        return output.text
    if output.output_type == "error":
        return f"[ERRO] {output.ename}: {output.evalue}"
    data = output.get("data", {})
    if "image/png" in data:
        return "[figura]"
    # Tabelas formatadas (Styler) só têm o conteúdo no HTML; o texto puro é "<Styler at 0x...>"
    if "text/html" in data and "<table" in data["text/html"]:
        return html_tables_to_text(data["text/html"])
    return data.get("text/plain", "")


def dump_outputs(path: Path = DEFAULT_NOTEBOOK, from_marker: str | None = None) -> str:
    """Texto das células a partir de ``from_marker``: markdown na íntegra e saídas das células de código.

    É o material que o auditor compara: cada número afirmado num markdown precisa aparecer numa
    saída de código.
    """
    notebook = load(path)
    blocks: list[str] = []
    start = _start_index(notebook, from_marker)
    for index, cell in enumerate(notebook.cells[start:], start=start):
        if cell.cell_type == "markdown":
            blocks.append(f"--- célula {index} (markdown)\n{cell.source}")
        elif cell.cell_type == "code":
            outputs = "\n".join(_output_text(output) for output in cell.get("outputs", []))
            blocks.append(f"--- célula {index} (código, execução {cell.execution_count})\n{outputs}")
    return "\n\n".join(blocks)


def extract_figures(out_dir: Path, path: Path = DEFAULT_NOTEBOOK, from_marker: str | None = None) -> list[Path]:
    """Salva as figuras (PNG) das células a partir de ``from_marker``, para serem inspecionadas uma a uma.

    Returns:
        Caminhos dos arquivos gravados, no formato ``celula<índice>_<n>.png``.
    """
    notebook = load(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    start = _start_index(notebook, from_marker)
    for index, cell in enumerate(notebook.cells[start:], start=start):
        figures = [output["data"]["image/png"] for output in cell.get("outputs", []) if "image/png" in output.get("data", {})]
        for number, encoded in enumerate(figures):
            target = out_dir / f"celula{index}_{number}.png"
            target.write_bytes(base64.b64decode(encoded))
            written.append(target)
    return written


def check_health(path: Path = DEFAULT_NOTEBOOK) -> list[str]:
    """Lista problemas que impedem considerar o notebook entregue.

    - saídas de erro;
    - células de código com conteúdo e sem execução (inclui as editadas por ``replace_in_cell``).

    A ordem das execuções não é verificada de propósito: execuções parciais no VS Code não são
    um defeito do notebook, e o ``execute()`` sempre roda tudo em ordem.

    Returns:
        Descrições legíveis; lista vazia quando está tudo certo.
    """
    problems: list[str] = []
    for index, cell in enumerate(load(path).cells):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        if cell.execution_count is None:
            first_line = cell.source.strip().splitlines()[0][:60]
            problems.append(f"{path.name}: célula {index} não executada ({first_line!r})")
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                problems.append(f"{path.name}: célula {index} com erro {output.ename}: {output.evalue}")
    return problems


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("health", help="lista erros e células não executadas")
    dump = commands.add_parser("dump", help="texto das células e saídas")
    dump.add_argument("--from", dest="from_marker")
    figures = commands.add_parser("figures", help="salva as figuras em PNG")
    figures.add_argument("out_dir", type=Path)
    figures.add_argument("--from", dest="from_marker")
    commands.add_parser("execute", help="executa o notebook inteiro")
    arguments = parser.parse_args()

    # O console do Windows usa cp1252 por padrão; as saídas têm acentos e símbolos (π, ×)
    sys.stdout.reconfigure(encoding="utf-8")
    if arguments.command == "health":
        problems = check_health(arguments.notebook)
        print("\n".join(problems) if problems else "notebook saudável")
        return 1 if problems else 0
    if arguments.command == "dump":
        print(dump_outputs(arguments.notebook, arguments.from_marker))
    elif arguments.command == "figures":
        for figure in extract_figures(arguments.out_dir, arguments.notebook, arguments.from_marker):
            print(figure)
    elif arguments.command == "execute":
        execute(arguments.notebook)
        print("executado")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
