"""Operações de edição e inspeção do notebook, com um notebook sintético."""

import base64
from pathlib import Path

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

from tools import notebook_tools as nbt

# PNG de 1×1 pixel, o menor arquivo de figura válido
TINY_PNG = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )
).decode()


@pytest.fixture
def notebook_path(tmp_path: Path) -> Path:
    executed = new_code_cell("x = 1\nprint(x)", execution_count=1)
    executed.outputs = [new_output("stream", name="stdout", text="1\n")]
    figure = new_code_cell("plot()", execution_count=2)
    figure.outputs = [new_output("display_data", data={"image/png": TINY_PNG, "text/plain": "<Figure>"})]
    notebook = new_notebook(
        cells=[new_markdown_cell("## Seção A"), executed, new_markdown_cell("## Seção B"), figure]
    )
    path = tmp_path / "exemplo.ipynb"
    nbformat.write(notebook, path)
    return path


def test_find_cell_requires_unique_marker(notebook_path: Path) -> None:
    notebook = nbt.load(notebook_path)
    assert nbt.find_cell_index(notebook, "## Seção B") == 2
    with pytest.raises(nbt.CellNotFoundError):
        nbt.find_cell_index(notebook, "## Seção")  # aparece em duas células
    with pytest.raises(nbt.CellNotFoundError):
        nbt.find_cell_index(notebook, "inexistente")


def test_delete_cell_removes_only_the_marked_cell(notebook_path: Path) -> None:
    removed = nbt.delete_cell("## Seção A", notebook_path)
    assert removed.source == "## Seção A"
    assert [cell.source for cell in nbt.load(notebook_path).cells] == ["x = 1\nprint(x)", "## Seção B", "plot()"]
    with pytest.raises(nbt.CellNotFoundError):
        nbt.delete_cell("marcador inexistente", notebook_path)


def test_truncate_from_removes_the_marked_cell_and_everything_after(notebook_path: Path) -> None:
    removed = nbt.truncate_from("## Seção B", notebook_path)
    remaining = nbt.load(notebook_path).cells
    assert [cell.source for cell in remaining] == ["## Seção A", "x = 1\nprint(x)"]
    assert [cell.source for cell in removed] == ["## Seção B", "plot()"]
    # As células anteriores ao corte mantêm as saídas: só o que foi removido precisa ser refeito
    assert remaining[1].outputs


def test_truncate_from_with_unknown_marker_changes_nothing(notebook_path: Path) -> None:
    before = notebook_path.read_bytes()
    with pytest.raises(nbt.CellNotFoundError):
        nbt.truncate_from("marcador inexistente", notebook_path)
    assert notebook_path.read_bytes() == before


def test_insert_after_places_cells_by_marker(notebook_path: Path) -> None:
    nbt.insert_after("## Seção A", [nbt.markdown_cell("inserida")], notebook_path)
    assert nbt.load(notebook_path).cells[1].source == "inserida"


def test_replace_in_cell_clears_outputs(notebook_path: Path) -> None:
    nbt.replace_in_cell("print(x)", "x = 1", "x = 2", notebook_path)
    cell = nbt.load(notebook_path).cells[1]
    assert cell.source.startswith("x = 2")
    assert cell.outputs == [] and cell.execution_count is None


def test_set_cell_source_replaces_whole_cell_and_clears_outputs(notebook_path: Path) -> None:
    nbt.set_cell_source("print(x)", "y = 3", notebook_path)
    cell = nbt.load(notebook_path).cells[1]
    assert cell.source == "y = 3"
    assert cell.outputs == [] and cell.execution_count is None


def test_replace_in_cell_rejects_ambiguous_text(notebook_path: Path) -> None:
    with pytest.raises(ValueError, match="aparece 2 vezes"):
        nbt.replace_in_cell("print(x)", "x", "y", notebook_path)


def test_health_flags_unexecuted_and_error_cells(notebook_path: Path) -> None:
    assert nbt.check_health(notebook_path) == []
    nbt.replace_in_cell("print(x)", "x = 1", "x = 2", notebook_path)  # vira não executada
    broken = new_code_cell("1 / 0", execution_count=3)
    broken.outputs = [new_output("error", ename="ZeroDivisionError", evalue="division by zero", traceback=[])]
    nbt.append_cells([broken], notebook_path)
    problems = nbt.check_health(notebook_path)
    assert len(problems) == 2
    assert any("não executada" in problem for problem in problems)
    assert any("ZeroDivisionError" in problem for problem in problems)


def test_dump_outputs_includes_markdown_and_outputs_from_marker(notebook_path: Path) -> None:
    text = nbt.dump_outputs(notebook_path, from_marker="## Seção B")
    assert "## Seção A" not in text
    assert "## Seção B" in text and "[figura]" in text


def test_dump_reads_styled_tables_from_html(notebook_path: Path) -> None:
    import pandas as pd

    styled = pd.DataFrame({"KTA": [0.56, 0.42]}, index=["1 rad", "π"]).style.set_caption("Tabela de teste")
    cell = new_code_cell("tabela", execution_count=5)
    cell.outputs = [
        new_output("execute_result", data={"text/html": styled.to_html(), "text/plain": "<Styler at 0x1>"}, execution_count=5)
    ]
    nbt.append_cells([cell], notebook_path)
    text = nbt.dump_outputs(notebook_path, from_marker="tabela")
    assert "Styler" not in text
    assert "[tabela] Tabela de teste" in text
    assert "1 rad | 0.560000" in text and "π | 0.420000" in text


def test_extract_figures_writes_png(notebook_path: Path, tmp_path: Path) -> None:
    written = nbt.extract_figures(tmp_path / "figuras", notebook_path)
    assert [path.name for path in written] == ["celula3_0.png"]
    assert written[0].read_bytes().startswith(b"\x89PNG")
