"""Identidade visual dos gráficos do TCC.

As cores das classes seguem uma paleta validada para daltonismo (diferença perceptual
ΔE ≥ 24 entre as duas cores, inclusive sob protanopia), e cada classe é sempre desenhada
com a mesma cor em todas as figuras: o leitor aprende uma vez que laranja é fraude.
"""

from typing import Final

import matplotlib as mpl

from fraud_qml.data import FRAUD_LABEL, LEGITIMATE_LABEL

CLASS_COLORS: Final[dict[int, str]] = {LEGITIMATE_LABEL: "#2a78d6", FRAUD_LABEL: "#eb6834"}
CLASS_NAMES: Final[dict[int, str]] = {LEGITIMATE_LABEL: "legítima", FRAUD_LABEL: "fraude"}

# Cor única para gráficos de uma só série (magnitude), a mesma da classe majoritária
NEUTRAL_SERIES_COLOR: Final[str] = "#2a78d6"

_SURFACE_COLOR: Final[str] = "#fcfcfb"
_TEXT_PRIMARY: Final[str] = "#1a1a19"
_TEXT_SECONDARY: Final[str] = "#5f5e58"
_GRID_COLOR: Final[str] = "#e6e5e0"


def apply_plot_style() -> None:
    """Aplica o estilo padrão: fundo claro, grade discreta, texto em tons neutros.

    Eixos e grade ficam recessivos para que a atenção vá para os dados; o texto nunca
    usa a cor de uma série, apenas tons neutros.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": _SURFACE_COLOR,
            "axes.facecolor": _SURFACE_COLOR,
            "savefig.facecolor": _SURFACE_COLOR,
            "figure.dpi": 110,
            "axes.edgecolor": _GRID_COLOR,
            "axes.labelcolor": _TEXT_SECONDARY,
            "axes.titlecolor": _TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": _GRID_COLOR,
            "grid.linewidth": 0.8,
            "xtick.color": _TEXT_SECONDARY,
            "ytick.color": _TEXT_SECONDARY,
            "text.color": _TEXT_PRIMARY,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "font.size": 10,
        }
    )
