"""Testes de significância da Seção 9: o teste de McNemar exato e a correção de Holm.

**McNemar.** Compara dois classificadores nas mesmas linhas. As linhas em que os dois acertam, ou os dois erram, não
dizem nada sobre qual é melhor; só contam as **discordantes**: ``b`` linhas em que só o primeiro acerta e ``c`` em que
só o segundo acerta. Se os dois tivessem a mesma taxa de erro, cada discordante cairia de um lado ou do outro com
chance 1/2, e ``b`` seguiria uma binomial com ``n = b + c`` e ``p = 1/2``. A versão **exata** calcula o p por essa
binomial (Raschka 2018, seções 4.3 e 4.4), em vez da aproximação qui-quadrado, que precisa de muitas discordantes.

O McNemar conta **erros**, e um erro numa fraude (uma fraude sem alerta) pesa o mesmo que um erro numa legítima (um
alarme falso). Por isso o resultado traz as discordantes separadas por classe: o leitor vê se a diferença está nas
fraudes ou nos alarmes.

**Holm.** Com várias comparações, a chance de alguma dar p < 0,05 por acaso cresce. A correção de Holm ordena os p do
menor para o maior e multiplica o k-ésimo menor por (m − k + 1), mantendo a ordem (um p ajustado nunca fica abaixo do
anterior). Controla a chance de **algum** falso positivo entre as m comparações, como Bonferroni, mas rejeita mais.

Este módulo fica fora do que a busca do grid importa (``tuning.package_dependencies``): mudá-lo não invalida o cache.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binomtest

from fraud_qml.data import FRAUD_LABEL

# Sob a hipótese de taxas de erro iguais, cada discordante cai de um lado ou do outro com esta chance
EQUAL_ERROR_PROBABILITY: Final[float] = 0.5


@dataclass(frozen=True)
class McNemarResult:
    """As linhas discordantes de dois classificadores e o p exato do McNemar.

    Attributes:
        first_only_correct: Linhas em que só o primeiro acerta (``b``).
        second_only_correct: Linhas em que só o segundo acerta (``c``).
        first_only_correct_frauds: Das ``b``, quantas são fraudes (o primeiro alerta e o segundo não).
        second_only_correct_frauds: Das ``c``, quantas são fraudes (o segundo alerta e o primeiro não).
        p_value: p bilateral exato, pela binomial com ``n = b + c`` e ``p = 1/2``; 1 quando não há discordantes.
    """

    first_only_correct: int
    second_only_correct: int
    first_only_correct_frauds: int
    second_only_correct_frauds: int
    p_value: float

    @property
    def discordant(self) -> int:
        """Total de linhas discordantes (``b + c``)."""
        return self.first_only_correct + self.second_only_correct

    @property
    def first_only_correct_legitimate(self) -> int:
        """Das ``b``, quantas são legítimas (o segundo dá um alarme falso que o primeiro não dá)."""
        return self.first_only_correct - self.first_only_correct_frauds

    @property
    def second_only_correct_legitimate(self) -> int:
        """Das ``c``, quantas são legítimas (o primeiro dá um alarme falso que o segundo não dá)."""
        return self.second_only_correct - self.second_only_correct_frauds


def mcnemar_exact(labels: ArrayLike, first_predictions: ArrayLike, second_predictions: ArrayLike) -> McNemarResult:
    """Teste de McNemar exato entre as classes previstas por dois classificadores nas mesmas linhas.

    Args:
        labels: Rótulos verdadeiros.
        first_predictions: Classe prevista pelo primeiro classificador.
        second_predictions: Classe prevista pelo segundo, nas mesmas linhas.

    Returns:
        As discordantes (no total e nas fraudes) e o p bilateral exato.

    Raises:
        ValueError: Se os três vetores não tiverem o mesmo tamanho.
    """
    truth = np.asarray(labels)
    first, second = np.asarray(first_predictions), np.asarray(second_predictions)
    if not truth.shape == first.shape == second.shape:
        raise ValueError(f"tamanhos diferentes: {truth.shape}, {first.shape} e {second.shape}")
    first_correct, second_correct = first == truth, second == truth
    is_fraud = truth == FRAUD_LABEL
    only_first = first_correct & ~second_correct
    only_second = second_correct & ~first_correct
    b, c = int(only_first.sum()), int(only_second.sum())
    p_value = 1.0 if b + c == 0 else float(binomtest(b, b + c, EQUAL_ERROR_PROBABILITY, alternative="two-sided").pvalue)
    return McNemarResult(
        first_only_correct=b,
        second_only_correct=c,
        first_only_correct_frauds=int((only_first & is_fraud).sum()),
        second_only_correct_frauds=int((only_second & is_fraud).sum()),
        p_value=p_value,
    )


def holm_adjusted(p_values: Sequence[float]) -> NDArray[np.float64]:
    """Valores de p ajustados pela correção de Holm, na ordem em que foram dados.

    O k-ésimo menor p (k = 1, ..., m) é multiplicado por m − k + 1; depois, cada ajustado vira o máximo entre ele e os
    ajustados dos p menores (a correção é em degraus e não pode inverter a ordem), limitado a 1. Uma comparação é
    significativa ao nível α quando o p ajustado dela é menor que α.

    Args:
        p_values: Os p das m comparações.

    Returns:
        Os p ajustados, na mesma ordem.
    """
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    multipliers = len(values) - np.arange(len(values))
    stepped = np.minimum(np.maximum.accumulate(values[order] * multipliers), 1.0)
    adjusted = np.empty_like(values)
    adjusted[order] = stepped
    return adjusted
