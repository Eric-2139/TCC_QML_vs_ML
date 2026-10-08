"""Transformações que aprendem com os dados: escalonamento, redução, preparo para codificação quântica.

Tudo aqui é montado como etapa de ``Pipeline``. Assim, em cada fold da validação cruzada, a
média do scaler, os eixos do PCA, o ranking de seleção e o sorteio do undersampling são
reajustados só com as linhas de treino daquele fold: o vazamento fica impossível por construção.
"""

from functools import partial
from typing import Final, Literal, Self

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

ScalerName = Literal["standard", "robust"]
ReducerName = Literal["pca", "mutual_info"]
AngleRange = tuple[float, float]

# Os feature maps do Qiskit giram o qubit por α·x com α = 2, então o período em x é π. No mapa Z, até π/2
# "valores mais distantes" sempre significa "estados menos parecidos"; acima disso, extremos opostos voltam a
# se parecer. No ZZ a monotonicidade acaba antes (≈ 0,6 rad): ali a largura é escolhida por validação cruzada.
DEFAULT_ANGLE_RANGE: Final[AngleRange] = (0.0, float(np.pi / 2))

# Mesma quantidade de legítimas e fraudes no treino (razão minoria/maioria = 1)
BALANCED_SAMPLING_STRATEGY: Final[float] = 1.0


def make_scaler(name: ScalerName) -> StandardScaler | RobustScaler:
    """Cria o scaler de entrada.

    - ``standard``: subtrai a média e divide pelo desvio padrão. Média e desvio são puxados
      por valores extremos.
    - ``robust``: subtrai a mediana e divide pelo intervalo interquartil (IQR). Mediana e IQR
      ignoram as caudas, então valores extremos não mudam a escala dos demais.

    Args:
        name: Qual scaler criar.

    Returns:
        Scaler não ajustado.
    """
    scalers: dict[str, StandardScaler | RobustScaler] = {"standard": StandardScaler(), "robust": RobustScaler()}
    return scalers[name]


def make_reducer(name: ReducerName, n_components: int, random_state: int) -> PCA | SelectKBest:
    """Cria a etapa que reduz as features ao número de qubits.

    - ``pca``: combina todas as features em ``n_components`` eixos de máxima variância. É não
      supervisionado: não sabe o que é fraude.
    - ``mutual_info``: mantém as ``n_components`` features com maior informação mútua com o
      rótulo. É supervisionado, mas avalia cada feature isoladamente (duas features
      redundantes podem ser escolhidas juntas).

    Args:
        name: Qual redutor criar.
        n_components: Número de colunas de saída (= qubits na codificação de ângulo).
        random_state: Semente (o estimador de informação mútua é aleatório).

    Returns:
        Redutor não ajustado.
    """
    if name == "pca":
        return PCA(n_components=n_components, random_state=random_state)
    # partial fixa a semente sem perder a compatibilidade com clone/pickle
    return SelectKBest(partial(mutual_info_classif, random_state=random_state), k=n_components)


def build_angle_preprocessor(
    scaler: ScalerName = "standard",
    reducer: ReducerName = "pca",
    n_components: int = 5,
    angle_range: AngleRange = DEFAULT_ANGLE_RANGE,
    random_state: int = 0,
) -> Pipeline:
    """Monta scaler → redutor → MinMax para a codificação de ângulo.

    A ordem importa. O scaler vem antes do redutor porque o PCA maximiza variância: sem
    escalonar, a feature de maior escala dominaria os componentes. O MinMax vem depois porque
    precisa limitar os valores **que viram ângulos**, e esses são as saídas do redutor.

    ``clip=True`` é essencial: ângulos são periódicos. Uma transação da validação mais extrema
    que todas do treino ultrapassaria o intervalo e "daria a volta" no círculo, ficando parecida
    com o extremo oposto. O clip a mantém na borda.

    Args:
        scaler: Scaler de entrada.
        reducer: Redutor de dimensionalidade.
        n_components: Número de features de saída (= qubits).
        angle_range: Intervalo dos ângulos; funciona como largura de banda do kernel.
        random_state: Semente do redutor.

    Returns:
        Pipeline não ajustado.
    """
    return Pipeline(
        [
            ("scaler", make_scaler(scaler)),
            ("reducer", make_reducer(reducer, n_components, random_state)),
            ("angle_scaler", MinMaxScaler(feature_range=angle_range, clip=True)),
        ]
    )


class AmplitudeEncodingPreparer(TransformerMixin, BaseEstimator):
    """Completa cada linha até 2ⁿ valores e normaliza (norma L2 = 1) para a codificação de amplitude.

    Um estado quântico de n qubits tem 2ⁿ amplitudes cuja soma dos quadrados é 1. A
    normalização descarta a **norma** do vetor original: x e 2x viram o mesmo estado. Como
    fraudes tendem a ficar longe do centro dos dados (norma grande), isso apagaria sinal útil.
    Completar com uma constante diferente de zero preserva essa informação: o ângulo entre o
    vetor e o eixo da constante depende da norma de x.

    Args:
        n_qubits: Número de qubits; a saída tem ``2 ** n_qubits`` colunas.
        padding_value: Valor das posições completadas. Com 0 (completar com zeros, a prática
            mais comum), a norma original se perde.
    """

    def __init__(self, n_qubits: int = 5, padding_value: float = 1.0) -> None:
        self.n_qubits = n_qubits
        self.padding_value = padding_value

    def fit(self, features: ArrayLike, labels: ArrayLike | None = None) -> Self:
        array = check_array(features)
        n_amplitudes = 2**self.n_qubits
        if array.shape[1] > n_amplitudes:
            raise ValueError(f"{array.shape[1]} features não cabem em {self.n_qubits} qubits ({n_amplitudes} amplitudes).")
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, features: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "n_features_in_")
        array = check_array(features)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(f"esperadas {self.n_features_in_} features, recebidas {array.shape[1]}")
        n_padding = 2**self.n_qubits - self.n_features_in_
        padded = np.hstack([array, np.full((array.shape[0], n_padding), self.padding_value)])
        norms = np.linalg.norm(padded, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("linha com norma zero não corresponde a nenhum estado quântico")
        return padded / norms


def build_amplitude_preprocessor(
    scaler: ScalerName = "standard", n_qubits: int = 5, padding_value: float = 1.0
) -> Pipeline:
    """Monta scaler → preparo de amplitude (sem redução: todas as features cabem em 2ⁿ).

    Args:
        scaler: Scaler de entrada; iguala as escalas antes que a normalização misture as features.
        n_qubits: Número de qubits.
        padding_value: Constante usada para completar até 2ⁿ.

    Returns:
        Pipeline não ajustado.
    """
    return Pipeline(
        [
            ("scaler", make_scaler(scaler)),
            ("amplitude", AmplitudeEncodingPreparer(n_qubits=n_qubits, padding_value=padding_value)),
        ]
    )


def make_undersampler(random_state: int) -> RandomUnderSampler:
    """Cria o undersampler que iguala as classes descartando legítimas ao acaso.

    Só age durante o ``fit`` de um ``imblearn.pipeline.Pipeline``. Na predição ele é pulado,
    então validação e teste são sempre avaliados com a proporção real de fraudes.

    Args:
        random_state: Semente do sorteio (a análise de robustez varia esta semente).

    Returns:
        Undersampler não ajustado.
    """
    return RandomUnderSampler(sampling_strategy=BALANCED_SAMPLING_STRATEGY, replacement=False, random_state=random_state)
