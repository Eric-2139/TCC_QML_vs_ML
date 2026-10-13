"""Fábrica dos pipelines dos quatro modelos e a medição de um ajuste.

Todo pipeline começa pelo undersampling e termina no modelo (etapa ``model``). O que fica no meio
tem sempre os mesmos nomes de etapa, para que o grid da Seção 6 separe as decisões em três grupos:

- **entrada:** ``scaler``, ``reducer``, ``angle_scaler`` ou ``amplitude``;
- **codificação** (só os quânticos): ``encoder``;
- **modelo:** ``model``.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import check_random_state

from fraud_qml import config
from fraud_qml.evaluation import fraud_scores
from fraud_qml.preprocessing import (
    DEFAULT_ANGLE_RANGE,
    AngleRange,
    ReducerName,
    ScalerName,
    build_amplitude_preprocessor,
    build_angle_preprocessor,
    make_scaler,
    make_undersampler,
)
from fraud_qml.profiling import CostMeter, CostReport
from fraud_qml.quantum.encodings import (
    AMPLITUDE_ENCODING,
    DEFAULT_FEATURE_MAP_REPS,
    AngleFeatureMapName,
    Entanglement,
    StatevectorEncoder,
)
from fraud_qml.quantum.kernels import QuantumKernelSVC
from fraud_qml.quantum.variational import (
    DEFAULT_ANSATZ_REPS,
    DEFAULT_MAX_ITERATIONS,
    VariationalQuantumClassifier,
)

QuantumVariation = Literal["angle", "amplitude"]
ClassicalInput = Literal["all_features", "angle_components"]

# Padrões do scikit-learn, escritos aqui para que a Seção 5 ("hiperparâmetros padrão") não dependa
# de um valor implícito que mude entre versões da biblioteca.
DEFAULT_SVM_C: Final[float] = 1.0
DEFAULT_SVM_GAMMA: Final[str] = "scale"
DEFAULT_N_ESTIMATORS: Final[int] = 50
DEFAULT_LEARNING_RATE: Final[float] = 1.0
DEFAULT_AMPLITUDE_PADDING: Final[float] = 1.0

Step = tuple[str, BaseEstimator]

# O AdaBoost do scikit-learn sorteia a semente de cada classificador fraco como um inteiro abaixo deste valor
_MAX_SEED: Final[int] = int(np.iinfo(np.int32).max)


def _undersampling_step(random_state: int) -> Step:
    return ("undersampler", make_undersampler(random_state))


def _angle_steps(scaler: ScalerName, reducer: ReducerName, angle_range: AngleRange, random_state: int) -> list[Step]:
    return list(build_angle_preprocessor(scaler, reducer, config.N_QUBITS, angle_range, random_state).steps)


def _classical_input_steps(model_input: ClassicalInput, scaler: ScalerName, random_state: int) -> list[Step]:
    if model_input == "angle_components":
        # Entrada idêntica à da variação A dos quânticos: os mesmos 5 valores que viram ângulos
        return _angle_steps(scaler, "pca", DEFAULT_ANGLE_RANGE, random_state)
    return [("scaler", make_scaler(scaler))]


def quantum_encoding_steps(
    variation: QuantumVariation = "angle",
    scaler: ScalerName = "standard",
    reducer: ReducerName = "pca",
    angle_range: AngleRange = DEFAULT_ANGLE_RANGE,
    feature_map: AngleFeatureMapName = "zz",
    feature_map_reps: int = DEFAULT_FEATURE_MAP_REPS,
    entanglement: Entanglement = "linear",
    padding_value: float = DEFAULT_AMPLITUDE_PADDING,
    random_state: int = config.RANDOM_STATE,
) -> list[Step]:
    """Etapas de entrada e de codificação dos modelos quânticos, até o estado de cada linha.

    - **Variação A (``angle``):** scaler → redutor para ``N_QUBITS`` componentes → MinMax na faixa
      de ângulos → feature map (uma feature por qubit).
    - **Variação B (``amplitude``):** scaler → completar até 2ⁿ com uma constante e normalizar. O
      vetor resultante já é o estado; não há feature map.

    Args:
        variation: Qual das duas variações de codificação.
        scaler: Scaler de entrada.
        reducer: Redutor de dimensionalidade (só na variação A).
        angle_range: Faixa dos ângulos (só na variação A).
        feature_map: Mapa de rotação (só na variação A).
        feature_map_reps: Repetições do feature map (só na variação A).
        entanglement: Pares acoplados no mapa ``zz`` (só na variação A).
        padding_value: Constante de preenchimento (só na variação B).
        random_state: Semente do redutor.

    Returns:
        Lista de etapas ``(nome, transformador)``, para compor um pipeline.
    """
    if variation == "amplitude":
        return [
            *build_amplitude_preprocessor(scaler, config.N_QUBITS, padding_value).steps,
            ("encoder", StatevectorEncoder(encoding=AMPLITUDE_ENCODING)),
        ]
    return [
        *_angle_steps(scaler, reducer, angle_range, random_state),
        ("encoder", StatevectorEncoder(feature_map, feature_map_reps, entanglement)),
    ]


def build_dummy_pipeline(random_state: int = config.RANDOM_STATE) -> ImbPipeline:
    """Linha de base sem modelo: prevê a proporção de classes do treino balanceado e ignora as features."""
    return ImbPipeline([_undersampling_step(random_state), ("model", DummyClassifier(strategy="prior"))])


def build_svm_pipeline(
    model_input: ClassicalInput = "all_features",
    scaler: ScalerName = "standard",
    C: float = DEFAULT_SVM_C,  # noqa: N803  (mesmo nome do parâmetro no scikit-learn)
    gamma: str | float = DEFAULT_SVM_GAMMA,
    random_state: int = config.RANDOM_STATE,
) -> ImbPipeline:
    """SVM clássico com kernel RBF.

    Args:
        model_input: ``all_features`` (referência com as 31 features) ou ``angle_components`` (a
            mesma entrada da variação A dos quânticos).
        scaler: Scaler de entrada.
        C: Penalidade por erro de classificação no treino.
        gamma: Largura de banda do RBF.
        random_state: Semente do undersampling e do redutor.

    Returns:
        Pipeline não ajustado.
    """
    return ImbPipeline(
        [
            _undersampling_step(random_state),
            *_classical_input_steps(model_input, scaler, random_state),
            ("model", SVC(kernel="rbf", C=C, gamma=gamma)),
        ]
    )


def build_adaboost_pipeline(
    model_input: ClassicalInput = "all_features",
    scaler: ScalerName = "standard",
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    random_state: int = config.RANDOM_STATE,
) -> ImbPipeline:
    """AdaBoost clássico; o classificador fraco é o padrão do scikit-learn (árvore de um corte, *stump*).

    Args:
        model_input: ``all_features`` ou ``angle_components``, como no SVM.
        scaler: Scaler de entrada. Com todas as features ele não muda nada (um *stump* só compara
            cada feature com um limiar); fica no pipeline para a entrada ser a mesma do SVM.
        n_estimators: Número máximo de classificadores fracos (rodadas de boosting).
        learning_rate: Fator que encolhe o peso de cada classificador fraco no voto.
        random_state: Semente do undersampling, do redutor e do boosting.

    Returns:
        Pipeline não ajustado.
    """
    return ImbPipeline(
        [
            _undersampling_step(random_state),
            *_classical_input_steps(model_input, scaler, random_state),
            ("model", AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)),
        ]
    )


def build_qsvm_pipeline(
    encoding_steps: list[Step],
    C: float = DEFAULT_SVM_C,  # noqa: N803  (mesmo nome do parâmetro no scikit-learn)
    random_state: int = config.RANDOM_STATE,
) -> ImbPipeline:
    """QSVM: SVM sobre o kernel de fidelidade dos estados codificados.

    Args:
        encoding_steps: Saída de ``quantum_encoding_steps``.
        C: Penalidade por erro de classificação no treino.
        random_state: Semente do undersampling.

    Returns:
        Pipeline não ajustado.
    """
    return ImbPipeline([_undersampling_step(random_state), *encoding_steps, ("model", QuantumKernelSVC(C=C))])


def build_quantum_adaboost_pipeline(
    encoding_steps: list[Step],
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    ansatz_reps: int = DEFAULT_ANSATZ_REPS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    random_state: int = config.RANDOM_STATE,
) -> ImbPipeline:
    """Quantum AdaBoost: o mesmo ``AdaBoostClassifier`` do clássico, com classificador fraco variacional.

    Args:
        encoding_steps: Saída de ``quantum_encoding_steps``.
        n_estimators: Número máximo de classificadores fracos (rodadas de boosting).
        learning_rate: Fator que encolhe o peso de cada classificador fraco no voto.
        ansatz_reps: Repetições do bloco do ansatz de cada classificador fraco.
        max_iterations: Avaliações da função de custo por classificador fraco (COBYLA).
        random_state: Semente do undersampling e do boosting (que sorteia os pesos iniciais de cada rodada).

    Returns:
        Pipeline não ajustado.
    """
    weak_learner = VariationalQuantumClassifier(ansatz_reps=ansatz_reps, max_iterations=max_iterations)
    booster = AdaBoostClassifier(
        estimator=weak_learner, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state
    )
    return ImbPipeline([_undersampling_step(random_state), *encoding_steps, ("model", booster)])


def build_weak_quantum_classifier_pipeline(
    encoding_steps: list[Step],
    ansatz_reps: int = DEFAULT_ANSATZ_REPS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    random_state: int = config.RANDOM_STATE,
) -> Pipeline:
    """Um classificador variacional sozinho, sem boosting e **sem undersampling**.

    Serve aos diagnósticos feitos dentro do treino já balanceado (por exemplo, comparar codificações
    no classificador fraco), em que o sorteio do undersampling já aconteceu.

    Args:
        encoding_steps: Saída de ``quantum_encoding_steps``.
        ansatz_reps: Repetições do bloco do ansatz.
        max_iterations: Avaliações da função de custo (COBYLA).
        random_state: Semente dos pesos iniciais.

    Returns:
        Pipeline não ajustado.
    """
    model = VariationalQuantumClassifier(ansatz_reps=ansatz_reps, max_iterations=max_iterations, random_state=random_state)
    return Pipeline([*encoding_steps, ("model", model)])


@dataclass(frozen=True)
class BoostingReplay:
    """O que o AdaBoost fez em cada rodada, refeito a partir do modelo ajustado.

    Attributes:
        errors: Erro ponderado (ε) de cada classificador fraco mantido, nos pesos da rodada dele.
        next_sample_weight: Pesos das linhas que a rodada seguinte à última mantida recebeu.
        next_error: ε do classificador fraco da rodada seguinte, treinado nesses pesos com a semente
            que o boosting lhe daria. Quando o boosting parou antes das rodadas pedidas, é o
            classificador que o scikit-learn descartou.
    """

    errors: NDArray[np.float64]
    next_sample_weight: NDArray[np.float64]
    next_error: float


def replay_boosting(booster: AdaBoostClassifier, features: ArrayLike, labels: ArrayLike) -> BoostingReplay:
    """Refaz a reponderação de um ``AdaBoostClassifier`` ajustado e treina o classificador da rodada seguinte.

    O scikit-learn não guarda o classificador que fez o boosting parar (erro ponderado ≥ 0,5): ele é
    descartado. Para vê-lo, a função repete as contas do boosting com os classificadores mantidos
    (peso das linhas erradas multiplicado por e^α, depois normalização) e treina mais um, com a
    semente seguinte da mesma sequência.

    Args:
        booster: AdaBoost ajustado.
        features: As mesmas linhas de treino que o booster recebeu (já transformadas pelo pipeline).
        labels: Os rótulos dessas linhas.

    Returns:
        Os erros refeitos, os pesos da rodada seguinte e o erro do classificador dela.

    Raises:
        ValueError: Se as sementes dos classificadores mantidos não forem as que esta função refaz
            (o scikit-learn mudou a forma de sorteá-las, e o classificador seguinte não seria o mesmo).
    """
    targets = np.asarray(labels)
    seed_sequence = check_random_state(booster.random_state)
    smallest_weight = np.finfo(np.float64).eps
    sample_weight = np.full(len(targets), 1.0 / len(targets))
    errors: list[float] = []
    for estimator, vote_weight in zip(booster.estimators_, booster.estimator_weights_, strict=False):
        if estimator.get_params()["random_state"] != seed_sequence.randint(_MAX_SEED):
            raise ValueError("as sementes dos classificadores fracos não seguem a sequência esperada")
        # O scikit-learn impede pesos menores que o épsilon da máquina antes de cada rodada
        sample_weight = np.clip(sample_weight, smallest_weight, None)
        incorrect = estimator.predict(features) != targets
        errors.append(float(np.average(incorrect, weights=sample_weight)))
        # As mesmas operações do scikit-learn, na mesma ordem: o otimizador do classificador variacional
        # amplifica até diferenças de arredondamento nos pesos, e o classificador seguinte sairia outro
        sample_weight = np.exp(np.log(sample_weight) + vote_weight * incorrect * (sample_weight > 0))
        sample_weight /= np.sum(sample_weight)
    sample_weight = np.clip(sample_weight, smallest_weight, None)
    next_estimator = clone(booster.estimator_).set_params(random_state=seed_sequence.randint(_MAX_SEED))
    next_estimator.fit(features, targets, sample_weight=sample_weight)
    next_error = float(np.average(next_estimator.predict(features) != targets, weights=sample_weight))
    return BoostingReplay(errors=np.asarray(errors), next_sample_weight=sample_weight, next_error=next_error)


@dataclass(frozen=True)
class ScoredFit:
    """Resultado de ajustar um pipeline e pontuar um conjunto de avaliação.

    Attributes:
        scores: Pontuação de fraude de cada linha avaliada.
        predictions: Classe prevista para cada linha avaliada (limiar padrão do modelo).
        fit_cost: Custo do ajuste (undersampling, pré-processamento, codificação e treino).
        predict_cost: Custo de pontuar o conjunto de avaliação (pré-processamento, codificação e modelo).
    """

    scores: NDArray[np.float64]
    predictions: NDArray[np.int_]
    fit_cost: CostReport
    predict_cost: CostReport


def fit_and_score(
    pipeline: ImbPipeline,
    train_features: ArrayLike,
    train_labels: ArrayLike,
    evaluation_features: ArrayLike,
    trace_python_memory: bool = False,
) -> ScoredFit:
    """Ajusta o pipeline no treino e pontua o conjunto de avaliação, medindo o custo de cada parte.

    As classes previstas saem das mesmas linhas já transformadas, fora da medição: transformar o
    conjunto de avaliação duas vezes (uma para a pontuação, outra para a classe) contaria em dobro
    a parte mais cara dos modelos quânticos, a simulação dos estados.

    Args:
        pipeline: Pipeline não ajustado (é ajustado no lugar).
        train_features: Features de treino.
        train_labels: Rótulos de treino.
        evaluation_features: Features do conjunto de avaliação.
        trace_python_memory: Liga o ``tracemalloc``; deixe desligado para medir tempo.

    Returns:
        Pontuações, classes previstas e os dois custos.
    """
    with CostMeter(trace_python_memory=trace_python_memory) as fit_meter:
        pipeline.fit(train_features, train_labels)
    model = pipeline.named_steps["model"]
    with CostMeter(trace_python_memory=trace_python_memory) as predict_meter:
        transformed = evaluation_features
        for _, step in pipeline.steps[:-1]:
            # O undersampler só age no ajuste: na predição nenhuma linha é descartada
            if not hasattr(step, "fit_resample"):
                transformed = step.transform(transformed)
        scores = fraud_scores(model, transformed)
    return ScoredFit(
        scores=scores,
        predictions=np.asarray(model.predict(transformed)),
        fit_cost=fit_meter.report,
        predict_cost=predict_meter.report,
    )
