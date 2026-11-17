"""Curvas de diagnóstico da Seção 8: de aprendizado (tamanho do treino) e de validação (um hiperparâmetro por vez).

Cada ponto de curva tem duas notas de **avaliação** (PR-AUC e ROC-AUC nas linhas de avaliação, na proporção real de
fraudes) e duas de **treino**:

- **PR-AUC no conjunto de ajuste:** em todas as linhas passadas ao ``fit`` (a definição do scikit-learn). As fraudes
  são todas vistas pelo modelo; das legítimas, o undersampling usa só uma em cada ~600. A proporção é a real, então a
  nota se compara com a de avaliação, e a diferença mede o quanto o modelo ordena melhor as fraudes que viu do que as
  que não viu (as legítimas, quase todas novas, estão nos dois lados).
- **ROC-AUC nas linhas vistas:** as do undersampling, metade fraude. A ROC-AUC não depende da proporção.

**A PR-AUC ponderada nas linhas vistas não serve como nota de treino** (``prevalence_weights``): com 284 legítimas
representando 169.951, cada uma pesa cerca de 598, e uma única legítima mal ordenada vale 598 alarmes falsos. A nota
salta conforme haja zero ou uma legítima no topo; a curva de aprendizado a calcula só para mostrar isso.

**Por que não o ``learning_curve`` e o ``validation_curve`` do scikit-learn.** O ``learning_curve`` só cresce até o
tamanho do fold de ajuste (456 linhas depois do undersampling), e não até as 568 do modelo da Seção 7; nenhum dos dois
dá a ROC-AUC nas linhas que o undersampling sorteou; e o ``validation_curve`` ajustaria um AdaBoost por teto de
rodadas, em vez de ler as rodadas de um único ajuste.

Este módulo não é importado pelo ``tuning``: mudá-lo não invalida o cache das buscas da Seção 6.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from fraud_qml import config
from fraud_qml.data import FRAUD_LABEL
from fraud_qml.evaluation import fraud_scores, weighted_average_precision
from fraud_qml.models import InputEncoding, ModelSettings, build_pipeline
from fraud_qml.preprocessing import make_undersampler
from fraud_qml.profiling import CostMeter
from fraud_qml.tuning import CONFIGURATION_COLUMNS, cv_folds, describe_input, describe_model

# Frações das fraudes do treino na curva de aprendizado. 0,8 é o tamanho de um fold de ajuste da validação cruzada
# (227 ou 228 das 284 fraudes) e 1,0 é o modelo da Seção 7: os dois pontos medidos no mesmo conjunto de avaliação
# isolam o efeito do tamanho do treino (a hipótese da Seção 7.2)
LEARNING_CURVE_FRACTIONS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75, 0.8, 1.0)

# Valores de cada curva de validação: todos os do grid da Seção 6 e um passo além de cada borda que sobrou depois
# da segunda rodada (CC19). Nenhum deles muda a configuração escolhida
CURVE_VALUES: Final[dict[str, tuple[float, ...]]] = {
    "C": (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    "gamma": (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
    "range_width": (0.05, 0.1, 0.25, 0.5, 1.0, math.pi / 2),
    "padding_value": (0.5, 1.0, 3.0, 5.0, 10.0, 20.0),
    "learning_rate": (0.02, 0.05, 0.1, 0.5, 1.0, 2.0),
}
CEILING_PARAMETER: Final[str] = "n_estimators"
# Teto lido rodada a rodada nos dois AdaBoost: o maior da segunda rodada da Seção 6
CURVE_MAX_ROUNDS: Final[int] = 200
# Rodadas em que a curva do teto é lida (espaçadas para um eixo em escala log; incluem os tetos do grid)
CURVE_STAGES: Final[tuple[int, ...]] = (1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200)
_INPUT_PARAMETERS: Final[frozenset[str]] = frozenset({"range_width", "padding_value"})

METRIC_COLUMNS: Final[tuple[str, ...]] = ("pr_auc", "roc_auc", "train_pr_auc", "train_roc_auc")
LEARNING_CURVE_COLUMNS: Final[tuple[str, ...]] = (
    "pipeline",
    "fraction",
    "seed",
    "frauds_seen",
    "rows_seen",
    *METRIC_COLUMNS,
    "seen_weighted_pr_auc",
    "kept_estimators",
    "failed",
    "fit_seconds",
    "score_seconds",
)
VALIDATION_CURVE_COLUMNS: Final[tuple[str, ...]] = (*CONFIGURATION_COLUMNS, "fold", *METRIC_COLUMNS, "kept_estimators", "failed")

# Mesma mensagem que o ``tuning`` reconhece: o primeiro classificador fraco já é pior que o acaso, e o
# AdaBoostClassifier não consegue ajustar nada
_WORSE_THAN_RANDOM_MESSAGE: Final[str] = "worse than random"


# ---------------------------------------------------------------- a nota no treino


def prevalence_weights(sampled_labels: ArrayLike, pool_labels: ArrayLike) -> NDArray[np.float64]:
    """Peso de cada linha sorteada para que a proporção ponderada das classes volte à do conjunto de onde ela saiu.

    O peso de uma linha da classe c é (linhas de c no conjunto de origem) ÷ (linhas de c sorteadas). Com o
    undersampling, as fraudes pesam 1 (todas são sorteadas) e cada legítima vale as centenas que ela representa.

    Args:
        sampled_labels: Rótulos das linhas sorteadas (as que o modelo viu).
        pool_labels: Rótulos do conjunto de onde elas foram sorteadas.

    Returns:
        Um peso por linha sorteada.

    Raises:
        ValueError: Se uma classe sorteada não existir no conjunto de origem.
    """
    sampled = np.asarray(sampled_labels)
    pool = np.asarray(pool_labels)
    weights = np.empty(len(sampled), dtype=float)
    for label in np.unique(sampled):
        in_sample = sampled == label
        in_pool = int(np.sum(pool == label))
        if in_pool == 0:
            raise ValueError(f"a classe {label} foi sorteada, mas não existe no conjunto de origem")
        weights[in_sample] = in_pool / int(in_sample.sum())
    return weights


def training_metrics(
    seen_labels: ArrayLike, seen_scores: ArrayLike, pool_labels: ArrayLike, pool_scores: ArrayLike
) -> dict[str, float]:
    """As duas notas de treino: PR-AUC no conjunto de ajuste inteiro e ROC-AUC nas linhas vistas.

    Args:
        seen_labels: Rótulos das linhas que o modelo viu (as do undersampling).
        seen_scores: Pontuação do modelo nelas.
        pool_labels: Rótulos de todas as linhas passadas ao ``fit``.
        pool_scores: Pontuação do modelo nelas.

    Returns:
        ``train_pr_auc`` e ``train_roc_auc``.
    """
    return {
        "train_pr_auc": float(average_precision_score(pool_labels, pool_scores, pos_label=FRAUD_LABEL)),
        "train_roc_auc": float(roc_auc_score(seen_labels, seen_scores)),
    }


def _evaluation_metrics(labels: ArrayLike, scores: ArrayLike) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(labels, scores, pos_label=FRAUD_LABEL)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def _is_worse_than_random(error: ValueError) -> bool:
    return _WORSE_THAN_RANDOM_MESSAGE in str(error)


# ---------------------------------------------------------------- curva de aprendizado


def nested_fraction_rows(labels: ArrayLike, fraction: float, random_state: int) -> NDArray[np.intp]:
    """Posições de uma fração estratificada das linhas, aninhada entre frações da mesma semente.

    Cada classe é embaralhada uma vez pela semente, e a fração pega as primeiras linhas desse embaralhamento: com a
    mesma semente, as fraudes de 25% estão dentro das de 50%, e assim por diante. Com 1,0, devolve todas as linhas
    na ordem original, para que o ponto final seja exatamente o modelo treinado no treino inteiro.

    Args:
        labels: Rótulos de todas as linhas.
        fraction: Fração de cada classe, em (0, 1].
        random_state: Semente do embaralhamento.

    Returns:
        Posições em ordem crescente.

    Raises:
        ValueError: Se a fração estiver fora de (0, 1].
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"a fração precisa estar em (0, 1], recebida {fraction}")
    target = np.asarray(labels)
    if fraction == 1.0:
        return np.arange(len(target))
    rng = np.random.default_rng(random_state)
    chosen: list[NDArray[np.intp]] = []
    for label in np.unique(target):
        positions = rng.permutation(np.flatnonzero(target == label))
        chosen.append(positions[: round(fraction * len(positions))])
    return np.sort(np.concatenate(chosen))


def _take_rows(data: ArrayLike, rows: NDArray[np.intp]) -> ArrayLike:
    """Linhas por posição, preservando o tipo (tabela do pandas ou array).

    O tipo importa no Quantum AdaBoost: a mesma conta feita sobre uma tabela ou sobre um array muda o arredondamento,
    e o COBYLA amplifica a diferença até outro modelo (Seção 8.1). Para o ponto final reproduzir a Seção 7, que
    ajustou sobre as tabelas, as linhas são tiradas sem converter.
    """
    return data.iloc[rows] if hasattr(data, "iloc") else np.asarray(data)[rows]


def _kept_estimators(model: BaseEstimator) -> float:
    return float(len(model.estimators_)) if isinstance(model, AdaBoostClassifier) else np.nan


def learning_curve_point(
    input_encoding: InputEncoding,
    model: ModelSettings,
    fraction: float,
    seed: int,
    train_features: ArrayLike,
    train_labels: ArrayLike,
    evaluation_features: ArrayLike,
    evaluation_labels: ArrayLike,
) -> dict[str, object]:
    """Um ponto da curva de aprendizado: o pipeline ajustado numa fração do treino e pontuado nos dois lados.

    A semente vale para tudo o que é sorteado: a fração, o undersampling, o redutor e o boosting. Com a fração 1,0, a
    semente do experimento e as tabelas da Seção 7, o pipeline é o da Seção 7.

    Args:
        input_encoding: Entrada e codificação (congeladas na Seção 6).
        model: Modelo (congelado na Seção 6).
        fraction: Fração de cada classe do treino.
        seed: Semente.
        train_features: Treino inteiro (tabela ou array).
        train_labels: Rótulos do treino.
        evaluation_features: Conjunto de avaliação.
        evaluation_labels: Rótulos do conjunto de avaliação.

    Returns:
        Fração, semente, fraudes e linhas vistas, as métricas, classificadores mantidos, falha e tempos.
    """
    labels = np.asarray(train_labels)
    rows = nested_fraction_rows(labels, fraction, seed)
    pool_features = train_features if fraction == 1.0 else _take_rows(train_features, rows)
    pool_labels = labels[rows]
    pool_target = train_labels if fraction == 1.0 else _take_rows(train_labels, rows)
    pipeline = build_pipeline(input_encoding, model, seed)
    point: dict[str, object] = {"fraction": fraction, "seed": seed, "frauds_seen": int(np.sum(pool_labels == FRAUD_LABEL))}
    with CostMeter(trace_python_memory=False) as fit_meter:
        try:
            pipeline.fit(pool_features, pool_target)
        except ValueError as error:
            if not _is_worse_than_random(error):
                raise
            return {**point, "rows_seen": np.nan, **dict.fromkeys(METRIC_COLUMNS, np.nan), "seen_weighted_pr_auc": np.nan,
                    "kept_estimators": 0.0, "failed": True, "fit_seconds": np.nan, "score_seconds": np.nan}
    seen = pipeline.named_steps["undersampler"].sample_indices_
    seen_labels = pool_labels[seen]
    with CostMeter(trace_python_memory=False) as score_meter:
        evaluation_scores = np.asarray(pipeline.decision_function(evaluation_features), dtype=float)
    pool_scores = np.asarray(pipeline.decision_function(pool_features), dtype=float)
    seen_scores = pool_scores[seen]
    return {
        **point,
        "frauds_seen": int(np.sum(seen_labels == FRAUD_LABEL)),
        "rows_seen": len(seen),
        **_evaluation_metrics(evaluation_labels, evaluation_scores),
        **training_metrics(seen_labels, seen_scores, pool_labels, pool_scores),
        "seen_weighted_pr_auc": weighted_average_precision(seen_labels, seen_scores, prevalence_weights(seen_labels, pool_labels)),
        "kept_estimators": _kept_estimators(pipeline.named_steps["model"]),
        "failed": False,
        "fit_seconds": fit_meter.report.wall_seconds,
        "score_seconds": score_meter.report.wall_seconds,
    }


def learning_curve_scores(
    configurations: Mapping[str, tuple[InputEncoding, ModelSettings]],
    fractions: Sequence[float],
    seeds: Sequence[int],
    train_features: ArrayLike,
    train_labels: ArrayLike,
    evaluation_features: ArrayLike,
    evaluation_labels: ArrayLike,
    n_jobs: int = config.N_JOBS,
) -> pd.DataFrame:
    """Curva de aprendizado de cada configuração: cada fração × cada semente, em paralelo.

    Args:
        configurations: Nome do pipeline → (entrada, modelo).
        fractions: Frações das fraudes (e das legítimas) do treino.
        seeds: Sementes; cada uma sorteia a fração, o undersampling, o redutor e o boosting.
        train_features: Treino inteiro (tabela ou array; o tipo é preservado).
        train_labels: Rótulos do treino.
        evaluation_features: Conjunto de avaliação.
        evaluation_labels: Rótulos dele.
        n_jobs: Processos paralelos; o resultado não depende deste valor.

    Returns:
        Uma linha por pipeline, fração e semente, com as colunas de ``LEARNING_CURVE_COLUMNS``.
    """
    tasks = [(name, fraction, seed) for name in configurations for fraction in fractions for seed in seeds]
    points = Parallel(n_jobs=n_jobs)(
        delayed(learning_curve_point)(
            *configurations[name], fraction, seed, train_features, train_labels, evaluation_features, evaluation_labels
        )
        for name, fraction, seed in tasks
    )
    rows = [{"pipeline": name, **point} for (name, _, _), point in zip(tasks, points, strict=True)]
    return pd.DataFrame(rows, columns=list(LEARNING_CURVE_COLUMNS))


# ---------------------------------------------------------------- curvas de validação


def curve_parameters(input_encoding: InputEncoding, model: ModelSettings) -> tuple[str, ...]:
    """Os hiperparâmetros que ganham uma curva de validação em cada pipeline.

    São os numéricos da Seção 6: C e γ no SVM; C e a largura de banda da codificação no QSVM (o intervalo do MinMax
    na variação A, a constante na B); a taxa de aprendizado e o teto de rodadas nos dois AdaBoost, e a largura de
    banda no Quantum AdaBoost.

    Args:
        input_encoding: Entrada e codificação.
        model: Modelo.

    Returns:
        Nomes dos campos de ``InputEncoding`` ou ``ModelSettings``.
    """
    bandwidth = "range_width" if input_encoding.kind == "angle" else "padding_value"
    by_family: dict[str, tuple[str, ...]] = {
        "svm": ("C", "gamma"),
        "qsvm": ("C", bandwidth),
        "adaboost": ("learning_rate", CEILING_PARAMETER),
        "quantum_adaboost": ("learning_rate", CEILING_PARAMETER, bandwidth),
    }
    return by_family[model.family]


@dataclass(frozen=True)
class CurvePoint:
    """Uma configuração ajustada para uma curva de validação.

    Attributes:
        parameter: O hiperparâmetro que a curva varia.
        value: O valor dele neste ponto (no teto de rodadas, o maior; os menores são lidos rodada a rodada).
        input_encoding: Entrada e codificação deste ponto.
        model: Modelo deste ponto.
        stages: Rodadas lidas (só no ponto do teto; nos outros, só o teto do modelo).
    """

    parameter: str
    value: float
    input_encoding: InputEncoding
    model: ModelSettings
    stages: tuple[int, ...] = ()


def curve_points(
    input_encoding: InputEncoding, model: ModelSettings, max_rounds: int = CURVE_MAX_ROUNDS,
    stages: Sequence[int] = CURVE_STAGES,
) -> list[CurvePoint]:
    """Os pontos das curvas de validação de um pipeline: um hiperparâmetro por vez, os outros no valor escolhido.

    Nos AdaBoost, a curva do teto é um único ajuste com ``max_rounds`` rodadas, lido nas rodadas ``stages``
    (é o mesmo que ajustar com cada teto, Seção 6.1.2). Os pontos das outras curvas usam o teto escolhido.

    Args:
        input_encoding: Entrada e codificação escolhidas.
        model: Modelo escolhido.
        max_rounds: Teto do ajuste da curva do teto.
        stages: Rodadas lidas na curva do teto (as acima de ``max_rounds`` são ignoradas).

    Returns:
        Os pontos; o da configuração escolhida aparece uma vez por curva.

    Raises:
        ValueError: Se o valor escolhido de um hiperparâmetro não estiver entre os valores da curva (a curva não
            passaria pela configuração escolhida), ou se o teto escolhido não estiver entre as rodadas lidas.
    """
    points: list[CurvePoint] = []
    for parameter in curve_parameters(input_encoding, model):
        if parameter == CEILING_PARAMETER:
            read = tuple(sorted({stage for stage in stages if stage <= max_rounds} | {max_rounds}))
            if model.n_estimators not in read:
                raise ValueError(f"o teto escolhido ({model.n_estimators}) não está entre as rodadas lidas {read}")
            points.append(CurvePoint(parameter, float(max_rounds), input_encoding, replace(model, n_estimators=max_rounds), read))
            continue
        chosen = getattr(input_encoding if parameter in _INPUT_PARAMETERS else model, parameter)
        values = CURVE_VALUES[parameter]
        if not any(math.isclose(float(chosen), value) for value in values):
            raise ValueError(f"o valor escolhido de {parameter} ({chosen}) não está na curva {values}")
        for value in values:
            if parameter in _INPUT_PARAMETERS:
                points.append(CurvePoint(parameter, value, replace(input_encoding, **{parameter: value}), model))
            else:
                points.append(CurvePoint(parameter, value, input_encoding, replace(model, **{parameter: value})))
    return points


def _scored_rows(
    model: ModelSettings,
    stage_scores: Sequence[tuple[int, NDArray[np.float64], NDArray[np.float64]]],
    seen: NDArray[np.intp],
    pool_labels: NDArray[np.int_],
    evaluation_labels: NDArray[np.int_],
    kept: float,
) -> list[dict[str, object]]:
    """Uma linha por rodada lida (ou uma só, fora dos AdaBoost): as notas de avaliação e de treino."""
    rows: list[dict[str, object]] = []
    for stage, pool_scores, evaluation_scores in stage_scores:
        settings = replace(model, n_estimators=stage) if model.is_boosting else model
        rows.append(
            {
                **describe_model(settings),
                **_evaluation_metrics(evaluation_labels, evaluation_scores),
                **training_metrics(pool_labels[seen], pool_scores[seen], pool_labels, pool_scores),
                "kept_estimators": float(min(stage, kept)) if model.is_boosting else np.nan,
                "failed": False,
            }
        )
    return rows


def _staged_scores(
    booster: AdaBoostClassifier, pool_input: NDArray[np.float64], evaluation_input: NDArray[np.float64], stages: Sequence[int]
) -> list[tuple[int, NDArray[np.float64], NDArray[np.float64]]]:
    """Pontuações nas rodadas pedidas; depois da parada do boosting, cada rodada repete a última."""
    wanted = sorted(stages)
    read: list[tuple[int, NDArray[np.float64], NDArray[np.float64]]] = []
    last: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    position = 0
    pairs = zip(booster.staged_decision_function(pool_input), booster.staged_decision_function(evaluation_input), strict=True)
    for round_number, (pool_scores, evaluation_scores) in enumerate(pairs, start=1):
        last = (np.asarray(pool_scores, dtype=float), np.asarray(evaluation_scores, dtype=float))
        while position < len(wanted) and wanted[position] == round_number:
            read.append((round_number, *last))
            position += 1
        if position == len(wanted):
            break
    if last is None:
        raise RuntimeError("o AdaBoost ajustado não tem nenhum classificador fraco")
    read.extend((stage, *last) for stage in wanted[position:])
    return read


def _validation_curve_task(
    input_encoding: InputEncoding,
    models: Sequence[tuple[ModelSettings, tuple[int, ...]]],
    fold: int,
    train_index: NDArray[np.intp],
    evaluation_index: NDArray[np.intp],
    features: NDArray[np.float64],
    labels: NDArray[np.int_],
    random_state: int,
) -> list[dict[str, object]]:
    """Uma entrada num fold, com todos os modelos dela: as mesmas operações, na mesma ordem, da busca da Seção 6."""
    pool_features, pool_labels = features[train_index], labels[train_index]
    undersampler = make_undersampler(random_state)
    sampled_features, sampled_labels = undersampler.fit_resample(pool_features, pool_labels)
    seen = undersampler.sample_indices_
    transformer = Pipeline(input_encoding.steps(random_state))
    train_input = transformer.fit_transform(sampled_features, sampled_labels)
    pool_input = transformer.transform(pool_features)
    evaluation_input = transformer.transform(features[evaluation_index])
    evaluation_labels = labels[evaluation_index]
    rows: list[dict[str, object]] = []
    for model, stages in models:
        estimator = model.build(random_state)
        try:
            estimator.fit(train_input, sampled_labels)
        except ValueError as error:
            if not _is_worse_than_random(error):
                raise
            rows.extend(
                {**describe_model(replace(model, n_estimators=stage) if model.is_boosting else model),
                 **dict.fromkeys(METRIC_COLUMNS, np.nan), "kept_estimators": 0.0, "failed": True}
                for stage in (stages or (model.n_estimators,))
            )
            continue
        if model.is_boosting:
            stage_scores = _staged_scores(estimator, pool_input, evaluation_input, stages or (model.n_estimators,))
        else:
            stage_scores = [(model.n_estimators, fraud_scores(estimator, pool_input), fraud_scores(estimator, evaluation_input))]
        rows.extend(_scored_rows(model, stage_scores, seen, pool_labels, evaluation_labels, _kept_estimators(estimator)))
    return [{**describe_input(input_encoding), **row, "fold": fold} for row in rows]


def validation_curve_scores(
    points: Sequence[CurvePoint],
    features: ArrayLike,
    labels: ArrayLike,
    n_splits: int = config.CV_N_SPLITS,
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = config.N_JOBS,
) -> pd.DataFrame:
    """Notas de treino e de avaliação de cada ponto, nos mesmos folds e com o mesmo sorteio da busca da Seção 6.

    Pontos repetidos (a configuração escolhida aparece em todas as curvas) são ajustados uma vez só, e os modelos
    de uma mesma entrada dividem a simulação dos estados, como na busca.

    Args:
        points: Saída de ``curve_points`` (de um ou de vários pipelines).
        features: Treino inteiro.
        labels: Rótulos.
        n_splits: Número de folds.
        random_state: Semente dos folds, do undersampling, do redutor e do boosting.
        n_jobs: Processos paralelos; o resultado não depende deste valor.

    Returns:
        Uma linha por configuração e fold (e por rodada lida, no ponto do teto), com ``VALIDATION_CURVE_COLUMNS``.
    """
    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels)
    models_by_input: dict[InputEncoding, dict[ModelSettings, set[int]]] = {}
    for point in points:
        models_by_input.setdefault(point.input_encoding, {}).setdefault(point.model, set()).update(point.stages)
    folds = cv_folds(label_array, n_splits, random_state)
    tasks = [
        (input_encoding, [(model, tuple(sorted(stages))) for model, stages in models.items()], fold, train, evaluation)
        for input_encoding, models in models_by_input.items()
        for fold, (train, evaluation) in enumerate(folds)
    ]
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_validation_curve_task)(input_encoding, models, fold, train, evaluation, feature_array, label_array, random_state)
        for input_encoding, models, fold, train, evaluation in tasks
    )
    return pd.DataFrame([row for task_rows in outputs for row in task_rows], columns=list(VALIDATION_CURVE_COLUMNS))


def _matching_rows(frame: pd.DataFrame, values: Mapping[str, object]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in values.items():
        mask &= frame[column].isna() if value is None else frame[column].eq(value)
    return mask


def validation_curve_table(
    fold_scores: pd.DataFrame, input_encoding: InputEncoding, model: ModelSettings, parameter: str
) -> pd.DataFrame:
    """Uma curva de validação lida das notas: o valor do hiperparâmetro, o fold e as métricas.

    No teto de rodadas, o valor escolhido aparece duas vezes (no ajuste com o teto escolhido e na rodada lida do
    ajuste longo); as duas notas são iguais (há teste), e a curva fica com uma.

    Args:
        fold_scores: Saída de ``validation_curve_scores``.
        input_encoding: Entrada escolhida.
        model: Modelo escolhido (o teto dele é o lido nas curvas que não são a do teto).
        parameter: O hiperparâmetro da curva.

    Returns:
        Colunas ``value``, ``fold``, as métricas, ``kept_estimators`` e ``failed``, em ordem de valor e fold.

    Raises:
        ValueError: Se algum valor da curva não tiver as notas de todos os folds.
    """
    fixed: dict[str, object] = {**describe_input(input_encoding), **describe_model(model)}
    fixed.pop(parameter)
    rows = fold_scores[_matching_rows(fold_scores, fixed)]
    if parameter != CEILING_PARAMETER:
        values = CURVE_VALUES[parameter]
        rows = rows[np.isclose(rows[parameter].to_numpy(dtype=float)[:, None], values).any(axis=1)]
    table = rows.rename(columns={parameter: "value"})[["value", "fold", *METRIC_COLUMNS, "kept_estimators", "failed"]]
    table = table.astype({"value": float}).drop_duplicates(["value", "fold"]).sort_values(["value", "fold"]).reset_index(drop=True)
    counts = table.groupby("value")["fold"].nunique()
    if table.empty or counts.nunique() != 1:
        raise ValueError(f"a curva de {parameter} não tem as notas de todos os folds em todos os valores")
    return table
