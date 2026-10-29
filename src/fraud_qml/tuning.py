"""Busca de hiperparâmetros (Seção 6): validação cruzada sobre o pipeline inteiro, em dois níveis.

Cada combinação do grid é um pipeline completo (undersampling → entrada → codificação → modelo), e o
undersampling é sorteado dentro de cada fold, só com as linhas de treino daquele fold (PLAN, decisão de
arquitetura 3). Os folds de validação ficam na proporção real de fraudes, e a escolha é pela PR-AUC.

**Por que dois níveis.** A entrada e a codificação não dependem dos hiperparâmetros do modelo. Para cada
entrada e cada fold, o undersampling, o pré-processamento e a simulação dos estados são feitos uma vez, e
todos os valores de C, γ, taxa de aprendizado e repetições do ansatz reaproveitam o resultado. O número é
o mesmo de ajustar o pipeline inteiro em cada combinação (há teste); muda só a quantidade de simulação.

**Por que os AdaBoost são lidos rodada a rodada.** ``n_estimators`` é um teto: um único ajuste com o maior
valor dá, por ``staged_decision_function``, a pontuação que cada teto menor daria (há teste).
"""

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from fraud_qml import config
from fraud_qml.data import FRAUD_LABEL
from fraud_qml.environment import collect_versions
from fraud_qml.evaluation import fraud_scores
from fraud_qml.models import InputEncoding, ModelSettings
from fraud_qml.preprocessing import ReducerName, ScalerName, make_undersampler
from fraud_qml.profiling import CostMeter
from fraud_qml.quantum.encodings import AngleFeatureMapName, Entanglement

# Mensagem do scikit-learn quando o primeiro classificador fraco já é pior que o acaso (erro ponderado ≥ 0,5):
# em vez de parar, o AdaBoostClassifier não consegue ajustar nada
_WORSE_THAN_RANDOM_MESSAGE: Final[str] = "worse than random"

METRIC_COLUMNS: Final[tuple[str, ...]] = ("pr_auc", "roc_auc")
FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "fold",
    *METRIC_COLUMNS,
    "kept_estimators",
    "failed",
    "input_seconds",
    "fit_seconds",
    "score_seconds",
)

# Campos de cada descrição que valem para cada tipo; os outros viram vazio na tabela de resultados,
# para que um valor padrão ignorado (a faixa dos ângulos na amplitude, por exemplo) não pareça uma escolha
_INPUT_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "all_features": ("kind", "scaler"),
    "angle_components": ("kind", "scaler", "reducer"),
    "angle": ("kind", "scaler", "reducer", "range_width", "feature_map", "feature_map_reps", "entanglement"),
    "amplitude": ("kind", "scaler", "padding_value"),
}
_MODEL_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "svm": ("family", "C", "gamma"),
    "qsvm": ("family", "C"),
    "adaboost": ("family", "n_estimators", "learning_rate"),
    "quantum_adaboost": ("family", "n_estimators", "learning_rate", "ansatz_reps", "max_iterations"),
}


def describe_input(input_encoding: InputEncoding) -> dict[str, object]:
    """Campos da entrada que valem para o tipo dela; os demais ficam ``None``.

    No mapa ``ry`` as repetições não se aplicam (repetir a rotação só multiplicaria o ângulo), e o
    emaranhamento só existe no ``zz``.
    """
    relevant = set(_INPUT_FIELDS[input_encoding.kind])
    if input_encoding.kind == "angle":
        if input_encoding.feature_map != "zz":
            relevant.discard("entanglement")
        if input_encoding.feature_map == "ry":
            relevant.discard("feature_map_reps")
    return {name: (value if name in relevant else None) for name, value in asdict(input_encoding).items()}


def describe_model(model: ModelSettings) -> dict[str, object]:
    """Campos do modelo que valem para a família dele; os demais ficam ``None``."""
    relevant = set(_MODEL_FIELDS[model.family])
    return {name: (value if name in relevant else None) for name, value in asdict(model).items()}


INPUT_COLUMNS: Final[tuple[str, ...]] = tuple(asdict(InputEncoding("all_features")))
MODEL_COLUMNS: Final[tuple[str, ...]] = tuple(asdict(ModelSettings("svm")))
CONFIGURATION_COLUMNS: Final[tuple[str, ...]] = (*INPUT_COLUMNS, *MODEL_COLUMNS)


@dataclass(frozen=True)
class SearchSpace:
    """O grid de um modelo: as entradas (com a codificação) × as configurações do modelo.

    Attributes:
        name: Nome do modelo, usado nas tabelas e no cache.
        input_encodings: Opções de entrada e de codificação.
        models: Opções do modelo. Nos AdaBoost, todas com ``n_estimators`` igual ao maior estágio.
        boosting_stages: Tetos de rodadas lidos de cada ajuste dos AdaBoost; vazio nos SVM.

    Raises:
        ValueError: Se os estágios não combinarem com os modelos.
    """

    name: str
    input_encodings: tuple[InputEncoding, ...]
    models: tuple[ModelSettings, ...]
    boosting_stages: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.input_encodings or not self.models:
            raise ValueError("o grid precisa de pelo menos uma entrada e um modelo")
        boosting = {model.is_boosting for model in self.models}
        if len(boosting) != 1:
            raise ValueError("um grid não mistura AdaBoost com SVM")
        if boosting == {True}:
            if not self.boosting_stages:
                raise ValueError("os AdaBoost precisam dos tetos de rodadas a ler (boosting_stages)")
            if any(model.n_estimators != max(self.boosting_stages) for model in self.models):
                raise ValueError("cada AdaBoost precisa ser ajustado com o maior teto da lista")
        elif self.boosting_stages:
            raise ValueError("os SVM não têm rodadas para ler")

    @property
    def n_combinations(self) -> int:
        """Número de pipelines diferentes que a validação cruzada compara."""
        return len(self.input_encodings) * len(self.models) * max(1, len(self.boosting_stages))

    def n_tasks(self, n_splits: int) -> int:
        """Tarefas paralelas: uma por entrada e fold (cada uma ajusta todos os modelos)."""
        return len(self.input_encodings) * n_splits


# ---------------------------------------------------------------- opções do grid


def angle_input_options(
    scalers: Sequence[ScalerName], reducers: Sequence[ReducerName]
) -> tuple[tuple[ScalerName, ReducerName], ...]:
    """Caminhos até os 5 componentes: scaler × redutor, sem duplicar o que dá o mesmo resultado.

    Com informação mútua no meio, StandardScaler e RobustScaler escolhem as mesmas features e o MinMax
    desfaz a escala (CC8, Seção 4.6.2): esse caminho entra com um scaler só.
    """
    return tuple(
        (scaler, reducer)
        for reducer in reducers
        for scaler in (scalers if reducer == "pca" else scalers[:1])
    )


def angle_encoding_grid(
    scalers: Sequence[ScalerName],
    reducers: Sequence[ReducerName],
    range_widths: Sequence[float],
    feature_maps: Sequence[AngleFeatureMapName],
    feature_map_reps: Sequence[int],
    entanglements: Sequence[Entanglement],
) -> tuple[InputEncoding, ...]:
    """Opções da variação A: caminho até os componentes × largura da faixa × mapa.

    Repetições valem para Z e ZZ; emaranhamento, só para o ZZ; a rotação em Y entra uma vez por largura.
    """
    maps: list[tuple[AngleFeatureMapName, int, Entanglement]] = []
    for feature_map in feature_maps:
        if feature_map == "ry":
            maps.append(("ry", feature_map_reps[0], entanglements[0]))
            continue
        for reps in feature_map_reps:
            for entanglement in entanglements if feature_map == "zz" else entanglements[:1]:
                maps.append((feature_map, reps, entanglement))
    return tuple(
        InputEncoding(
            "angle", scaler=scaler, reducer=reducer, range_width=width, feature_map=feature_map,
            feature_map_reps=reps, entanglement=entanglement,
        )
        for scaler, reducer in angle_input_options(scalers, reducers)
        for width in range_widths
        for feature_map, reps, entanglement in maps
    )


def amplitude_encoding_grid(scalers: Sequence[ScalerName], padding_values: Sequence[float]) -> tuple[InputEncoding, ...]:
    """Opções da variação B: scaler × constante de preenchimento (31 features + 1 constante = 5 qubits)."""
    return tuple(InputEncoding("amplitude", scaler=scaler, padding_value=padding) for scaler in scalers for padding in padding_values)


def classical_input_grid(
    all_feature_scalers: Sequence[ScalerName], component_scalers: Sequence[ScalerName], reducers: Sequence[ReducerName]
) -> tuple[InputEncoding, ...]:
    """Opções de entrada dos clássicos: as 31 features e os 5 componentes da variação A.

    Args:
        all_feature_scalers: Scalers com as 31 features (no AdaBoost, um só: o *stump* ignora a escala).
        component_scalers: Scalers antes do redutor, no caminho dos 5 componentes.
        reducers: Redutores para os 5 componentes.
    """
    all_features = tuple(InputEncoding("all_features", scaler=scaler) for scaler in all_feature_scalers)
    components = tuple(
        InputEncoding("angle_components", scaler=scaler, reducer=reducer)
        for scaler, reducer in angle_input_options(component_scalers, reducers)
    )
    return all_features + components


SCALER_GRID: Final[tuple[ScalerName, ...]] = ("standard", "robust")
REDUCER_GRID: Final[tuple[ReducerName, ...]] = ("pca", "mutual_info")
# O mapa Z com 1 repetição e a RY têm o mesmo kernel (CC14): a RY só entra no modelo que mede o estado
QSVM_FEATURE_MAPS: Final[tuple[AngleFeatureMapName, ...]] = ("z", "zz")
QUANTUM_ADABOOST_FEATURE_MAPS: Final[tuple[AngleFeatureMapName, ...]] = ("z", "zz", "ry")


def project_search_spaces() -> dict[str, SearchSpace]:
    """Os quatro grids do TCC (Seção 6), montados a partir das opções de ``config``.

    - **SVM:** 31 features (Standard ou Robust) e os 5 componentes (scaler × redutor) × C × γ.
    - **AdaBoost:** 31 features (um scaler só: o *stump* ignora a escala) e os 5 componentes × taxa de
      aprendizado × teto de rodadas.
    - **QSVM:** variação A (caminho até os componentes × largura × mapa Z ou ZZ) e variação B (scaler ×
      constante) × C.
    - **Quantum AdaBoost:** as mesmas variações, com a RY entre os mapas, × taxa de aprendizado ×
      repetições do ansatz × teto de rodadas.

    Os clássicos não variam a largura da faixa: no *stump* ela não muda nada, e no RBF reescalar os 5
    componentes por um mesmo fator equivale a mudar o γ, que já está no grid.
    """
    ceiling = max(config.BOOSTING_N_ESTIMATORS_GRID)
    angle_options = (SCALER_GRID, REDUCER_GRID, config.ANGLE_RANGE_WIDTH_GRID)
    map_options = (config.FEATURE_MAP_REPS_GRID, config.ENTANGLEMENT_GRID)
    amplitude = amplitude_encoding_grid(SCALER_GRID, config.AMPLITUDE_PADDING_GRID)
    return {
        "svm": SearchSpace(
            "svm",
            classical_input_grid(SCALER_GRID, SCALER_GRID, REDUCER_GRID),
            tuple(ModelSettings("svm", C=c, gamma=gamma) for c in config.SVM_C_GRID for gamma in config.SVM_GAMMA_GRID),
        ),
        "adaboost": SearchSpace(
            "adaboost",
            classical_input_grid(SCALER_GRID[:1], SCALER_GRID, REDUCER_GRID),
            tuple(
                ModelSettings("adaboost", n_estimators=ceiling, learning_rate=rate)
                for rate in config.BOOSTING_LEARNING_RATE_GRID
            ),
            boosting_stages=config.BOOSTING_N_ESTIMATORS_GRID,
        ),
        "qsvm": SearchSpace(
            "qsvm",
            angle_encoding_grid(*angle_options, QSVM_FEATURE_MAPS, *map_options) + amplitude,
            tuple(ModelSettings("qsvm", C=c) for c in config.SVM_C_GRID),
        ),
        "quantum_adaboost": SearchSpace(
            "quantum_adaboost",
            angle_encoding_grid(*angle_options, QUANTUM_ADABOOST_FEATURE_MAPS, *map_options) + amplitude,
            tuple(
                ModelSettings("quantum_adaboost", n_estimators=ceiling, learning_rate=rate, ansatz_reps=reps)
                for rate in config.BOOSTING_LEARNING_RATE_GRID
                for reps in config.ANSATZ_REPS_GRID
            ),
            boosting_stages=config.BOOSTING_N_ESTIMATORS_GRID,
        ),
    }


# ---------------------------------------------------------------- avaliação de uma entrada num fold


def _stage_scores(booster: AdaBoostClassifier, features: ArrayLike, stages: Sequence[int]) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """Pontuação que cada teto de rodadas daria, lida de um ajuste com o maior teto.

    Se o boosting parou antes de um teto, aquele teto daria o mesmo modelo final.
    """
    wanted = sorted(stages)
    last_scores: NDArray[np.float64] | None = None
    position = 0
    for round_number, scores in enumerate(booster.staged_decision_function(features), start=1):
        last_scores = np.asarray(scores, dtype=float)
        while position < len(wanted) and wanted[position] == round_number:
            yield wanted[position], last_scores
            position += 1
    if last_scores is None:
        raise RuntimeError("o AdaBoost ajustado não tem nenhum classificador fraco")
    for stage in wanted[position:]:
        yield stage, last_scores


def _metric_row(labels: NDArray[np.int_], scores: NDArray[np.float64]) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(labels, scores, pos_label=FRAUD_LABEL)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def _score_model(
    model: ModelSettings,
    stages: Sequence[int],
    train_input: NDArray[np.float64],
    train_labels: NDArray[np.int_],
    validation_input: NDArray[np.float64],
    validation_labels: NDArray[np.int_],
    random_state: int,
) -> list[dict[str, object]]:
    estimator = model.build(random_state)
    stage_settings = [replace(model, n_estimators=stage) for stage in stages] if model.is_boosting else [model]
    with CostMeter(trace_python_memory=False) as fit_meter:
        try:
            estimator.fit(train_input, train_labels)
        except ValueError as error:
            if _WORSE_THAN_RANDOM_MESSAGE not in str(error):
                raise
            failed = {"pr_auc": np.nan, "roc_auc": np.nan, "kept_estimators": 0, "failed": True}
            return [{**describe_model(settings), **failed} for settings in stage_settings]
    fit_seconds = fit_meter.report.wall_seconds
    rows: list[dict[str, object]] = []
    with CostMeter(trace_python_memory=False) as score_meter:
        if model.is_boosting:
            kept = len(estimator.estimators_)
            for stage, scores in _stage_scores(estimator, validation_input, stages):
                settings = replace(model, n_estimators=stage)
                rows.append(
                    {**describe_model(settings), **_metric_row(validation_labels, scores), "kept_estimators": min(stage, kept)}
                )
        else:
            scores = fraud_scores(estimator, validation_input)
            rows.append({**describe_model(model), **_metric_row(validation_labels, scores), "kept_estimators": np.nan})
    for row in rows:
        row.update(failed=False, fit_seconds=fit_seconds, score_seconds=score_meter.report.wall_seconds)
    return rows


def evaluate_input_on_fold(
    space: SearchSpace,
    input_encoding: InputEncoding,
    fold: int,
    train_index: NDArray[np.intp],
    validation_index: NDArray[np.intp],
    features: NDArray[np.float64],
    labels: NDArray[np.int_],
    random_state: int,
) -> tuple[list[dict[str, object]], float]:
    """Uma tarefa do grid: uma entrada num fold, com todos os modelos do espaço.

    Faz o que o pipeline completo faria, na mesma ordem: sorteia o treino balanceado só entre as linhas
    de treino do fold, ajusta a entrada (e simula os estados) nele, transforma o fold de validação, e
    então ajusta e pontua cada modelo sobre essas mesmas linhas transformadas.

    Args:
        space: O grid.
        input_encoding: A entrada avaliada.
        fold: Número do fold.
        train_index: Linhas de treino do fold.
        validation_index: Linhas de validação do fold (proporção real de fraudes).
        features: Todas as linhas do treino.
        labels: Os rótulos delas.
        random_state: Semente do undersampling, do redutor e do boosting.

    Returns:
        Uma linha de resultado por modelo (e por teto de rodadas, nos AdaBoost), e o tempo da tarefa.
    """
    with CostMeter(trace_python_memory=False) as task_meter:
        with CostMeter(trace_python_memory=False) as input_meter:
            sampled_features, sampled_labels = make_undersampler(random_state).fit_resample(
                features[train_index], labels[train_index]
            )
            transformer = Pipeline(input_encoding.steps(random_state))
            train_input = transformer.fit_transform(sampled_features, sampled_labels)
            validation_input = transformer.transform(features[validation_index])
        validation_labels = labels[validation_index]
        rows: list[dict[str, object]] = []
        for model in space.models:
            for row in _score_model(
                model, space.boosting_stages, train_input, sampled_labels, validation_input, validation_labels, random_state
            ):
                rows.append({**describe_input(input_encoding), **row, "fold": fold, "input_seconds": input_meter.report.wall_seconds})
    return rows, task_meter.report.wall_seconds


# ---------------------------------------------------------------- a busca e o cache


@dataclass(frozen=True)
class GridSearchResult:
    """Resultado bruto de uma busca.

    Attributes:
        space_name: Nome do modelo.
        fold_scores: Uma linha por combinação e fold.
        task_seconds: Tempo de cada tarefa (entrada × fold), medido dentro do processo que a executou.
        wall_seconds: Tempo total da busca, do início ao fim, com as tarefas em paralelo.
        from_cache: Se o resultado veio do cache em disco (então ``wall_seconds`` é o da execução original).
    """

    space_name: str
    fold_scores: pd.DataFrame
    task_seconds: NDArray[np.float64]
    wall_seconds: float
    from_cache: bool


def cv_folds(labels: ArrayLike, n_splits: int, random_state: int) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Os folds estratificados, os mesmos para todas as combinações e todos os modelos (comparação pareada)."""
    target = np.asarray(labels)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros(len(target)), target))


def run_grid_search(
    space: SearchSpace,
    features: ArrayLike,
    labels: ArrayLike,
    n_splits: int = config.CV_N_SPLITS,
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = config.N_JOBS,
) -> GridSearchResult:
    """Validação cruzada de todas as combinações de um grid, em paralelo (uma tarefa por entrada e fold).

    Args:
        space: O grid.
        features: Linhas de treino (o treino inteiro: o undersampling acontece dentro de cada fold).
        labels: Rótulos.
        n_splits: Número de folds.
        random_state: Semente dos folds, do undersampling, do redutor e do boosting.
        n_jobs: Processos paralelos; o resultado não depende deste valor.

    Returns:
        As pontuações de cada combinação em cada fold e os tempos.
    """
    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels)
    folds = cv_folds(label_array, n_splits, random_state)
    tasks = [(input_encoding, fold, train, validation) for input_encoding in space.input_encodings for fold, (train, validation) in enumerate(folds)]
    with CostMeter(trace_python_memory=False) as wall_meter:
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_input_on_fold)(space, input_encoding, fold, train, validation, feature_array, label_array, random_state)
            for input_encoding, fold, train, validation in tasks
        )
    rows = [row for task_rows, _ in outputs for row in task_rows]
    fold_scores = pd.DataFrame(rows, columns=[*CONFIGURATION_COLUMNS, *FOLD_COLUMNS])
    return GridSearchResult(
        space_name=space.name,
        fold_scores=fold_scores,
        task_seconds=np.array([seconds for _, seconds in outputs]),
        wall_seconds=wall_meter.report.wall_seconds,
        from_cache=False,
    )


def package_source_fingerprint() -> str:
    """Hash do código-fonte do pacote: qualquer mudança em ``fraud_qml`` invalida o cache do grid."""
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(package_dir.rglob("*.py")):
        digest.update(source.relative_to(package_dir).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def grid_cache_key(space: SearchSpace, features: ArrayLike, labels: ArrayLike, n_splits: int, random_state: int) -> str:
    """Identidade de uma busca: o grid, os dados, os folds, o código do pacote e as versões das bibliotecas."""
    return joblib.hash(
        (
            space,
            joblib.hash(np.asarray(features, dtype=float)),
            joblib.hash(np.asarray(labels)),
            n_splits,
            random_state,
            package_source_fingerprint(),
            collect_versions(),
        )
    )


def cached_grid_search(
    space: SearchSpace,
    features: ArrayLike,
    labels: ArrayLike,
    cache_dir: Path = config.CACHE_DIR,
    n_splits: int = config.CV_N_SPLITS,
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = config.N_JOBS,
) -> GridSearchResult:
    """``run_grid_search`` com cache em disco, para reexecutar o notebook sem refazer o grid.

    O cache só é reaproveitado se nada que muda o resultado mudou (``grid_cache_key``); qualquer
    alteração no código do pacote, nos dados, no grid ou nas versões refaz a busca.

    Returns:
        O resultado, com ``from_cache`` indicando de onde veio.
    """
    cache_file = cache_dir / f"grid_{space.name}_{grid_cache_key(space, features, labels, n_splits, random_state)}.joblib"
    if cache_file.is_file():
        return replace(joblib.load(cache_file), from_cache=True)
    result = run_grid_search(space, features, labels, n_splits, random_state, n_jobs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, cache_file)
    return result


def project_grid_seconds(pilot_task_seconds: Sequence[float], n_tasks: int, n_jobs: int) -> float:
    """Projeção conservadora do tempo de um grid a partir de um piloto.

    Supõe que toda tarefa demora o mesmo que a **mais lenta** do piloto e que as tarefas rodam em
    levas de ``n_jobs``.

    Args:
        pilot_task_seconds: Tempos das tarefas do piloto.
        n_tasks: Tarefas do grid completo.
        n_jobs: Processos paralelos.

    Returns:
        Segundos projetados.
    """
    return float(max(pilot_task_seconds)) * math.ceil(n_tasks / n_jobs)


# ---------------------------------------------------------------- leitura dos resultados


def summarize_grid(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Média e desvio entre folds de cada combinação, da melhor para a pior PR-AUC média.

    Uma combinação que falhou em algum fold (o primeiro classificador fraco já pior que o acaso) fica
    sem média, no fim da tabela: a média dos outros folds não seria comparável.

    Returns:
        Uma linha por combinação, com ``pr_auc_mean``, ``pr_auc_std``, ``roc_auc_mean``, ``roc_auc_std``,
        ``failed_folds``, ``kept_estimators_min`` e ``kept_estimators_max``, e a posição (``rank``).
    """
    grouped = fold_scores.groupby(list(CONFIGURATION_COLUMNS), dropna=False, sort=False)
    summary = pd.DataFrame(
        {
            "pr_auc_mean": grouped["pr_auc"].agg(lambda values: values.mean(skipna=False)),
            "pr_auc_std": grouped["pr_auc"].std(),
            "roc_auc_mean": grouped["roc_auc"].agg(lambda values: values.mean(skipna=False)),
            "roc_auc_std": grouped["roc_auc"].std(),
            "failed_folds": grouped["failed"].sum().astype(int),
            "kept_estimators_min": grouped["kept_estimators"].min(),
            "kept_estimators_max": grouped["kept_estimators"].max(),
        }
    ).reset_index()
    summary = summary.sort_values("pr_auc_mean", ascending=False, na_position="last", kind="stable").reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def _defined(row: Mapping[str, object], names: Sequence[str]) -> dict[str, object]:
    return {name: row[name] for name in names if row[name] is not None and not (isinstance(row[name], float) and math.isnan(row[name]))}


def configuration_from_row(row: Mapping[str, object]) -> tuple[InputEncoding, ModelSettings]:
    """Reconstrói a entrada e o modelo de uma linha de resultado (para montar o pipeline escolhido).

    Campos vazios voltam ao padrão, que é o valor que a combinação de fato usou.
    """
    input_fields = _defined(row, INPUT_COLUMNS)
    model_fields = _defined(row, MODEL_COLUMNS)
    for integer_field in ("feature_map_reps", "n_estimators", "ansatz_reps", "max_iterations"):
        for fields in (input_fields, model_fields):
            if integer_field in fields:
                fields[integer_field] = int(fields[integer_field])  # type: ignore[call-overload]
    return InputEncoding(**input_fields), ModelSettings(**model_fields)  # type: ignore[arg-type]


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_grid_report(
    path: Path,
    result: GridSearchResult,
    summary: pd.DataFrame,
    metadata: Mapping[str, object],
    top: int = 10,
) -> None:
    """Grava o resumo de um grid em JSON, para conferir os números do TCC sem reexecutar o notebook.

    Args:
        path: Arquivo de saída (em ``artifacts/results/``).
        result: O resultado bruto.
        summary: Saída de ``summarize_grid``.
        metadata: Versões, commit, fingerprint do split e o que mais identificar a execução.
        top: Quantas combinações, das melhores, entram no arquivo.
    """
    payload = {
        "model": result.space_name,
        "date": date.today().isoformat(),
        "combinations": int(len(summary)),
        "folds": int(result.fold_scores["fold"].nunique()),
        "wall_seconds": result.wall_seconds,
        "failed_combinations": int((summary["failed_folds"] > 0).sum()),
        "best": summary.head(top).to_dict(orient="records"),
        **metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
