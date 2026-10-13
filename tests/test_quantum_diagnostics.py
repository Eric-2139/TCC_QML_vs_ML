"""Varreduras de concentração e alinhamento dos kernels."""

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import StratifiedKFold, cross_validate

from fraud_qml.models import build_weak_quantum_classifier_pipeline, quantum_encoding_steps
from fraud_qml.preprocessing import build_amplitude_preprocessor, build_angle_preprocessor
from fraud_qml.quantum.diagnostics import (
    AMPLITUDE_SWEEP_COLUMNS,
    ANGLE_SWEEP_COLUMNS,
    WEAK_CLASSIFIER_SWEEP_COLUMNS,
    KernelSubsample,
    amplitude_alignment_sweep,
    angle_alignment_sweep,
    weak_classifier_encoding_sweep,
)
from fraud_qml.quantum.encodings import build_angle_feature_map
from fraud_qml.quantum.kernels import (
    amplitude_kernel_matrix,
    angle_kernel_matrix,
    centered_kernel_target_alignment,
    off_diagonal_values,
)

RNG_SEED = 0
N_QUBITS = 2
N_ROWS = 24
CONFIGURATIONS = {"Z + PCA": ("z", "pca"), "ZZ + PCA": ("zz", "pca")}
WIDTHS = (0.5, 1.0)


def _subsample(seed: int) -> KernelSubsample:
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], N_ROWS // 2)
    features = rng.normal(size=(N_ROWS, 5)) + labels[:, None]
    return features, labels


@pytest.fixture
def subsamples() -> dict[int, KernelSubsample]:
    return {seed: _subsample(seed) for seed in (RNG_SEED, RNG_SEED + 1)}


def test_angle_sweep_has_one_row_per_combination(subsamples: dict[int, KernelSubsample]) -> None:
    sweep = angle_alignment_sweep(subsamples, CONFIGURATIONS, WIDTHS, N_QUBITS, RNG_SEED)
    assert tuple(sweep.columns) == ANGLE_SWEEP_COLUMNS
    assert len(sweep) == len(CONFIGURATIONS) * len(WIDTHS) * len(subsamples)
    assert sweep["alignment"].between(-1, 1).all() and sweep["mean_similarity"].between(0, 1).all()


def test_angle_sweep_matches_a_direct_computation(subsamples: dict[int, KernelSubsample]) -> None:
    # A função precisa reproduzir exatamente o procedimento manual (preparar os ângulos, montar o kernel, medir)
    sweep = angle_alignment_sweep(subsamples, CONFIGURATIONS, WIDTHS, N_QUBITS, RNG_SEED)
    features, labels = subsamples[RNG_SEED + 1]
    angles = build_angle_preprocessor("standard", "pca", N_QUBITS, (0.0, 1.0), RNG_SEED).fit_transform(features, labels)
    kernel = angle_kernel_matrix(build_angle_feature_map("zz", N_QUBITS), angles)
    row = sweep.set_index(["configuration", "range_width", "subsample"]).loc[("ZZ + PCA", 1.0, RNG_SEED + 1)]
    assert row["alignment"] == pytest.approx(centered_kernel_target_alignment(kernel, labels), abs=1e-12)
    assert row["mean_similarity"] == pytest.approx(off_diagonal_values(kernel).mean(), abs=1e-12)


def test_angle_sweep_does_not_depend_on_parallelism(subsamples: dict[int, KernelSubsample]) -> None:
    sequential = angle_alignment_sweep(subsamples, CONFIGURATIONS, WIDTHS, N_QUBITS, RNG_SEED, n_jobs=1)
    parallel = angle_alignment_sweep(subsamples, CONFIGURATIONS, WIDTHS, N_QUBITS, RNG_SEED, n_jobs=2)
    pd.testing.assert_frame_equal(sequential, parallel, check_exact=False, atol=1e-12)


def test_amplitude_sweep_matches_a_direct_computation(subsamples: dict[int, KernelSubsample]) -> None:
    sweep = amplitude_alignment_sweep(subsamples, ("standard", "robust"), (0.0, 1.0), N_QUBITS + 1)
    assert tuple(sweep.columns) == AMPLITUDE_SWEEP_COLUMNS
    assert len(sweep) == 2 * 2 * len(subsamples)
    features, labels = subsamples[RNG_SEED]
    kernel = amplitude_kernel_matrix(build_amplitude_preprocessor("robust", N_QUBITS + 1, 1.0).fit_transform(features))
    row = sweep.set_index(["scaler", "padding_value", "subsample"]).loc[("robust", 1.0, RNG_SEED)]
    assert row["alignment"] == pytest.approx(centered_kernel_target_alignment(kernel, labels), abs=1e-12)


def _weak_classifier_sweep(subsamples: dict[int, KernelSubsample], n_jobs: int) -> pd.DataFrame:
    return weak_classifier_encoding_sweep(
        subsamples,
        configurations={"fase": ("z", 1), "RY": ("ry", 1)},
        reducers=("pca",),
        scalers=("standard",),
        range_widths=(1.0, np.pi / 2),
        ansatz_reps_options=(1,),
        initialization_seeds=(RNG_SEED,),
        max_iterations=15,
        n_splits=2,
        random_state=RNG_SEED,
        n_jobs=n_jobs,
    )


def test_weak_classifier_sweep_has_one_row_per_combination_and_fold(subsamples: dict[int, KernelSubsample]) -> None:
    sweep = _weak_classifier_sweep(subsamples, n_jobs=1)
    assert tuple(sweep.columns) == WEAK_CLASSIFIER_SWEEP_COLUMNS
    # 2 codificações × 1 redutor × 1 scaler × 2 larguras × 1 ansatz × 2 subamostras × 1 semente × 2 folds
    assert len(sweep) == 16
    groups = sweep.groupby(["configuration", "range_width", "subsample"])["fold"].apply(list)
    assert groups.map(lambda folds: folds == [0, 1]).all()
    assert sweep[["roc_auc", "accuracy", "train_accuracy"]].to_numpy().min() >= 0.0
    assert sweep[["roc_auc", "accuracy", "train_accuracy"]].to_numpy().max() <= 1.0


def test_weak_classifier_sweep_does_not_depend_on_parallelism(subsamples: dict[int, KernelSubsample]) -> None:
    pd.testing.assert_frame_equal(_weak_classifier_sweep(subsamples, n_jobs=1), _weak_classifier_sweep(subsamples, n_jobs=2))


def test_weak_classifier_sweep_matches_a_direct_cross_validation(subsamples: dict[int, KernelSubsample]) -> None:
    sweep = _weak_classifier_sweep(subsamples, n_jobs=1)
    features, labels = subsamples[RNG_SEED]
    steps = quantum_encoding_steps("angle", angle_range=(0.0, 1.0), feature_map="ry", feature_map_reps=1, random_state=RNG_SEED)
    pipeline = build_weak_quantum_classifier_pipeline(steps, ansatz_reps=1, max_iterations=15, random_state=RNG_SEED)
    folds = StratifiedKFold(2, shuffle=True, random_state=RNG_SEED)
    direct = cross_validate(pipeline, features, labels, cv=folds, scoring=("roc_auc", "accuracy"))
    rows = sweep[(sweep["configuration"] == "RY") & (sweep["range_width"] == 1.0) & (sweep["subsample"] == RNG_SEED)]
    np.testing.assert_allclose(rows["roc_auc"], direct["test_roc_auc"])
    np.testing.assert_allclose(rows["accuracy"], direct["test_accuracy"])
