"""Medição de custo computacional."""

import math
import time

import numpy as np
import pytest

from fraud_qml.profiling import RSS_SAMPLING_INTERVAL_SECONDS, CostMeter

ALLOCATION_MIB = 64


def test_measures_wall_time() -> None:
    with CostMeter() as meter:
        time.sleep(0.05)
    assert meter.report.wall_seconds >= 0.05


def test_measures_python_and_native_memory() -> None:
    with CostMeter() as meter:
        # np.ones escreve em todas as páginas, então a memória vira RSS de fato
        buffer = np.ones(ALLOCATION_MIB * 1024 * 1024 // 8)
        # O RSS é amostrado a cada 10 ms: o pico precisa durar alguns intervalos para ser
        # visto com certeza (os fits reais duram segundos, então isso reflete o uso real)
        time.sleep(RSS_SAMPLING_INTERVAL_SECONDS * 10)
        del buffer
    # O numpy reporta suas alocações ao tracemalloc; o RSS captura a mesma memória pelo SO
    assert meter.report.python_peak_mib >= ALLOCATION_MIB * 0.9
    assert meter.report.rss_peak_increase_mib >= ALLOCATION_MIB * 0.5


def test_without_python_tracing_reports_nan() -> None:
    with CostMeter(trace_python_memory=False) as meter:
        pass
    assert math.isnan(meter.report.python_peak_mib)


def test_report_unavailable_inside_block() -> None:
    with CostMeter() as meter:
        with pytest.raises(RuntimeError):
            _ = meter.report
