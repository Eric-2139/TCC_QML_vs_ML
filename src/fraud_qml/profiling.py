"""Medição do custo computacional (tempo e memória) de um trecho de código.

O TCC mede custo em duas dimensões (seção 3.5): tempo e pico de memória. O ``tracemalloc``
pedido na metodologia só enxerga alocações feitas pelo interpretador Python; o simulador
Aer (C++) e a libsvm do SVC alocam memória nativa que ele não vê. Por isso medimos também
o pico de RSS do processo (memória física ocupada), amostrado em uma thread paralela, e
reportamos as duas grandezas lado a lado.
"""

import math
import threading
import time
import tracemalloc
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Self

import psutil

BYTES_PER_MIB: Final[int] = 1024 * 1024

# 10 ms captura picos de fits de segundos sem custo perceptível; picos mais curtos que
# isso não mudam a conclusão sobre viabilidade de memória.
RSS_SAMPLING_INTERVAL_SECONDS: Final[float] = 0.01


@dataclass(frozen=True)
class CostReport:
    """Custo computacional de um trecho de código.

    Attributes:
        wall_seconds: Tempo de parede decorrido.
        python_peak_mib: Pico de memória alocada pelo Python (``tracemalloc``); NaN quando
            a medição foi feita sem rastreamento.
        rss_peak_increase_mib: Quanto o pico de RSS do processo subiu em relação ao início
            do trecho; inclui memória nativa (Aer, libsvm, numpy).
    """

    wall_seconds: float
    python_peak_mib: float
    rss_peak_increase_mib: float


class _RssPeakSampler(threading.Thread):
    """Thread que amostra o RSS do processo e guarda o maior valor observado."""

    def __init__(self, interval_seconds: float) -> None:
        super().__init__(daemon=True)
        self._process = psutil.Process()
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self.peak_bytes: int = self._process.memory_info().rss

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)
            self._stop_event.wait(self._interval_seconds)

    def stop(self) -> int:
        """Encerra a amostragem e devolve o pico observado, em bytes."""
        self._stop_event.set()
        self.join()
        # Uma última leitura cobre um pico que tenha ocorrido após a amostra anterior
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)
        return self.peak_bytes


class CostMeter:
    """Context manager que mede tempo, pico de memória Python e pico de RSS.

    Exemplo:
        >>> with CostMeter() as meter:
        ...     model.fit(features, labels)
        >>> meter.report.wall_seconds

    O relatório só existe após a saída do bloco; acessá-lo antes gera ``RuntimeError``.

    Args:
        trace_python_memory: Liga o ``tracemalloc``. Ele intercepta cada alocação do
            Python e deixa código Python intensivo várias vezes mais lento; para comparar
            TEMPO entre modelos de forma justa, meça com ``False`` (o pico Python vira NaN).
        rss_sampling_interval_seconds: Intervalo entre leituras do RSS.
    """

    def __init__(
        self,
        trace_python_memory: bool = True,
        rss_sampling_interval_seconds: float = RSS_SAMPLING_INTERVAL_SECONDS,
    ) -> None:
        self._trace_python_memory = trace_python_memory
        self._rss_sampling_interval_seconds = rss_sampling_interval_seconds
        self._report: CostReport | None = None
        self._started_tracemalloc = False
        self._rss_sampler: _RssPeakSampler | None = None
        self._rss_at_start_bytes = 0
        self._start_seconds = 0.0

    @property
    def report(self) -> CostReport:
        if self._report is None:
            raise RuntimeError("O relatório só fica disponível depois que o bloco 'with' termina.")
        return self._report

    def __enter__(self) -> Self:
        # Se alguém já está rastreando (medição aninhada), reaproveitamos o rastreamento
        # em vez de reiniciá-lo, para não apagar a medição externa.
        if self._trace_python_memory:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._started_tracemalloc = True
            tracemalloc.reset_peak()

        self._rss_sampler = _RssPeakSampler(self._rss_sampling_interval_seconds)
        self._rss_at_start_bytes = self._rss_sampler.peak_bytes
        self._rss_sampler.start()

        # O relógio começa por último para não contar a preparação da própria medição
        self._start_seconds = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        wall_seconds = time.perf_counter() - self._start_seconds
        python_peak_mib = math.nan
        if self._trace_python_memory:
            _, python_peak_bytes = tracemalloc.get_traced_memory()
            python_peak_mib = python_peak_bytes / BYTES_PER_MIB
        if self._started_tracemalloc:
            tracemalloc.stop()

        assert self._rss_sampler is not None
        rss_peak_bytes = self._rss_sampler.stop()

        self._report = CostReport(
            wall_seconds=wall_seconds,
            python_peak_mib=python_peak_mib,
            rss_peak_increase_mib=max(rss_peak_bytes - self._rss_at_start_bytes, 0) / BYTES_PER_MIB,
        )
