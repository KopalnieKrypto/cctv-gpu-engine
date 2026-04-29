"""Per-job system telemetry — captured by the worker, surfaced on the dashboard.

The worker calls :meth:`MetricsCollector.sample` on every progress callback
during ``process_job``; samples accumulate in a :class:`MetricsAggregator`,
and the resulting peak/avg dict is written into ``status.json`` under the
``metrics`` key (read back by :mod:`gpu_service.dashboard`).

Three layers, mirroring the ``R2ClientLike`` boundary-driven pattern in
:mod:`gpu_service.worker`:

* :class:`MetricsCollector` — Protocol consumed by the worker. Tests pass
  an in-memory fake; production wires :class:`SystemMetricsCollector`.
* :class:`MetricsAggregator` — pure aggregation, zero I/O, fully unit-tested.
* :class:`SystemMetricsCollector` / :class:`NullMetricsCollector` — concrete
  implementations. ``SystemMetricsCollector`` lazy-imports ``psutil`` and
  ``pynvml`` so this module stays importable on the macOS dev box where
  neither is installed (Linux-only extras in ``pyproject.toml``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class MetricsSample:
    """Single point-in-time snapshot of host + GPU resource usage.

    GPU fields are ``None`` when the sample was taken on a host without
    NVIDIA drivers / NVML library — :class:`SystemMetricsCollector` sets
    them to ``None`` after a one-time NVML init failure so subsequent
    samples don't repeatedly try (and fail) to talk to the driver.
    """

    cpu_util_pct: float
    ram_used_pct: float
    disk_used_pct: float
    gpu_util_pct: float | None = None
    gpu_temp_c: float | None = None
    gpu_mem_used_mb: float | None = None


class MetricsCollector(Protocol):
    """Structural type: anything with ``sample() -> MetricsSample`` plugs in."""

    def sample(self) -> MetricsSample: ...


def _peak(values: list[float]) -> float | None:
    return max(values) if values else None


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass
class MetricsAggregator:
    """Accumulate :class:`MetricsSample` instances → summary dict.

    Computes peak + avg for utilization-like fields (CPU, GPU, RAM) and
    just the most recent value for disk (which is monotonic and slow-moving
    — peak vs. avg vs. last would all be approximately equal). GPU fields
    are skipped from the rollup if every sample has them as ``None``.
    """

    samples: list[MetricsSample] = field(default_factory=list)

    def add(self, sample: MetricsSample) -> None:
        self.samples.append(sample)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {
                "samples_count": 0,
                "cpu_util_peak_pct": None,
                "cpu_util_avg_pct": None,
                "ram_used_peak_pct": None,
                "disk_used_pct": None,
                "gpu_util_peak_pct": None,
                "gpu_util_avg_pct": None,
                "gpu_temp_peak_c": None,
                "gpu_mem_peak_mb": None,
            }

        cpu = [s.cpu_util_pct for s in self.samples]
        ram = [s.ram_used_pct for s in self.samples]
        gpu_util = [s.gpu_util_pct for s in self.samples if s.gpu_util_pct is not None]
        gpu_temp = [s.gpu_temp_c for s in self.samples if s.gpu_temp_c is not None]
        gpu_mem = [s.gpu_mem_used_mb for s in self.samples if s.gpu_mem_used_mb is not None]

        return {
            "samples_count": len(self.samples),
            "cpu_util_peak_pct": _peak(cpu),
            "cpu_util_avg_pct": _avg(cpu),
            "ram_used_peak_pct": _peak(ram),
            "disk_used_pct": self.samples[-1].disk_used_pct,
            "gpu_util_peak_pct": _peak(gpu_util),
            "gpu_util_avg_pct": _avg(gpu_util),
            "gpu_temp_peak_c": _peak(gpu_temp),
            "gpu_mem_peak_mb": _peak(gpu_mem),
        }


class NullMetricsCollector:
    """Returns a fixed zero sample. Used as a fallback when ``psutil``/NVML
    can't be imported (e.g. macOS dev box, broken pip install) — the worker
    keeps running and the dashboard simply shows ``—`` in metrics columns.
    """

    def sample(self) -> MetricsSample:
        return MetricsSample(
            cpu_util_pct=0.0,
            ram_used_pct=0.0,
            disk_used_pct=0.0,
        )


class SystemMetricsCollector:
    """psutil + pynvml-backed collector for the gpu-service container.

    NVML is initialised once on construction; if init or the first GPU
    handle fails, all subsequent samples emit ``gpu_*=None`` and don't
    retry — the alternative (probing every sample) would spam logs and
    burn CPU on a host that simply doesn't have a GPU.

    Disk usage is measured against ``/`` rather than ``WORKDIR`` because
    the operator-facing concern is "is the host filesystem full?", and
    on the gpu-service container both points resolve to the same overlayfs
    layer anyway.
    """

    def __init__(self, disk_path: str = "/") -> None:
        import psutil

        self._psutil = psutil
        self._disk_path = disk_path
        self._gpu_handle: Any | None = None
        self._pynvml: Any | None = None

        try:
            import pynvml

            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:
            # Any NVML failure → permanent CPU-only mode for this collector.
            # psutil-only telemetry is still useful (RAM/CPU/disk).
            self._gpu_handle = None
            self._pynvml = None

        # First call to psutil.cpu_percent() always returns 0.0 — prime it
        # so the first real sample reflects actual usage, not a cold start.
        self._psutil.cpu_percent(interval=None)

    def sample(self) -> MetricsSample:
        cpu = float(self._psutil.cpu_percent(interval=None))
        ram = float(self._psutil.virtual_memory().percent)
        disk = float(self._psutil.disk_usage(self._disk_path).percent)

        gpu_util: float | None = None
        gpu_temp: float | None = None
        gpu_mem: float | None = None

        if self._gpu_handle is not None and self._pynvml is not None:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util = float(util.gpu)
                gpu_temp = float(
                    self._pynvml.nvmlDeviceGetTemperature(
                        self._gpu_handle, self._pynvml.NVML_TEMPERATURE_GPU
                    )
                )
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_mem = float(mem.used) / (1024 * 1024)
            except Exception:
                # Driver hiccup: emit None for this single sample but keep
                # the handle so the next sample can recover.
                gpu_util = gpu_temp = gpu_mem = None

        return MetricsSample(
            cpu_util_pct=cpu,
            ram_used_pct=ram,
            disk_used_pct=disk,
            gpu_util_pct=gpu_util,
            gpu_temp_c=gpu_temp,
            gpu_mem_used_mb=gpu_mem,
        )
