"""Tests for the per-job telemetry layer (gpu_service.metrics).

Only the pure pieces — :class:`MetricsAggregator` and
:class:`NullMetricsCollector` — are exercised here. The
:class:`SystemMetricsCollector` requires real ``psutil`` + NVML and is
covered indirectly by the ``make test-gpu`` end-to-end run.
"""

from __future__ import annotations

from gpu_service.metrics import (
    MetricsAggregator,
    MetricsSample,
    NullMetricsCollector,
)


def _sample(
    cpu: float = 0.0,
    ram: float = 0.0,
    disk: float = 0.0,
    gpu_util: float | None = None,
    gpu_temp: float | None = None,
    gpu_mem: float | None = None,
) -> MetricsSample:
    return MetricsSample(
        cpu_util_pct=cpu,
        ram_used_pct=ram,
        disk_used_pct=disk,
        gpu_util_pct=gpu_util,
        gpu_temp_c=gpu_temp,
        gpu_mem_used_mb=gpu_mem,
    )


def test_aggregator_empty_summary_uses_none() -> None:
    """No samples → every metric is None and samples_count is 0.

    Guards against divide-by-zero in ``_avg`` and ensures the dashboard
    renders ``—`` (the None-sentinel) for jobs that finished before the
    first progress callback fired.
    """
    summary = MetricsAggregator().summary()

    assert summary["samples_count"] == 0
    assert summary["cpu_util_peak_pct"] is None
    assert summary["cpu_util_avg_pct"] is None
    assert summary["ram_used_peak_pct"] is None
    assert summary["disk_used_pct"] is None
    assert summary["gpu_util_peak_pct"] is None
    assert summary["gpu_util_avg_pct"] is None
    assert summary["gpu_temp_peak_c"] is None
    assert summary["gpu_mem_peak_mb"] is None


def test_aggregator_peak_avg_correct() -> None:
    """3 samples → peak/avg match arithmetic on each field."""
    agg = MetricsAggregator()
    agg.add(_sample(cpu=10, ram=20, disk=30, gpu_util=40, gpu_temp=50, gpu_mem=600))
    agg.add(_sample(cpu=80, ram=40, disk=31, gpu_util=70, gpu_temp=72, gpu_mem=900))
    agg.add(_sample(cpu=20, ram=30, disk=32, gpu_util=10, gpu_temp=55, gpu_mem=300))

    summary = agg.summary()

    assert summary["samples_count"] == 3
    assert summary["cpu_util_peak_pct"] == 80
    assert summary["cpu_util_avg_pct"] == (10 + 80 + 20) / 3
    assert summary["ram_used_peak_pct"] == 40
    # disk_used_pct = last sample (monotonic)
    assert summary["disk_used_pct"] == 32
    assert summary["gpu_util_peak_pct"] == 70
    assert summary["gpu_util_avg_pct"] == (40 + 70 + 10) / 3
    assert summary["gpu_temp_peak_c"] == 72
    assert summary["gpu_mem_peak_mb"] == 900


def test_aggregator_handles_missing_gpu() -> None:
    """All samples missing GPU → gpu_* fields stay None, CPU/RAM still work.

    Models the macOS dev box / no-NVIDIA-host case where
    :class:`SystemMetricsCollector` permanently emits ``gpu_*=None``.
    """
    agg = MetricsAggregator()
    agg.add(_sample(cpu=15, ram=25, disk=40))
    agg.add(_sample(cpu=25, ram=35, disk=41))

    summary = agg.summary()

    assert summary["samples_count"] == 2
    assert summary["cpu_util_peak_pct"] == 25
    assert summary["cpu_util_avg_pct"] == 20
    assert summary["ram_used_peak_pct"] == 35
    assert summary["disk_used_pct"] == 41
    assert summary["gpu_util_peak_pct"] is None
    assert summary["gpu_util_avg_pct"] is None
    assert summary["gpu_temp_peak_c"] is None
    assert summary["gpu_mem_peak_mb"] is None


def test_null_collector_returns_zeros() -> None:
    """NullMetricsCollector emits a zero CPU/RAM/disk + None GPU sample.

    Ensures the worker can run under the fallback collector without
    crashing on attribute access — peak/avg over a single zero sample
    is still well-defined.
    """
    sample = NullMetricsCollector().sample()

    assert sample.cpu_util_pct == 0.0
    assert sample.ram_used_pct == 0.0
    assert sample.disk_used_pct == 0.0
    assert sample.gpu_util_pct is None
    assert sample.gpu_temp_c is None
    assert sample.gpu_mem_used_mb is None
