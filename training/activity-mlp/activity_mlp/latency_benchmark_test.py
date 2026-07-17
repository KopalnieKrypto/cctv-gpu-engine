"""Measured RTX 5070 latency gate for the frozen issue #34 model."""

from __future__ import annotations

from io import StringIO

import pytest

from activity_mlp.latency_benchmark import benchmark_latency


class _Classifier:
    model_version = "activity-mlp-v1.0.0"
    model_sha256 = "4835d97e"

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, _detection) -> str:
        self.calls += 1
        return "standing"


def _clock_for_durations_ms(durations_ms: list[float]):
    readings = []
    cursor_ns = 1_000_000_000
    for duration_ms in durations_ms:
        readings.extend([cursor_ns, cursor_ns + int(duration_ms * 1_000_000)])
        cursor_ns += 10_000_000
    return iter(readings).__next__


def test_benchmark_records_every_raw_timing_and_conservative_p95() -> None:
    classifier = _Classifier()
    output = StringIO()

    result = benchmark_latency(
        classifier=classifier,
        detection=object(),
        warmup_count=2,
        measured_count=5,
        clock_ns=_clock_for_durations_ms([1.0, 2.0, 3.0, 4.0, 5.0]),
        heartbeat_every=2,
        heartbeat_output=output,
    )

    assert classifier.calls == 7
    assert result["warmup_count"] == 2
    assert result["measured_count"] == 5
    assert result["timings_ms"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["summary_ms"] == {
        "min": 1.0,
        "mean": 3.0,
        "median": 3.0,
        "p95_higher": 5.0,
        "max": 5.0,
    }
    assert result["gate"] == {"p95_limit_ms": 5.0, "passed": True}
    assert "latency heartbeat measured=2/5" in output.getvalue()


def test_benchmark_marks_gate_failed_when_p95_exceeds_five_ms() -> None:
    result = benchmark_latency(
        classifier=_Classifier(),
        detection=object(),
        warmup_count=0,
        measured_count=3,
        clock_ns=_clock_for_durations_ms([1.0, 1.0, 5.01]),
        heartbeat_every=100,
        heartbeat_output=StringIO(),
    )

    assert result["summary_ms"]["p95_higher"] == pytest.approx(5.01)
    assert result["gate"]["passed"] is False


@pytest.mark.parametrize("warmup,measured", [(-1, 1), (0, 0)])
def test_benchmark_rejects_invalid_counts(warmup: int, measured: int) -> None:
    with pytest.raises(ValueError, match="counts"):
        benchmark_latency(
            classifier=_Classifier(),
            detection=object(),
            warmup_count=warmup,
            measured_count=measured,
            clock_ns=_clock_for_durations_ms([1.0]),
            heartbeat_output=StringIO(),
        )
