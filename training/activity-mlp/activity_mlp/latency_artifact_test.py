"""Automated numeric gate over the immutable real-GPU latency artifact."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).parents[1] / "results" / "latency-rtx5070.json"
ARTIFACT_SHA256 = "516bcf4bd92ea612ca655cd51e3a6a7a28e14087f4a0ffce304b112e81b1232f"
MODEL_SHA256 = "4835d97e368567838d2c6ba2ccaf329ee541de283cfa377e72188783ac89cd67"


@pytest.mark.perf
def test_frozen_rtx5070_latency_artifact_passes_every_numeric_contract() -> None:
    raw = ARTIFACT.read_bytes()
    artifact = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == ARTIFACT_SHA256
    assert artifact["model"]["sha256"] == MODEL_SHA256
    assert artifact["model"]["feature_schema_version"] == "activity-mlp-features-v1"
    assert artifact["warmup_count"] >= 100
    assert artifact["measured_count"] >= 1000
    assert len(artifact["timings_ms"]) == artifact["measured_count"]
    ordered = sorted(artifact["timings_ms"])
    measured_p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    assert artifact["summary_ms"]["p95_higher"] == measured_p95
    assert measured_p95 <= 5.0
    assert artifact["gate"] == {"p95_limit_ms": 5.0, "passed": True}
    assert artifact["runtime"]["active_providers"][0] == "CUDAExecutionProvider"
    assert any(gpu["name"] == "NVIDIA GeForce RTX 5070" for gpu in artifact["hardware"]["gpus"])
