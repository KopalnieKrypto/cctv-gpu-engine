"""Automated promotion gate over same-image RTX 5070 resource evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).parents[1] / "results" / "resources-rtx5070-film1.json"
ARTIFACT_SHA256 = "0673d038f4e8d11ddeb7f3b5320c33c1fef6de2cc560f4b52fef5b12d96abbed"
VIDEO_SHA256 = "2f6ef8a0eaa1b1c96f3171ea48f5e25e6008ca7af4c04d634945220717dbceb8"
MODEL_SHA256 = "4835d97e368567838d2c6ba2ccaf329ee541de283cfa377e72188783ac89cd67"


@pytest.mark.perf
def test_frozen_resource_artifact_passes_real_gpu_comparison_gate() -> None:
    raw = ARTIFACT.read_bytes()
    artifact = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == ARTIFACT_SHA256
    assert artifact["video"]["sha256"] == VIDEO_SHA256
    assert artifact["video"]["duration_s"] == 299.883
    assert artifact["hardware"]["name"] == "NVIDIA GeForce RTX 5070"
    assert artifact["hardware"]["memory_used_mb"] == 2
    assert artifact["methodology"] == {
        "fresh_container_per_arm": True,
        "tracking_enabled": True,
        "source_bind_mount": False,
        "same_image_video_gpu": True,
        "peak_vram_scope": "whole selected GPU sampled through nvidia-smi",
    }

    arms = artifact["arms"]
    assert set(arms) == {"heuristic", "vlm", "mlp"}
    for classifier, arm in arms.items():
        assert arm["result_total_frames"] == 300
        assert arm["detections_rows"] == 300
        assert arm["result_diagnostics"]["classifier"] == classifier
        assert arm["vram_sample_interval_s"] == 0.5
        assert arm["vram_samples"]

    assert arms["mlp"]["result_diagnostics"]["activity_model"]["sha256"] == MODEL_SHA256
    assert arms["mlp"]["wallclock_s"] < arms["vlm"]["wallclock_s"]
    assert arms["mlp"]["peak_vram_mb"] < arms["vlm"]["peak_vram_mb"]
    assert artifact["promotion_resource_gate"] == {
        "mlp_faster_than_vlm": True,
        "mlp_lower_peak_vram_than_vlm": True,
        "passed": True,
    }
