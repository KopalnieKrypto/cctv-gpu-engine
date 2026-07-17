"""Contracts for the same-image end-to-end resource benchmark."""

from __future__ import annotations

import pytest

from activity_mlp.resource_benchmark import build_docker_command, resource_promotion_gate


def test_docker_arm_uses_baked_source_and_models_with_tracking_enabled(tmp_path) -> None:
    video = tmp_path / "film-1.mp4"
    output = tmp_path / "mlp"

    command = build_docker_command(
        docker="docker",
        image="cctv-gpu-engine:issue34",
        video=video,
        output_dir=output,
        classifier="mlp",
        gpu_device="1",
        hf_cache_volume="cctv-gpu-engine_hf-cache",
    )

    assert command[:7] == [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "device=1",
        "--pid=host",
        "-v",
    ]
    assert f"{video.resolve()}:/input/video.mp4:ro" in command
    assert f"{output.resolve()}:/output" in command
    assert "cctv-gpu-engine_hf-cache:/root/.cache/huggingface" in command
    assert command[-15:] == [
        "-m",
        "pipeline.analyze",
        "/input/video.mp4",
        "--output",
        "/output/result.json",
        "--dump-detections",
        "/output/detections.jsonl",
        "--model",
        "/app/models/yolo11s-pose.onnx",
        "--classifier",
        "mlp",
        "--activity-model",
        "/app/models/activity-mlp-v1.0.0.onnx",
        "--activity-model-metadata",
        "/app/models/activity-mlp-v1.0.0.json",
    ]
    assert "--no-tracker" not in command
    assert not any(mount.endswith(":/app") for mount in command)


def test_resource_gate_requires_mlp_to_beat_vlm_on_both_measures() -> None:
    passed = resource_promotion_gate(
        {
            "heuristic": {"wallclock_s": 10.0, "peak_vram_mb": 900},
            "vlm": {"wallclock_s": 70.0, "peak_vram_mb": 7000},
            "mlp": {"wallclock_s": 11.0, "peak_vram_mb": 1000},
        }
    )

    assert passed == {
        "mlp_faster_than_vlm": True,
        "mlp_lower_peak_vram_than_vlm": True,
        "passed": True,
    }


@pytest.mark.parametrize(
    ("mlp_wallclock", "mlp_vram", "failed_key"),
    [(70.0, 1000, "mlp_faster_than_vlm"), (11.0, 7000, "mlp_lower_peak_vram_than_vlm")],
)
def test_resource_gate_is_strict(mlp_wallclock: float, mlp_vram: int, failed_key: str) -> None:
    gate = resource_promotion_gate(
        {
            "heuristic": {"wallclock_s": 10.0, "peak_vram_mb": 900},
            "vlm": {"wallclock_s": 70.0, "peak_vram_mb": 7000},
            "mlp": {"wallclock_s": mlp_wallclock, "peak_vram_mb": mlp_vram},
        }
    )

    assert gate[failed_key] is False
    assert gate["passed"] is False
