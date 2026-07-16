"""Measured acceptance assertions for issue #86's GPU benchmark artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.mark.perf
def test_pose_mode_results_cover_every_locked_numeric_bound():
    result_path = os.environ.get("POSE_BENCHMARK_RESULTS")
    if not result_path:
        pytest.skip("set POSE_BENCHMARK_RESULTS to the measured issue #86 result JSON")
    artifact = json.loads(Path(result_path).read_text(encoding="utf-8"))

    fixture = artifact["fixture"]
    assert fixture["frame_count"] >= 60
    assert len(fixture["window_ids"]) >= 3
    assert fixture["annotation_methodology"]
    assert artifact["locked_settings"] == {
        "confidence_threshold": 0.25,
        "nms_iou_threshold": 0.45,
        "true_positive_iou_threshold": 0.5,
    }

    arms = artifact["arms"]
    assert set(arms) == {"baseline_640", "full_frame_1280", "focused_roi_640"}
    expected_frame_ids = artifact["fixture"]["frame_ids"]
    required_checks = {
        "pilot_recall",
        "pilot_precision",
        "throughput_regression",
        "one_hour_extrapolated_s",
        "peak_process_gpu_vram_mb",
        "film_1_recall_no_regression",
        "film_2_recall_no_regression",
    }
    eligible = []
    for name, arm in arms.items():
        assert [frame["frame_id"] for frame in arm["frames"]] == expected_frame_ids
        assert len(arm["pose_inference_wallclock_s"]["samples"]) == fixture["frame_count"]
        assert arm["pose_inference_wallclock_s"]["p50"] >= 0
        assert arm["pose_inference_wallclock_s"]["p95"] >= 0
        assert arm["end_to_end"]["measured_frame_count"] > 0
        assert arm["end_to_end"]["measured_video_duration_s"] > 0
        assert arm["end_to_end"]["measured_wallclock_s"] > 0
        assert arm["end_to_end"]["formula"]
        assert arm["end_to_end"]["linearity_assumption"]
        assert arm["peak_process_gpu_vram_mb"] >= 0
        checks = {check["name"]: check for check in arm["eligibility_checks"]}
        assert set(checks) == required_checks
        if all(check["passed"] for check in checks.values()):
            eligible.append(name)

    decision = artifact["decision"]
    if eligible:
        winner = min(
            eligible,
            key=lambda name: (
                arms[name]["end_to_end"]["measured_wallclock_s"]
                / arms[name]["end_to_end"]["measured_video_duration_s"]
            ),
        )
        assert decision["winner"] == winner
        assert decision["production_default_changed"] is (winner != "baseline_640")
        assert decision["follow_up_issue"] is None
    else:
        assert decision["winner"] is None
        assert decision["production_default_changed"] is False
        assert decision["follow_up_issue"]
