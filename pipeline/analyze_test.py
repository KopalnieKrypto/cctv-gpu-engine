"""Tests for the analyze CLI — single-frame proof-of-concept entry point.

The detector and extractor are mocked at the boundary so we can verify the
CLI's argument parsing, orchestration, and JSON output schema without a GPU
or real video file.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from pipeline.analyze import main
from pipeline.postprocessing import Detection, Keypoint


def _detection(activity: str = "standing", confidence: float = 0.9) -> Detection:
    kps = [Keypoint(x=float(i), y=float(i * 2), vis=0.95) for i in range(17)]
    det = Detection(
        bbox=[100.0, 200.0, 300.0, 600.0],
        confidence=confidence,
        keypoints=kps,
    )
    det.activity = activity
    return det


class TestAnalyzeCLI:
    def test_outputs_json_with_person_count_and_per_detection_data(
        self, mocker, capsys
    ):
        # Mock extractor → returns a dummy frame
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.extract_frame_at",
            return_value=fake_frame,
        )

        # Mock detector loader → returns a fake detector that yields one person
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [
            _detection(activity="walking", confidence=0.87),
        ]
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        exit_code = main([
            "video.mp4",
            "--timestamp", "12.5",
            "--model", "models/yolo11n-pose.onnx",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        # Top-level schema
        assert payload["video"] == "video.mp4"
        assert payload["timestamp_s"] == 12.5
        assert payload["person_count"] == 1
        assert isinstance(payload["persons"], list)

        person = payload["persons"][0]
        assert person["activity"] == "walking"
        assert person["confidence"] == pytest.approx(0.87, abs=1e-5)
        assert person["bbox"] == pytest.approx([100.0, 200.0, 300.0, 600.0])
        assert len(person["keypoints"]) == 17
        kp0 = person["keypoints"][0]
        assert set(kp0.keys()) == {"x", "y", "vis"}

        # Detector was loaded with the requested model path and called once
        from pipeline.analyze import load_pose_model as patched_loader
        patched_loader.assert_called_once_with("models/yolo11n-pose.onnx")
        fake_detector.detect.assert_called_once()

    def test_outputs_empty_persons_list_when_no_detections(self, mocker, capsys):
        mocker.patch(
            "pipeline.analyze.extract_frame_at",
            return_value=np.zeros((480, 640, 3), dtype=np.uint8),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = []
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        exit_code = main([
            "video.mp4",
            "--timestamp", "0",
            "--model", "models/yolo11n-pose.onnx",
        ])

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["person_count"] == 0
        assert payload["persons"] == []

    def test_propagates_runtime_error_with_nonzero_exit_code(self, mocker, capsys):
        # Simulate the CUDA-missing error from load_pose_model
        mocker.patch(
            "pipeline.analyze.extract_frame_at",
            return_value=np.zeros((480, 640, 3), dtype=np.uint8),
        )
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            side_effect=RuntimeError("CUDAExecutionProvider not available"),
        )

        exit_code = main([
            "video.mp4",
            "--timestamp", "5",
            "--model", "models/yolo11n-pose.onnx",
        ])

        assert exit_code != 0
        captured = capsys.readouterr()
        assert "CUDAExecutionProvider" in captured.err
