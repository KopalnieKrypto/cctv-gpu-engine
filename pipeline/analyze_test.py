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
    def test_outputs_json_with_person_count_and_per_detection_data(self, mocker, capsys):
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

        exit_code = main(
            [
                "video.mp4",
                "--timestamp",
                "12.5",
                "--model",
                "models/yolo11n-pose.onnx",
            ]
        )

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

        exit_code = main(
            [
                "video.mp4",
                "--timestamp",
                "0",
                "--model",
                "models/yolo11n-pose.onnx",
            ]
        )

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

        exit_code = main(
            [
                "video.mp4",
                "--timestamp",
                "5",
                "--model",
                "models/yolo11n-pose.onnx",
            ]
        )

        assert exit_code != 0
        captured = capsys.readouterr()
        assert "CUDAExecutionProvider" in captured.err


class TestAnalyzeFullVideoMode:
    def test_output_mode_writes_standalone_html_with_summary(self, mocker, tmp_path):
        # Two frames: first frame has 2 walking persons, second frame has 1 standing
        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter(
                [
                    (0.0, fake_frame),
                    (1.0, fake_frame),
                ]
            ),
        )

        fake_detector = MagicMock()
        fake_detector.detect.side_effect = [
            [_detection(activity="walking"), _detection(activity="walking")],
            [_detection(activity="standing")],
        ]
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        out_path = tmp_path / "report.html"
        exit_code = main(
            [
                "video.mp4",
                "--output",
                str(out_path),
                "--model",
                "models/yolo11n-pose.onnx",
            ]
        )

        assert exit_code == 0
        assert out_path.exists()
        html = out_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower()
        # Detector ran for every frame
        assert fake_detector.detect.call_count == 2
        # Walking dominates → present in summary table
        assert "walking" in html.lower()

    def test_output_mode_handles_video_with_no_people(self, mocker, tmp_path):
        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame)]),
        )

        fake_detector = MagicMock()
        fake_detector.detect.return_value = []  # never any persons
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        out_path = tmp_path / "empty.html"
        exit_code = main(
            ["video.mp4", "--output", str(out_path), "--model", "models/yolo11n-pose.onnx"]
        )

        assert exit_code == 0
        html = out_path.read_text(encoding="utf-8")
        # No keyframes section content beyond the empty placeholder
        assert "No keyframes" in html or "no persons" in html.lower()


class TestRunFullVideoToHtml:
    """Library function used by both the CLI and the gpu_service worker.

    Mocks live at the boundaries we control: the ffmpeg frame iterator and
    the pose-model loader. Aggregator + report renderer run for real so we're
    asserting end-to-end pipeline behaviour, not the shape of internal calls.
    """

    def test_single_chunk_returns_html_bytes(self, mocker):
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_html

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            return_value=iter([(0.0, fake_frame), (1.0, fake_frame)]),
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="walking")]
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        html = run_full_video_to_html([Path("chunk_001.mp4")])

        assert isinstance(html, bytes)
        text = html.decode("utf-8")
        assert "<!doctype html" in text.lower() or "<html" in text.lower()
        assert "walking" in text.lower()
        assert fake_detector.detect.call_count == 2

    def test_two_chunks_aggregate_into_one_report(self, mocker):
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_html

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        # Each chunk has its own iter_frames result; iter_frames is called
        # once per chunk so side_effect cycles through these.
        chunk_a_frames = [(float(i), fake_frame) for i in range(30)]
        chunk_b_frames = [(float(i), fake_frame) for i in range(30)]
        mocker.patch(
            "pipeline.analyze.iter_frames",
            side_effect=[iter(chunk_a_frames), iter(chunk_b_frames)],
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [_detection(activity="walking")]
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        html = run_full_video_to_html([Path("a.mp4"), Path("b.mp4")])

        # Both chunks were fed through the detector → 60 inferences total
        assert fake_detector.detect.call_count == 60
        assert isinstance(html, bytes) and len(html) > 0

    def test_progress_callback_invoked_at_least_once_per_chunk(self, mocker):
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_html

        fake_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        mocker.patch(
            "pipeline.analyze.iter_frames",
            side_effect=[
                iter([(0.0, fake_frame), (1.0, fake_frame)]),
                iter([(0.0, fake_frame), (1.0, fake_frame)]),
            ],
        )
        fake_detector = MagicMock()
        fake_detector.detect.return_value = []
        mocker.patch(
            "pipeline.analyze.load_pose_model",
            return_value=fake_detector,
        )

        seen: list[int] = []
        run_full_video_to_html(
            [Path("a.mp4"), Path("b.mp4")],
            progress=seen.append,
        )

        # At least one update per chunk, monotonically non-decreasing, ≤ 100,
        # final update at 100% so the worker can write status.json correctly.
        assert len(seen) >= 2
        assert seen == sorted(seen)
        assert all(0 <= p <= 100 for p in seen)
        assert seen[-1] == 100

    def test_pipeline_runtime_error_propagates_uncaught(self, mocker):
        from pathlib import Path

        from pipeline.analyze import run_full_video_to_html

        mocker.patch(
            "pipeline.analyze.load_pose_model",
            side_effect=RuntimeError("CUDAExecutionProvider not available"),
        )

        with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
            run_full_video_to_html([Path("a.mp4")])
