"""Tests for the PoseDetector — model loading and end-to-end orchestration.

The CUDA execution provider and onnxruntime are mocked at the boundary so
these tests can run on any developer machine. A real GPU smoke test lives
under the ``gpu`` marker and is skipped by default.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from pipeline.pose_detector import PoseDetector, load_pose_model


class TestLoadPoseModel:
    def test_raises_runtime_error_when_cuda_provider_not_available(self, mocker):
        # Simulate a machine where ONNX Runtime has no CUDAExecutionProvider.
        mocker.patch(
            "pipeline.pose_detector.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        )

        with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
            load_pose_model("dummy/path/to/model.onnx")

    def test_raises_runtime_error_when_session_falls_back_to_cpu(self, mocker):
        # CUDA appears available, but the session silently falls back to CPU
        # (microsoft/onnxruntime#25145). We must catch this and refuse to run.
        mocker.patch(
            "pipeline.pose_detector.ort.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        fake_session = MagicMock()
        fake_session.get_providers.return_value = ["CPUExecutionProvider"]
        mocker.patch(
            "pipeline.pose_detector.ort.InferenceSession",
            return_value=fake_session,
        )

        with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
            load_pose_model("dummy/path/to/model.onnx")

    def test_returns_session_when_cuda_provider_active(self, mocker):
        mocker.patch(
            "pipeline.pose_detector.ort.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        fake_session = MagicMock()
        fake_session.get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        mocker.patch(
            "pipeline.pose_detector.ort.InferenceSession",
            return_value=fake_session,
        )

        detector = load_pose_model("dummy/path/to/model.onnx")

        assert detector is not None
        assert detector.session is fake_session

    def test_preloads_cuda_and_cudnn_dlls_before_creating_session(self, mocker):
        # When onnxruntime-gpu is installed alongside isolated nvidia-* pip
        # packages, the wheel does not embed an RPATH to site-packages/nvidia/.
        # We must call ort.preload_dlls(cuda=True, cudnn=True) BEFORE
        # InferenceSession() or the provider library load fails with
        # libcublasLt.so.12 not found and silently falls back to CPU.
        # Verified live on RTX 5070 / VPS 2026-04-07.
        mocker.patch(
            "pipeline.pose_detector.ort.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        fake_session = MagicMock()
        fake_session.get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        preload_mock = mocker.patch("pipeline.pose_detector.ort.preload_dlls")
        session_mock = mocker.patch(
            "pipeline.pose_detector.ort.InferenceSession",
            return_value=fake_session,
        )

        # Attach both child mocks to a parent so call ordering is observable.
        manager = MagicMock()
        manager.attach_mock(preload_mock, "preload")
        manager.attach_mock(session_mock, "session")

        load_pose_model("dummy/path/to/model.onnx")

        preload_mock.assert_called_once_with(cuda=True, cudnn=True)
        session_mock.assert_called_once()
        call_names = [c[0] for c in manager.mock_calls]
        assert call_names.index("preload") < call_names.index("session")


class TestPoseDetectorDetect:
    def _build_yolo_output(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        conf: float,
        keypoints: list[tuple[float, float, float]],
    ) -> np.ndarray:
        data = np.zeros((1, 56, 1), dtype=np.float32)
        data[0, 0, 0] = cx
        data[0, 1, 0] = cy
        data[0, 2, 0] = w
        data[0, 3, 0] = h
        data[0, 4, 0] = conf
        for k, (kx, ky, kv) in enumerate(keypoints):
            base = 5 + k * 3
            data[0, base, 0] = kx
            data[0, base + 1, 0] = ky
            data[0, base + 2, 0] = kv
        return data

    def test_detect_runs_full_pipeline_and_classifies_activity(self):
        # Build a synthetic ONNX output for a single, clearly-standing person.
        # bbox in 640-space: cx=320, cy=320, w=80, h=400 → portrait
        keypoints = [(0.0, 0.0, 0.0)] * 17
        # shoulders, hips, knees, ankles all visible and aligned vertically
        keypoints[5] = (310, 180, 0.95)  # L shoulder
        keypoints[6] = (330, 180, 0.95)  # R shoulder
        keypoints[11] = (310, 320, 0.95)  # L hip
        keypoints[12] = (330, 320, 0.95)  # R hip
        keypoints[13] = (310, 420, 0.95)  # L knee
        keypoints[14] = (330, 420, 0.95)  # R knee
        keypoints[15] = (310, 520, 0.95)  # L ankle
        keypoints[16] = (330, 520, 0.95)  # R ankle
        fake_output = self._build_yolo_output(
            cx=320, cy=320, w=80, h=400, conf=0.92, keypoints=keypoints
        )

        fake_session = MagicMock()
        fake_session.run.return_value = [fake_output]
        detector = PoseDetector(session=fake_session, input_name="images")

        # 640×640 BGR frame so coordinates are not rescaled
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        detections = detector.detect(img)

        # Session was called once with a [1,3,640,640] float32 tensor
        fake_session.run.assert_called_once()
        run_kwargs = fake_session.run.call_args
        feed = run_kwargs[0][1]
        assert "images" in feed
        assert feed["images"].shape == (1, 3, 640, 640)
        assert feed["images"].dtype == np.float32

        # One person detected, classified as standing
        assert len(detections) == 1
        det = detections[0]
        assert det.confidence == pytest.approx(0.92, abs=1e-5)
        assert det.activity == "standing"
        assert len(det.keypoints) == 17

    def test_detect_returns_empty_list_when_session_yields_no_detections(self):
        empty_output = np.zeros((1, 56, 0), dtype=np.float32)
        fake_session = MagicMock()
        fake_session.run.return_value = [empty_output]
        detector = PoseDetector(session=fake_session, input_name="images")

        img = np.full((720, 1280, 3), 0, dtype=np.uint8)
        detections = detector.detect(img)

        assert detections == []
