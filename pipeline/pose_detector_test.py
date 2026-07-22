"""Tests for the PoseDetector — model loading and end-to-end orchestration.

The CUDA execution provider and onnxruntime are mocked at the boundary so
these tests can run on any developer machine. A real GPU smoke test lives
under the ``gpu`` marker and is skipped by default.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import numpy as np
import pytest

from pipeline.pose_detector import PoseDetector, load_pose_model


def _mock_cuda_session(mocker, shape: list | None = None) -> MagicMock:
    """Patch ``ort`` so the real loader runs on a machine with no CUDA."""
    mocker.patch(
        "pipeline.pose_detector.ort.get_available_providers",
        return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    mocker.patch("pipeline.pose_detector.ort.preload_dlls")
    fake_session = MagicMock()
    fake_session.get_providers.return_value = ["CUDAExecutionProvider"]
    fake_input = MagicMock()
    fake_input.name = "images"
    fake_input.shape = shape or [1, 3, 640, 640]
    fake_session.get_inputs.return_value = [fake_input]
    mocker.patch("pipeline.pose_detector.ort.InferenceSession", return_value=fake_session)
    return fake_session


class TestLoadPoseModel:
    """Public loader contract for fixed-shape YOLO-pose models.

    Issue #86 assumptions recorded before the first RED:

    * Input: exactly one fixed rank-4 image tensor shaped ``[1, 3, S, S]``;
      ``S`` is a positive integer. Fixed 640 and 1280 are supported.
    * Output: the loader returns a detector that feeds the session at its
      declared ``S`` and returns detections in original-frame coordinates.
    * Errors: dynamic, non-square, wrong-rank, wrong-batch/channel, and
      otherwise malformed shapes fail at load time with an actionable error.
    * Boundaries intentionally deferred to later REDs: ROI cropping, benchmark
      eligibility, production-default selection, and real CUDA performance.
    """

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
        fake_input = MagicMock()
        fake_input.name = "images"
        fake_input.shape = [1, 3, 640, 640]
        fake_session.get_inputs.return_value = [fake_input]
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
        fake_input = MagicMock()
        fake_input.name = "images"
        fake_input.shape = [1, 3, 640, 640]
        fake_session.get_inputs.return_value = [fake_input]

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

    def test_fixed_1280_model_receives_1280_tensor_and_returns_original_coordinates(self, mocker):
        mocker.patch(
            "pipeline.pose_detector.ort.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        fake_session = MagicMock()
        fake_session.get_providers.return_value = ["CUDAExecutionProvider"]
        fake_input = MagicMock()
        fake_input.name = "images"
        fake_input.shape = [1, 3, 1280, 1280]
        fake_session.get_inputs.return_value = [fake_input]

        # A box at [480, 270, 1440, 810] in a 1920x1080 frame becomes
        # [320, 460, 960, 820] after a 1280 letterbox (scale=2/3, pad_y=280).
        keypoints = [(640.0, 640.0, 0.9)] * 17
        output = np.zeros((1, 56, 1), dtype=np.float32)
        output[0, 0:5, 0] = [640.0, 640.0, 640.0, 360.0, 0.9]
        for index, (x, y, visibility) in enumerate(keypoints):
            base = 5 + index * 3
            output[0, base : base + 3, 0] = [x, y, visibility]
        fake_session.run.return_value = [output]
        mocker.patch("pipeline.pose_detector.ort.InferenceSession", return_value=fake_session)

        detector = load_pose_model("models/fixed-1280.onnx")
        [detection] = detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

        feed = fake_session.run.call_args.args[1]
        assert feed["images"].shape == (1, 3, 1280, 1280)
        assert detection.bbox == pytest.approx([480.0, 270.0, 1440.0, 810.0])

    @pytest.mark.parametrize(
        ("shape", "reason"),
        [
            ([1, 3, "height", "width"], "fixed integer dimensions"),
            ([1, 3, 640], "rank 4"),
            ([2, 3, 640, 640], "batch dimension 1"),
            ([1, 1, 640, 640], "3 channels"),
            # [1, 3, 640, 1280] was rejected here until issue #100 — a
            # non-square export is now a supported shape, not a malformed one.
            ([1, 3, 0, 0], "positive"),
        ],
    )
    def test_invalid_model_shapes_fail_fast_with_the_violated_constraint(
        self, mocker, shape, reason
    ):
        mocker.patch(
            "pipeline.pose_detector.ort.get_available_providers",
            return_value=["CUDAExecutionProvider"],
        )
        fake_session = MagicMock()
        fake_session.get_providers.return_value = ["CUDAExecutionProvider"]
        fake_input = MagicMock()
        fake_input.name = "images"
        fake_input.shape = shape
        fake_session.get_inputs.return_value = [fake_input]
        mocker.patch("pipeline.pose_detector.ort.InferenceSession", return_value=fake_session)

        with pytest.raises(RuntimeError, match=reason):
            load_pose_model("models/invalid-shape.onnx")

    @pytest.mark.parametrize("input_count", [0, 2])
    def test_loader_rejects_models_without_exactly_one_image_input(self, mocker, input_count):
        mocker.patch(
            "pipeline.pose_detector.ort.get_available_providers",
            return_value=["CUDAExecutionProvider"],
        )
        fake_session = MagicMock()
        fake_session.get_providers.return_value = ["CUDAExecutionProvider"]
        inputs = []
        for index in range(input_count):
            model_input = MagicMock()
            model_input.name = f"images_{index}"
            model_input.shape = [1, 3, 640, 640]
            inputs.append(model_input)
        fake_session.get_inputs.return_value = inputs
        mocker.patch("pipeline.pose_detector.ort.InferenceSession", return_value=fake_session)

        with pytest.raises(RuntimeError, match="exactly one image input"):
            load_pose_model("models/wrong-input-count.onnx")


class TestLoadedModelIdentity:
    """Issue #98 — a detector must be able to name the weights it loaded.

    ``docker-compose.yml`` bind-mounts ``./models`` over the image's baked
    weights, so neither the Dockerfile ARG nor the env default proves anything
    about what actually ran. The only provable answer is the sha256 of the
    bytes on disk at the moment the session was created.
    """

    def test_stamps_the_resolved_path_and_the_sha256_of_the_loaded_file(self, mocker, tmp_path):
        weights = tmp_path / "yolo11s-pose.onnx"
        weights.write_bytes(b"pretend onnx bytes")
        _mock_cuda_session(mocker)

        detector = load_pose_model(str(weights))

        assert detector.model_path == str(weights)
        assert detector.model_sha256 == hashlib.sha256(b"pretend onnx bytes").hexdigest()

    def test_same_path_with_different_bytes_reports_a_different_sha256(self, mocker, tmp_path):
        # The bind-mount case: the path an operator sees is identical, the
        # weights behind it are not. A sha taken from a constant or from the
        # path would report these two runs as the same configuration.
        weights = tmp_path / "yolo11s-pose.onnx"
        _mock_cuda_session(mocker)

        weights.write_bytes(b"baked into the image")
        baked = load_pose_model(str(weights)).model_sha256
        weights.write_bytes(b"bind-mounted over it")
        mounted = load_pose_model(str(weights)).model_sha256

        assert baked == hashlib.sha256(b"baked into the image").hexdigest()
        assert mounted == hashlib.sha256(b"bind-mounted over it").hexdigest()
        assert baked != mounted

    def test_sha256_is_none_when_the_file_cannot_be_read(self, mocker):
        # Best-effort: diagnostics must never be the reason a job dies. In
        # production ORT has already opened this file by the time we hash it.
        _mock_cuda_session(mocker)

        detector = load_pose_model("dummy/path/to/model.onnx")

        assert detector.model_path == "dummy/path/to/model.onnx"
        assert detector.model_sha256 is None

    def test_input_size_comes_from_the_model_not_from_the_preprocessing_default(
        self, mocker, tmp_path
    ):
        from pipeline.preprocessing import IMG_SIZE

        weights = tmp_path / "yolo11s-pose-1280.onnx"
        weights.write_bytes(b"a 1280 export")
        _mock_cuda_session(mocker, shape=[1, 3, 1280, 1280])

        detector = load_pose_model(str(weights))

        assert detector.input_size == (1280, 1280)
        assert IMG_SIZE == 640  # the default the value must not have come from


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


class TestPoseDetectorZoneAssignment:
    """Issue #78 — the detector stamps ``Detection.zone_id`` from a foot point.

    A zone config is optional: with none supplied every detection keeps
    ``zone_id`` ``None`` (existing behaviour). With one, each detection is
    assigned by the midpoint of its bbox bottom edge, so a person standing in
    the ROI is attributed to it and a person outside every ROI stays ``None``.
    """

    def _output_for_bbox(self, cx: float, cy: float, w: float, h: float) -> np.ndarray:
        data = np.zeros((1, 56, 1), dtype=np.float32)
        data[0, 0, 0] = cx
        data[0, 1, 0] = cy
        data[0, 2, 0] = w
        data[0, 3, 0] = h
        data[0, 4, 0] = 0.9  # confidence above threshold
        return data

    def _detector_with_zone(self, polygon):
        from pipeline.zones import ZoneConfig

        config = ZoneConfig.from_dict(
            {"zones": [{"id": "bending-1", "name": "Giętarka 1", "polygon": polygon}]}
        )
        fake_session = MagicMock()
        # cx=320, cy=320, w=80, h=400 → bbox [280,120,360,520], foot=(320,520)
        fake_session.run.return_value = [self._output_for_bbox(320, 320, 80, 400)]
        return PoseDetector(session=fake_session, input_name="images", zones=config)

    def test_zone_id_none_when_no_zone_config(self):
        fake_session = MagicMock()
        fake_session.run.return_value = [self._output_for_bbox(320, 320, 80, 400)]
        detector = PoseDetector(session=fake_session, input_name="images")

        [det] = detector.detect(np.full((640, 640, 3), 128, dtype=np.uint8))

        assert det.zone_id is None

    def test_zone_id_set_when_foot_point_inside_zone(self):
        # Lower band contains the foot point (320, 520).
        detector = self._detector_with_zone([[0, 300], [640, 300], [640, 640], [0, 640]])

        [det] = detector.detect(np.full((640, 640, 3), 128, dtype=np.uint8))

        assert det.zone_id == "bending-1"

    def test_zone_id_none_when_foot_point_outside_zone(self):
        # Upper band does NOT contain the foot point (320, 520).
        detector = self._detector_with_zone([[0, 0], [640, 0], [640, 300], [0, 300]])

        [det] = detector.detect(np.full((640, 640, 3), 128, dtype=np.uint8))

        assert det.zone_id is None


class TestPoseDetectorInferenceROI:
    """Issue #86 inference-ROI contract, stated before its first RED.

    The optional top-level ``inference_roi`` names one existing semantic zone
    and a non-negative pixel margin. Its polygon bounding box plus margin is
    clipped to each frame and sent through exactly one model call. Returned
    bbox/keypoints are translated back to full-frame pixels before semantic
    zone assignment. With the field absent, full-frame inference is unchanged.
    """

    def test_roi_crop_maps_bbox_and_keypoints_back_before_zone_assignment(self):
        from pipeline.zones import ZoneConfig

        zones = ZoneConfig.from_dict(
            {
                "inference_roi": {"zone_id": "bending-1", "margin_px": 50},
                "zones": [
                    {
                        "id": "bending-1",
                        "name": "Giętarka 1",
                        "polygon": [[400, 300], [600, 300], [600, 500], [400, 500]],
                    }
                ],
            }
        )
        # The crop is [350,250]–[650,550], a 300x300 square. This model-space
        # box represents crop-local [50,50]–[250,250], hence full-frame
        # [400,300]–[600,500]. Its foot lies exactly on the semantic boundary.
        output = np.zeros((1, 56, 1), dtype=np.float32)
        scale = 640 / 300
        output[0, 0:5, 0] = [320, 320, 200 * scale, 200 * scale, 0.9]
        for index in range(17):
            base = 5 + index * 3
            output[0, base : base + 3, 0] = [320, 320, 0.9]
        fake_session = MagicMock()
        fake_session.run.return_value = [output]
        detector = PoseDetector(session=fake_session, input_name="images", zones=zones)

        [detection] = detector.detect(np.zeros((800, 1000, 3), dtype=np.uint8))

        fake_session.run.assert_called_once()
        assert detection.bbox == pytest.approx([400, 300, 600, 500], abs=1e-3)
        assert [(kp.x, kp.y) for kp in detection.keypoints] == pytest.approx(
            [(500, 400)] * 17,
            abs=1e-3,
        )
        assert detection.zone_id == "bending-1"


class TestNonSquareModelInput:
    """Issue #100 — the loader accepts an explicit ``[1, 3, H, W]`` export.

    A 16:9 frame into a square input spends 43.75% of every tensor on constant
    grey. A non-square export at the same width has identical detection scale
    for 0.60× the compute, so the loader must stop rejecting it — while keeping
    the guard that actually matters (dynamic dimensions) and keeping square
    exports loading exactly as before.
    """

    def test_non_square_export_loads_and_reports_its_width_and_height(self, mocker):
        _mock_cuda_session(mocker, shape=[1, 3, 736, 1280])

        detector = load_pose_model("models/yolo11s-pose-1280x736.onnx")

        assert detector.input_size == (1280, 736)

    def test_non_square_detector_feeds_the_session_that_exact_tensor(self, mocker):
        session = _mock_cuda_session(mocker, shape=[1, 3, 736, 1280])
        session.run.return_value = [np.zeros((1, 56, 0), dtype=np.float32)]

        detector = load_pose_model("models/yolo11s-pose-1280x736.onnx")
        detector.detect(np.zeros((2160, 3840, 3), dtype=np.uint8))

        feed = session.run.call_args.args[1]
        assert feed["images"].shape == (1, 3, 736, 1280)

    def test_square_export_still_loads_and_reports_a_square_pair(self, mocker):
        # Widening, not replacing — 640 and 1280 square exports are unaffected.
        _mock_cuda_session(mocker, shape=[1, 3, 640, 640])

        assert load_pose_model("models/yolo11s-pose.onnx").input_size == (640, 640)

    def test_dynamic_dimensions_are_still_rejected(self, mocker):
        # The guard that survives: a dynamic axis means the session would
        # re-optimise on every shape change, which is a different failure.
        _mock_cuda_session(mocker, shape=[1, 3, "height", "width"])

        with pytest.raises(RuntimeError, match="fixed integer dimensions"):
            load_pose_model("models/dynamic.onnx")

    def test_non_square_bboxes_come_back_in_original_frame_coordinates(self, mocker):
        # 3840×2160 into 1280×736: scale = 1/3, the image occupies 1280×720 of
        # the input with 8 rows of padding top and bottom. A model-space box of
        # [320, 128, 480, 488] is [960, 360, 1440, 1440] in the source frame.
        session = _mock_cuda_session(mocker, shape=[1, 3, 736, 1280])
        output = np.zeros((1, 56, 1), dtype=np.float32)
        output[0, 0:5, 0] = [400.0, 308.0, 160.0, 360.0, 0.9]
        session.run.return_value = [output]

        detector = load_pose_model("models/yolo11s-pose-1280x736.onnx")
        [detection] = detector.detect(np.zeros((2160, 3840, 3), dtype=np.uint8))

        assert detection.bbox == pytest.approx([960.0, 360.0, 1440.0, 1440.0])
