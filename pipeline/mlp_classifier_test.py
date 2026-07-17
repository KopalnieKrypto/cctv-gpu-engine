"""Behavior tests for checksum-pinned CUDA-only activity-MLP inference.

Runtime assumptions approved before RED for issue #34:

* One classifier call consumes one ``Detection`` and returns one of the frozen
  four labels. Batching and temporal smoothing are separate public concerns.
* The ONNX and its sidecar metadata must agree on weight checksum, feature
  schema, class order, and model version before a session is created.
* CUDA libraries are preloaded before session construction; both registered
  and active providers must contain ``CUDAExecutionProvider``. CPU fallback is
  an error.
* ONNX Runtime is the mocked external boundary in unit tests. Real provider and
  export parity are covered by the RTX 5070 GPU gate.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from pipeline.activity_features import FEATURE_DIM, feature_schema_manifest
from pipeline.mlp_classifier import load_activity_mlp
from pipeline.postprocessing import Detection, Keypoint


def test_classifier_loads_verified_cuda_model_and_predicts_one_person(tmp_path) -> None:
    model_path = tmp_path / "activity-mlp.onnx"
    model_path.write_bytes(b"model bytes")
    metadata_path = tmp_path / "model-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": {
                    "version": "activity-mlp-v1.0.0",
                    "sha256": hashlib.sha256(b"model bytes").hexdigest(),
                    "class_order": ["sitting", "standing", "walking", "running"],
                },
                "feature_schema": feature_schema_manifest(),
            }
        )
    )

    class FakeInput:
        name = "features"
        shape = ["batch", FEATURE_DIM]

    class FakeSession:
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_inputs(self):
            return [FakeInput()]

        def run(self, _outputs, inputs):
            assert inputs["features"].shape == (1, FEATURE_DIM)
            return [np.asarray([[0.1, 0.2, 0.6, 0.1]], dtype=np.float32)]

    class FakeOrt:
        preloaded = False
        providers = None

        @classmethod
        def get_available_providers(cls):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        @classmethod
        def preload_dlls(cls, *, cuda, cudnn):
            cls.preloaded = (cuda, cudnn)

        @classmethod
        def InferenceSession(cls, _path, *, providers):
            cls.providers = providers
            return FakeSession()

    classifier = load_activity_mlp(model_path, metadata_path, ort_module=FakeOrt)
    detection = Detection(
        bbox=[0.0, 0.0, 100.0, 200.0],
        confidence=1.0,
        keypoints=[Keypoint(x=10.0, y=20.0, vis=1.0) for _ in range(17)],
    )

    assert classifier.classify(detection) == "walking"
    assert classifier.model_version == "activity-mlp-v1.0.0"
    assert FakeOrt.preloaded == (True, True)
    assert FakeOrt.providers == ["CUDAExecutionProvider"]


def test_classifier_rejects_model_bytes_that_do_not_match_metadata(tmp_path) -> None:
    model_path = tmp_path / "activity-mlp.onnx"
    model_path.write_bytes(b"tampered")
    metadata_path = tmp_path / "model-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": {
                    "version": "activity-mlp-v1.0.0",
                    "sha256": hashlib.sha256(b"expected").hexdigest(),
                    "class_order": list(("sitting", "standing", "walking", "running")),
                },
                "feature_schema": feature_schema_manifest(),
            }
        )
    )

    with pytest.raises(RuntimeError, match="activity MLP checksum mismatch"):
        load_activity_mlp(model_path, metadata_path, ort_module=object())


def test_classifier_rejects_silent_cpu_fallback_after_session_creation(tmp_path) -> None:
    model_path = tmp_path / "activity-mlp.onnx"
    model_path.write_bytes(b"model")
    metadata_path = tmp_path / "model-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": {
                    "version": "activity-mlp-v1.0.0",
                    "sha256": hashlib.sha256(b"model").hexdigest(),
                    "class_order": ["sitting", "standing", "walking", "running"],
                },
                "feature_schema": feature_schema_manifest(),
            }
        )
    )

    class CpuSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        @staticmethod
        def preload_dlls(*, cuda, cudnn):
            assert (cuda, cudnn) == (True, True)

        @staticmethod
        def InferenceSession(_path, *, providers):
            assert providers == ["CUDAExecutionProvider"]
            return CpuSession()

    with pytest.raises(RuntimeError, match="registered but inactive"):
        load_activity_mlp(model_path, metadata_path, ort_module=FakeOrt)


def test_classifier_rejects_release_class_order_drift(tmp_path) -> None:
    model_path = tmp_path / "activity-mlp.onnx"
    model_path.write_bytes(b"model")
    metadata_path = tmp_path / "model-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": {
                    "version": "activity-mlp-v1.0.0",
                    "sha256": hashlib.sha256(b"model").hexdigest(),
                    "class_order": ["standing", "sitting", "walking", "running"],
                },
                "feature_schema": feature_schema_manifest(),
            }
        )
    )

    with pytest.raises(RuntimeError, match="activity MLP class order mismatch"):
        load_activity_mlp(model_path, metadata_path, ort_module=object())
