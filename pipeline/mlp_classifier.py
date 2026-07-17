"""Checksum-pinned CUDA-only ONNX activity-MLP inference."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.activity_features import (
    ACTIVITY_CLASSES,
    FEATURE_DIM,
    extract_activity_features,
    feature_schema_manifest,
)
from pipeline.postprocessing import Detection

CUDA_PROVIDER = "CUDAExecutionProvider"


class TrackActivitySmoother:
    """Majority smoothing keyed only by stable tracker identity."""

    def __init__(self, window: int = 5) -> None:
        self._window = window
        self._history: dict[int, deque[str]] = {}

    def smooth(self, detections: list[Detection]) -> list[Detection]:
        for detection in detections:
            if detection.track_id is None:
                continue
            raw_activity = detection.activity
            history = self._history.setdefault(detection.track_id, deque(maxlen=self._window))
            history.append(raw_activity)
            counts = Counter(history)
            detection.activity = max(
                counts,
                key=lambda activity: (counts[activity], activity == raw_activity),
            )
        return detections


@dataclass
class MLPClassifier:
    session: object
    input_name: str
    class_order: tuple[str, ...]
    model_version: str
    model_sha256: str

    def classify(self, detection: Detection) -> str:
        """Classify one detection independently."""
        features = extract_activity_features(detection)[None, :]
        probabilities = self.session.run(None, {self.input_name: features})[0]
        return self.class_order[int(np.argmax(probabilities[0]))]


def load_activity_mlp(
    model_path: str | Path,
    metadata_path: str | Path,
    *,
    ort_module: object | None = None,
) -> MLPClassifier:
    """Verify metadata/weights and construct a CUDA-only ONNX session."""
    if ort_module is None:
        import onnxruntime as ort_module

    model = Path(model_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    if model_sha256 != metadata["model"]["sha256"]:
        raise RuntimeError(
            "activity MLP checksum mismatch: "
            f"expected {metadata['model']['sha256']}, found {model_sha256}"
        )
    if metadata["feature_schema"] != feature_schema_manifest():
        raise RuntimeError("activity MLP feature schema does not match runtime schema")
    class_order = tuple(metadata["model"]["class_order"])
    if class_order != ACTIVITY_CLASSES:
        raise RuntimeError(
            f"activity MLP class order mismatch: expected {ACTIVITY_CLASSES}, found {class_order}"
        )

    available = ort_module.get_available_providers()
    if CUDA_PROVIDER not in available:
        raise RuntimeError(
            f"{CUDA_PROVIDER} not available — GPU required, no CPU fallback. "
            f"Available providers: {available}"
        )
    ort_module.preload_dlls(cuda=True, cudnn=True)
    session = ort_module.InferenceSession(str(model), providers=[CUDA_PROVIDER])
    active = session.get_providers()
    if CUDA_PROVIDER not in active:
        raise RuntimeError(
            f"{CUDA_PROVIDER} registered but inactive after activity MLP session init "
            f"(active providers: {active}) — refusing CPU fallback"
        )
    model_inputs = session.get_inputs()
    if len(model_inputs) != 1 or model_inputs[0].shape[1] != FEATURE_DIM:
        raise RuntimeError(f"invalid activity MLP input contract; expected [batch, {FEATURE_DIM}]")
    return MLPClassifier(
        session=session,
        input_name=model_inputs[0].name,
        class_order=class_order,
        model_version=metadata["model"]["version"],
        model_sha256=model_sha256,
    )
