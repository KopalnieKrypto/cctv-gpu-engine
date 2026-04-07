"""YOLO-pose ONNX inference on GPU.

Loads a ``yolo11n-pose.onnx`` model with the CUDA execution provider and
exposes a :class:`PoseDetector` that turns BGR frames into a list of
:class:`pipeline.postprocessing.Detection` objects with activity labels.

There is **no CPU fallback**: if the CUDA provider is unavailable or if a
session silently falls back to CPU, model loading raises ``RuntimeError`` so
the pipeline fails fast and visibly instead of silently running 10× slower.
See microsoft/onnxruntime#25145 for the silent-fallback bug we explicitly
guard against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from pipeline.activity_classifier import classify_activity
from pipeline.postprocessing import Detection, postprocess
from pipeline.preprocessing import preprocess

CUDA_PROVIDER = "CUDAExecutionProvider"


@dataclass
class PoseDetector:
    """Stateful wrapper around an ONNX Runtime session for YOLO-pose."""

    session: ort.InferenceSession
    input_name: str

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        """Run pose inference on a single BGR frame."""
        tensor, orig_w, orig_h = preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        detections = postprocess(outputs[0], orig_w=orig_w, orig_h=orig_h)
        for det in detections:
            det.activity = classify_activity(det)
        return detections


def load_pose_model(model_path: str) -> PoseDetector:
    """Load a YOLO-pose ONNX model on the CUDA execution provider.

    Raises:
        RuntimeError: if ``CUDAExecutionProvider`` is not registered with the
            installed onnxruntime build, or if it registers but the created
            session does not actually use it (silent CPU fallback).
    """
    available = ort.get_available_providers()
    if CUDA_PROVIDER not in available:
        raise RuntimeError(
            f"{CUDA_PROVIDER} not available — GPU required, no CPU fallback. "
            f"Available providers: {available}. "
            "Install onnxruntime-gpu and ensure CUDA 12.x is on the system."
        )

    # The onnxruntime-gpu wheel does not embed an RPATH to the isolated
    # nvidia-* pip packages we ship in this venv (cublas, cudnn, etc.), so
    # without an explicit preload it would fail to dlopen libcublasLt.so.12
    # and silently fall back to CPU. Calling ort.preload_dlls() resolves and
    # loads them from site-packages/nvidia/.../lib before the session starts.
    ort.preload_dlls(cuda=True, cudnn=True)

    session = ort.InferenceSession(model_path, providers=[CUDA_PROVIDER])
    active = session.get_providers()
    if CUDA_PROVIDER not in active:
        raise RuntimeError(
            f"{CUDA_PROVIDER} registered but inactive after session init "
            f"(active providers: {active}). This is a silent CPU fallback "
            "(microsoft/onnxruntime#25145) — refusing to run."
        )

    input_name = session.get_inputs()[0].name
    return PoseDetector(session=session, input_name=input_name)
