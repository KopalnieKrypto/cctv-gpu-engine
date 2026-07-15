"""OSNet Re-ID appearance embedding for person tracking (issue #32).

Turns each detected person's bbox crop into an appearance vector. Cosine
similarity between two vectors is what :mod:`pipeline.tracker` uses to decide
"same person" — the locked association metric from the 2026-05-27 advisory:
IoU is useless at 1 fps (a person moves too far between samples) and a
dedicated OSNet-class network beats reusing YOLO's intermediate features for
identity consistency.

Vectors are returned L2-normalized so the tracker's dot product *is* cosine
similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from pipeline.postprocessing import Detection

CUDA_PROVIDER = "CUDAExecutionProvider"

DEFAULT_REID_MODEL_PATH = "models/osnet_x0_25.onnx"

# OSNet's native input geometry: tall and narrow, matching a standing person.
REID_INPUT_H = 256
REID_INPUT_W = 128

# OSNet is trained on ImageNet-normalized crops; feeding it plain [0,1] pixels
# yields embeddings that still look plausible but discriminate far worse.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class OSNetEmbedder:
    """Stateful wrapper around an ONNX Runtime session for OSNet Re-ID."""

    session: ort.InferenceSession
    input_name: str

    def embed(self, frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        """Embed every detection's crop in one batched forward pass."""
        if not detections:
            return []
        batch = np.stack([self._crop(frame, det) for det in detections])
        features = self.session.run(None, {self.input_name: batch})[0]
        return [self._normalize(feature) for feature in features]

    def _crop(self, frame: np.ndarray, det: Detection) -> np.ndarray:
        """One person's bbox → normalized CHW tensor at OSNet input size."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        # Clamp into the image. YOLO overhangs the frame edge for anyone
        # entering or leaving shot, and a negative index would wrap around to
        # the opposite side of the frame rather than clip — embedding scenery
        # instead of the person. The +1 floors keep a degenerate box at one
        # pixel instead of an empty array that would crash cv2 mid-video.
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, x1 + 1), width)
        y2 = min(max(y2, y1 + 1), height)
        crop = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (REID_INPUT_W, REID_INPUT_H), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return arr.transpose(2, 0, 1)  # HWC → CHW

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray:
        vector = np.asarray(feature, dtype=np.float32)
        return vector / np.linalg.norm(vector)


def load_reid_model(model_path: str = DEFAULT_REID_MODEL_PATH) -> OSNetEmbedder:
    """Load an OSNet Re-ID ONNX model on the CUDA execution provider.

    Mirrors :func:`pipeline.pose_detector.load_pose_model`, including its
    refusal to run on CPU: a silent fallback here would not fail, it would just
    make every frame slow enough to blow the throughput budget.

    Raises:
        RuntimeError: if ``CUDAExecutionProvider`` is not registered with the
            installed onnxruntime build, or if it registers but the created
            session does not actually use it (silent CPU fallback,
            microsoft/onnxruntime#25145).
    """
    available = ort.get_available_providers()
    if CUDA_PROVIDER not in available:
        raise RuntimeError(
            f"{CUDA_PROVIDER} not available — GPU required, no CPU fallback. "
            f"Available providers: {available}. "
            "Install onnxruntime-gpu and ensure CUDA 12.x is on the system."
        )

    # Same missing-RPATH dance as the pose model: resolve the nvidia-* pip
    # packages before the session starts, or onnxruntime dlopens nothing and
    # quietly lands on CPU.
    ort.preload_dlls(cuda=True, cudnn=True)

    session = ort.InferenceSession(model_path, providers=[CUDA_PROVIDER])
    active = session.get_providers()
    if CUDA_PROVIDER not in active:
        raise RuntimeError(
            f"{CUDA_PROVIDER} registered but inactive after session init "
            f"(active providers: {active}). This is a silent CPU fallback "
            "(microsoft/onnxruntime#25145) — refusing to run."
        )

    return OSNetEmbedder(session=session, input_name=session.get_inputs()[0].name)
