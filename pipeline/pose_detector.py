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

import hashlib
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from pipeline.activity_classifier import classify_activity
from pipeline.postprocessing import Detection, postprocess
from pipeline.preprocessing import IMG_SIZE, InputSize, preprocess
from pipeline.zones import ZoneConfig

CUDA_PROVIDER = "CUDAExecutionProvider"

# Hash the weights a megabyte at a time — a pose model is tens of MB and this
# runs once per job, but there is no reason to hold a second copy in RAM.
_SHA256_BLOCK_BYTES = 1024 * 1024


def file_sha256(path: str) -> str | None:
    """SHA-256 of the file at ``path``, or ``None`` when it cannot be read.

    Read from the bytes on disk, never from a constant or a build ARG:
    ``docker-compose.yml`` bind-mounts ``./models`` over the image's baked
    weights, so a box provisioned earlier — or hand-patched since — can serve
    different weights with nothing in the output saying so (issue #98).

    Unreadable weights return ``None`` rather than raising: this feeds
    diagnostics, and diagnostics must never be the reason a job dies. In
    production ORT has already opened the same file by the time we hash it.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(_SHA256_BLOCK_BYTES), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _validated_input_size(shape: object) -> tuple[int, int]:
    """Return ``(width, height)`` for a supported fixed ``[1, 3, H, W]`` input.

    Non-square exports are supported since issue #100 — a 16:9 frame into a
    square input spends 43.75% of the tensor on constant grey for no gain in
    detection scale. The dynamic-dimension rejection stays: a dynamic axis
    makes onnxruntime re-optimise on every shape change.
    """

    def invalid(reason: str) -> RuntimeError:
        return RuntimeError(
            f"Invalid pose model input shape {shape!r}: {reason}; expected a "
            "fixed rank 4 [1, 3, H, W] tensor with positive integer spatial "
            "dimensions. Dynamic inputs are unsupported."
        )

    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise invalid("input must have rank 4")
    if any(not isinstance(dim, int) or isinstance(dim, bool) for dim in shape):
        raise invalid("all dimensions must be fixed integer dimensions")
    batch, channels, height, width = shape
    if batch != 1:
        raise invalid("input must have batch dimension 1")
    if channels != 3:
        raise invalid("input must have 3 channels")
    if height <= 0 or width <= 0:
        raise invalid("width and height must be positive")
    return width, height


@dataclass
class PoseDetector:
    """Stateful wrapper around an ONNX Runtime session for YOLO-pose."""

    session: ort.InferenceSession
    input_name: str
    # The model's declared input, as a square side or an explicit ``(w, h)``
    # since issue #100. :func:`load_pose_model` always stamps the pair; a bare
    # int is still accepted so hand-built detectors keep working.
    input_size: InputSize = IMG_SIZE
    # Optional ROI zones (issue #78). When set, each detection is stamped with
    # the zone its foot point falls in; when None, ``zone_id`` stays None.
    zones: ZoneConfig | None = None
    # Identity of the weights this session was created from (issue #98), so a
    # result.json can be attributed to the detector that produced it. Stamped
    # by :func:`load_pose_model`; ``None`` for hand-built detectors in tests.
    model_path: str | None = None
    model_sha256: str | None = None

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        """Run pose inference on a single BGR frame."""
        inference_frame = img_bgr
        offset_x = 0
        offset_y = 0
        if self.zones is not None:
            frame_h, frame_w = img_bgr.shape[:2]
            bounds = self.zones.inference_bounds(frame_w, frame_h)
            if bounds is not None:
                x1, y1, x2, y2 = bounds
                inference_frame = img_bgr[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1

        tensor, orig_w, orig_h = preprocess(inference_frame, input_size=self.input_size)
        outputs = self.session.run(None, {self.input_name: tensor})
        detections = postprocess(
            outputs[0], orig_w=orig_w, orig_h=orig_h, input_size=self.input_size
        )
        for det in detections:
            if offset_x or offset_y:
                det.bbox[0] += offset_x
                det.bbox[1] += offset_y
                det.bbox[2] += offset_x
                det.bbox[3] += offset_y
                for keypoint in det.keypoints:
                    keypoint.x += offset_x
                    keypoint.y += offset_y
            det.activity = classify_activity(det)
            if self.zones is not None:
                det.zone_id = self.zones.zone_for_detection(det)
        return detections


def load_pose_model(model_path: str, zones: ZoneConfig | None = None) -> PoseDetector:
    """Load a YOLO-pose ONNX model on the CUDA execution provider.

    ``zones`` — optional ROI config (issue #78). When given, the returned
    detector stamps each detection's ``zone_id`` from its foot point.

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

    model_inputs = session.get_inputs()
    if len(model_inputs) != 1:
        raise RuntimeError(
            "Invalid pose model interface: expected exactly one image input, "
            f"found {len(model_inputs)}."
        )
    model_input = model_inputs[0]
    input_name = model_input.name
    input_size = _validated_input_size(model_input.shape)
    return PoseDetector(
        session=session,
        input_name=input_name,
        input_size=input_size,
        zones=zones,
        model_path=model_path,
        model_sha256=file_sha256(model_path),
    )
