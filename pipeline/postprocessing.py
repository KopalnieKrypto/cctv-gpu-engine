"""YOLO-pose ONNX output parsing, NMS, and coordinate scaling.

The YOLO-pose ONNX output has shape ``[1, 56, N]`` where N is the number of
candidate detections. Channel layout per detection:

* rows 0-3: cx, cy, w, h (bbox center+size, in 640-space)
* row 4: confidence
* rows 5-55: 17 COCO keypoints × (x, y, visibility)

Coordinates are scaled back from 640-space to original image dimensions before
detections are returned to callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.preprocessing import IMG_SIZE

CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
NUM_KEYPOINTS = 17


@dataclass
class Keypoint:
    """A single COCO keypoint with absolute image coordinates."""

    x: float
    y: float
    vis: float


@dataclass
class Detection:
    """A single detected person with bbox and 17 COCO keypoints."""

    bbox: list[float]  # [x1, y1, x2, y2] in original image coords
    confidence: float
    keypoints: list[Keypoint]
    activity: str = "unknown"


def _iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-Union for two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection]) -> list[Detection]:
    """Greedy non-maximum suppression on a list of Detection objects.

    Higher-confidence detections are kept; any remaining detection that
    overlaps with a kept one above ``NMS_IOU_THRESHOLD`` is suppressed.
    """
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for candidate in sorted_dets:
        if all(_iou(candidate.bbox, k.bbox) <= NMS_IOU_THRESHOLD for k in kept):
            kept.append(candidate)
    return kept


def postprocess(
    output: np.ndarray,
    orig_w: int,
    orig_h: int,
) -> list[Detection]:
    """Parse a YOLO-pose ``[1, 56, N]`` tensor into Detection objects."""
    data = output[0]  # → [56, N]
    num_boxes = data.shape[1]
    sx = orig_w / IMG_SIZE
    sy = orig_h / IMG_SIZE

    detections: list[Detection] = []
    for i in range(num_boxes):
        conf = float(data[4, i])
        if conf < CONFIDENCE_THRESHOLD:
            continue
        cx = float(data[0, i])
        cy = float(data[1, i])
        w = float(data[2, i])
        h = float(data[3, i])

        x1 = (cx - w / 2) * sx
        y1 = (cy - h / 2) * sy
        x2 = (cx + w / 2) * sx
        y2 = (cy + h / 2) * sy

        kps: list[Keypoint] = []
        for k in range(NUM_KEYPOINTS):
            base = 5 + k * 3
            kps.append(
                Keypoint(
                    x=float(data[base, i]) * sx,
                    y=float(data[base + 1, i]) * sy,
                    vis=float(data[base + 2, i]),
                )
            )

        detections.append(
            Detection(
                bbox=[x1, y1, x2, y2],
                confidence=conf,
                keypoints=kps,
            )
        )

    return nms(detections)
