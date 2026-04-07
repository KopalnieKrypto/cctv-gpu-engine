"""Keyframe annotation: bounding box, COCO skeleton overlay, activity label.

Used by the report renderer to overlay detections on top of selected
keyframes before they are encoded as base64 PNG. The annotator is
GPU-independent — pure OpenCV draw calls on a numpy frame.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from pipeline.postprocessing import Detection

# COCO 17 skeleton edges (pairs of keypoint indices that should be connected
# with a line). Source: ultralytics/yolov8 pose visualisation defaults.
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 11),
    (6, 12),  # shoulders→hips
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # face
)

# BGR colours per activity (matches the SPEC §5.8 colour scheme).
ACTIVITY_COLORS: dict[str, tuple[int, int, int]] = {
    "sitting": (255, 128, 0),  # blue
    "standing": (0, 200, 0),  # green
    "walking": (0, 165, 255),  # orange
    "running": (0, 0, 255),  # red
    "unknown": (200, 200, 200),  # grey
}

KEYPOINT_VIS_MIN = 0.3


def annotate_frame(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Return a new BGR frame with bbox + skeleton + label drawn for each detection."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    for det in detections:
        color = ACTIVITY_COLORS.get(det.activity, ACTIVITY_COLORS["unknown"])
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        # Bounding box (2px solid)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2)

        # COCO skeleton overlay — only edges where both endpoints are visible
        kps = det.keypoints
        for a, b in COCO_SKELETON:
            if a >= len(kps) or b >= len(kps):
                continue
            ka, kb = kps[a], kps[b]
            if ka.vis < KEYPOINT_VIS_MIN or kb.vis < KEYPOINT_VIS_MIN:
                continue
            cv2.line(
                canvas,
                (int(round(ka.x)), int(round(ka.y))),
                (int(round(kb.x)), int(round(kb.y))),
                color,
                thickness=2,
            )

        # Visible keypoints as small circles
        for kp in kps:
            if kp.vis < KEYPOINT_VIS_MIN:
                continue
            cv2.circle(canvas, (int(round(kp.x)), int(round(kp.y))), 3, color, thickness=-1)

        # Activity label above the bbox
        label = det.activity
        label_y = max(0, y1 - 6)
        cv2.putText(
            canvas,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness=2,
        )

    return canvas
