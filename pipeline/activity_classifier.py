"""Rule-based activity classification from COCO 17 keypoints.

Classifies a single Detection into one of:
``sitting``, ``standing``, ``walking``, ``running`` (or ``unknown`` only when
the bbox is degenerate). All thresholds live as module-level constants so they
can be tuned without touching the decision logic.

Decision tree (per SPEC.md §5.6):

* If hips, knees, ankles are not all visible (vis ≥ 0.5): bbox aspect-ratio
  fallback (h/w < 1.5 → sitting, else standing).
* knee_angle = average angle(hip, knee, ankle) for both legs.
* hip_height_ratio = (bbox_bottom - avg_hip_y) / bbox_height.
* stride_ratio = |ankle_L_x - ankle_R_x| / |hip_L_x - hip_R_x|.
* torso_lean = angle from vertical between shoulders→hips midline.
* knee_angle < 120° AND hip_height_ratio < 0.40 → sitting.
* stride_ratio > 2.0 AND torso_lean > 15° → running.
* stride_ratio > 1.3 → walking.
* else → standing.
"""

from __future__ import annotations

import math
from typing import Literal

from pipeline.postprocessing import Detection, Keypoint

Activity = Literal["sitting", "standing", "walking", "running", "unknown"]

# Tunable thresholds — see SPEC.md §5.6.
KNEE_ANGLE_SIT = 120.0
HIP_HEIGHT_RATIO_SIT = 0.40
STRIDE_RATIO_WALK = 1.3
STRIDE_RATIO_RUN = 2.0
TORSO_LEAN_RUN = 15.0
KEYPOINT_VIS_MIN = 0.5
ASPECT_RATIO_SIT_FALLBACK = 1.5

# COCO 17 indices used by the classifier.
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16


def _angle_deg(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Angle at point ``b`` formed by segments ``ba`` and ``bc``, in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 180.0
    cos_val = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_val))


def _all_visible(kps: list[Keypoint], *indices: int) -> bool:
    return all(kps[i].vis >= KEYPOINT_VIS_MIN for i in indices)


def classify_activity(det: Detection) -> Activity:
    """Classify a Detection into one of the supported activity labels."""
    kps = det.keypoints
    x1, y1, x2, y2 = det.bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    if bbox_w <= 0 or bbox_h <= 0:
        return "unknown"

    lower_body = (KP_L_HIP, KP_R_HIP, KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE)
    if not _all_visible(kps, *lower_body):
        # Aspect ratio fallback: tall bbox = standing, short = sitting.
        return "sitting" if bbox_h / bbox_w < ASPECT_RATIO_SIT_FALLBACK else "standing"

    angle_l = _angle_deg(
        (kps[KP_L_HIP].x, kps[KP_L_HIP].y),
        (kps[KP_L_KNEE].x, kps[KP_L_KNEE].y),
        (kps[KP_L_ANKLE].x, kps[KP_L_ANKLE].y),
    )
    angle_r = _angle_deg(
        (kps[KP_R_HIP].x, kps[KP_R_HIP].y),
        (kps[KP_R_KNEE].x, kps[KP_R_KNEE].y),
        (kps[KP_R_ANKLE].x, kps[KP_R_ANKLE].y),
    )
    knee_angle = (angle_l + angle_r) / 2.0

    avg_hip_y = (kps[KP_L_HIP].y + kps[KP_R_HIP].y) / 2.0
    hip_height_ratio = (y2 - avg_hip_y) / bbox_h

    if knee_angle < KNEE_ANGLE_SIT and hip_height_ratio < HIP_HEIGHT_RATIO_SIT:
        return "sitting"

    hip_dx = abs(kps[KP_L_HIP].x - kps[KP_R_HIP].x)
    ankle_dx = abs(kps[KP_L_ANKLE].x - kps[KP_R_ANKLE].x)
    stride_ratio = ankle_dx / hip_dx if hip_dx > 1e-6 else 0.0

    # Torso lean: angle between (shoulders→hips) midline and the vertical axis.
    # Image y axis points down, so vertical = (0, +1).
    mid_shoulder = (
        (kps[KP_L_SHOULDER].x + kps[KP_R_SHOULDER].x) / 2,
        (kps[KP_L_SHOULDER].y + kps[KP_R_SHOULDER].y) / 2,
    )
    mid_hip = (
        (kps[KP_L_HIP].x + kps[KP_R_HIP].x) / 2,
        (kps[KP_L_HIP].y + kps[KP_R_HIP].y) / 2,
    )
    torso_dx = mid_hip[0] - mid_shoulder[0]
    torso_dy = mid_hip[1] - mid_shoulder[1]
    torso_len = math.hypot(torso_dx, torso_dy)
    if torso_len > 1e-6:
        cos_lean = max(-1.0, min(1.0, abs(torso_dy) / torso_len))
        torso_lean = math.degrees(math.acos(cos_lean))
    else:
        torso_lean = 0.0

    if stride_ratio > STRIDE_RATIO_RUN and torso_lean > TORSO_LEAN_RUN:
        return "running"
    if stride_ratio > STRIDE_RATIO_WALK:
        return "walking"
    return "standing"
