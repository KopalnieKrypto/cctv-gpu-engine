"""Rule-based activity classification from COCO 17 keypoints.

Classifies a single Detection into one of:
``sitting``, ``standing``, ``walking``, ``running`` (or ``unknown`` only when
the bbox is degenerate). All thresholds live as module-level constants so they
can be tuned without touching the decision logic.

Decision tree (per SPEC.md §5.6):

* If hips, knees, ankles are not all visible (vis ≥ 0.5):

  * If shoulders + hips visible: hip_bbox_ratio > 0.65 → sitting, else standing.
  * Else: bbox aspect-ratio fallback (h/w < 1.5 → sitting, else standing).

* knee_angle = average angle(hip, knee, ankle) for both legs.
* hip_height_ratio = (bbox_bottom - avg_hip_y) / bbox_height.
* torso_leg_ratio = torso_length / leg_length.
* 2-of-3 voting: knee_angle < 130° / hip_height_ratio < 0.55 /
  torso_leg_ratio > 1.1 → sitting.
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
KNEE_ANGLE_SIT = 130.0
HIP_HEIGHT_RATIO_SIT = 0.55
TORSO_LEG_RATIO_SIT = 1.1
STRIDE_RATIO_WALK = 1.3
STRIDE_RATIO_RUN = 2.0
TORSO_LEAN_RUN = 15.0
KEYPOINT_VIS_MIN = 0.5
ASPECT_RATIO_SIT_FALLBACK = 1.5
HIP_BBOX_RATIO_SIT_FALLBACK = 0.65

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
        # Improved fallback: use hip position in bbox when upper body visible.
        if _all_visible(kps, KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP):
            avg_hip_y = (kps[KP_L_HIP].y + kps[KP_R_HIP].y) / 2.0
            hip_bbox_ratio = (avg_hip_y - y1) / bbox_h
            if hip_bbox_ratio > HIP_BBOX_RATIO_SIT_FALLBACK:
                return "sitting"
            return "standing"
        # Last-resort: bbox aspect ratio.
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

    # Torso-to-leg length ratio: seated persons have compressed legs.
    mid_shoulder = (
        (kps[KP_L_SHOULDER].x + kps[KP_R_SHOULDER].x) / 2,
        (kps[KP_L_SHOULDER].y + kps[KP_R_SHOULDER].y) / 2,
    )
    mid_hip = (
        (kps[KP_L_HIP].x + kps[KP_R_HIP].x) / 2,
        (kps[KP_L_HIP].y + kps[KP_R_HIP].y) / 2,
    )
    mid_ankle = (
        (kps[KP_L_ANKLE].x + kps[KP_R_ANKLE].x) / 2,
        (kps[KP_L_ANKLE].y + kps[KP_R_ANKLE].y) / 2,
    )
    torso_length = math.hypot(mid_hip[0] - mid_shoulder[0], mid_hip[1] - mid_shoulder[1])
    leg_length = math.hypot(mid_ankle[0] - mid_hip[0], mid_ankle[1] - mid_hip[1])
    torso_leg_ratio = torso_length / leg_length if leg_length > 1e-6 else 0.0

    # Sitting: 2-of-3 voting — knee angle, hip position, torso/leg proportion.
    sitting_score = 0
    if knee_angle < KNEE_ANGLE_SIT:
        sitting_score += 1
    if hip_height_ratio < HIP_HEIGHT_RATIO_SIT:
        sitting_score += 1
    if torso_leg_ratio > TORSO_LEG_RATIO_SIT:
        sitting_score += 1
    if sitting_score >= 2:
        return "sitting"

    hip_dx = abs(kps[KP_L_HIP].x - kps[KP_R_HIP].x)
    ankle_dx = abs(kps[KP_L_ANKLE].x - kps[KP_R_ANKLE].x)
    stride_ratio = ankle_dx / hip_dx if hip_dx > 1e-6 else 0.0

    # Torso lean: angle between (shoulders→hips) midline and the vertical axis.
    torso_dy = mid_hip[1] - mid_shoulder[1]
    if torso_length > 1e-6:
        cos_lean = max(-1.0, min(1.0, abs(torso_dy) / torso_length))
        torso_lean = math.degrees(math.acos(cos_lean))
    else:
        torso_lean = 0.0

    if stride_ratio > STRIDE_RATIO_RUN and torso_lean > TORSO_LEAN_RUN:
        return "running"
    if stride_ratio > STRIDE_RATIO_WALK:
        return "walking"
    return "standing"


SMOOTHING_WINDOW = 5
DISPLACEMENT_WALK_THRESHOLD = 0.05


def bbox_center(det: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = det.bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


class ActivitySmoother:
    """Sliding-window majority vote + displacement-based walking override.

    Matches detections across frames by nearest bbox center (no tracking ID
    needed).  Two corrections are applied before the majority vote:

    1. **Displacement walking** — if the bbox center moved more than
       ``DISPLACEMENT_WALK_THRESHOLD`` (normalised by bbox height) since the
       previous frame, a ``standing`` classification is upgraded to
       ``walking``.  This catches frontal walking that stride-ratio misses.
    2. **Majority vote** — the (possibly corrected) raw label is voted
       against the last ``window`` frames to smooth flickering.
    """

    def __init__(self, window: int = SMOOTHING_WINDOW) -> None:
        self.window = window
        # Each entry: list of (center_x, center_y, activity) for one frame.
        self._history: list[list[tuple[float, float, str]]] = []

    def smooth(self, detections: list[Detection]) -> list[Detection]:
        prev_frame = self._history[-1] if self._history else []
        current: list[tuple[float, float, str]] = []
        for det in detections:
            cx, cy = bbox_center(det)
            bbox_h = det.bbox[3] - det.bbox[1]
            raw_activity = det.activity

            # Displacement corrections when we have a previous frame.
            if prev_frame and bbox_h > 1e-6:
                best_dist = float("inf")
                for px, py, _pact in prev_frame:
                    d = math.hypot(cx - px, cy - py)
                    if d < best_dist:
                        best_dist = d
                norm_disp = best_dist / bbox_h
                if raw_activity == "standing" and norm_disp > DISPLACEMENT_WALK_THRESHOLD:
                    # Person is moving but stride_ratio missed it (frontal walk).
                    raw_activity = "walking"
                elif raw_activity == "walking" and norm_disp <= DISPLACEMENT_WALK_THRESHOLD:
                    # Stride_ratio triggered walking but person is stationary.
                    raw_activity = "standing"

            votes: list[str] = [raw_activity]
            for past_frame in self._history:
                best_dist = float("inf")
                best_act = ""
                for px, py, pact in past_frame:
                    d = math.hypot(cx - px, cy - py)
                    if d < best_dist:
                        best_dist = d
                        best_act = pact
                if best_act:
                    votes.append(best_act)
            # Majority vote — ties broken by keeping the current frame's label.
            counts: dict[str, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            majority = max(counts, key=lambda k: (counts[k], k == raw_activity))
            det.activity = majority
            # Store raw (unsmoothed) classification to avoid self-reinforcement.
            current.append((cx, cy, raw_activity))

        self._history.append(current)
        if len(self._history) > self.window:
            self._history.pop(0)
        return detections
