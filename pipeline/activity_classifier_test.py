"""Tests for keypoint-based activity classification.

Synthetic keypoint geometries verify each branch of the heuristic decision
tree (sitting / standing / walking / running / fallback).
"""

from __future__ import annotations

from pipeline.activity_classifier import classify_activity
from pipeline.postprocessing import Detection, Keypoint

# COCO 17 indices we set in fixtures (others get visibility=0).
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16


def _person(
    bbox: tuple[float, float, float, float],
    *,
    shoulders: tuple[tuple[float, float], tuple[float, float]] | None = None,
    hips: tuple[tuple[float, float], tuple[float, float]] | None = None,
    knees: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ankles: tuple[tuple[float, float], tuple[float, float]] | None = None,
    visibility: float = 0.9,
) -> Detection:
    """Build a Detection with the named keypoints visible and the rest invisible."""
    kps = [Keypoint(0.0, 0.0, 0.0) for _ in range(17)]
    if shoulders is not None:
        (lx, ly), (rx, ry) = shoulders
        kps[KP_L_SHOULDER] = Keypoint(lx, ly, visibility)
        kps[KP_R_SHOULDER] = Keypoint(rx, ry, visibility)
    if hips is not None:
        (lx, ly), (rx, ry) = hips
        kps[KP_L_HIP] = Keypoint(lx, ly, visibility)
        kps[KP_R_HIP] = Keypoint(rx, ry, visibility)
    if knees is not None:
        (lx, ly), (rx, ry) = knees
        kps[KP_L_KNEE] = Keypoint(lx, ly, visibility)
        kps[KP_R_KNEE] = Keypoint(rx, ry, visibility)
    if ankles is not None:
        (lx, ly), (rx, ry) = ankles
        kps[KP_L_ANKLE] = Keypoint(lx, ly, visibility)
        kps[KP_R_ANKLE] = Keypoint(rx, ry, visibility)
    x1, y1, x2, y2 = bbox
    return Detection(bbox=[x1, y1, x2, y2], confidence=0.9, keypoints=kps)


class TestClassifyActivity:
    def test_sitting_when_knees_bent_and_hip_low_in_bbox(self):
        # Image y axis points DOWN. Bbox from y=100 (top) to y=300 (bottom).
        # Hips at y=270 → close to bbox bottom → hip_height_ratio = 30/200 = 0.15 < 0.40
        # Knees in front of and slightly above hips, ankles below knees but
        # forward → bent legs producing knee angle ~80° (< 120° SIT threshold).
        person = _person(
            bbox=(100, 100, 200, 300),
            shoulders=((130, 150), (170, 150)),
            hips=((130, 270), (170, 270)),
            knees=((150, 240), (180, 240)),  # forward & slightly up
            ankles=((150, 280), (180, 280)),  # below knees, same x → bent
        )

        assert classify_activity(person) == "sitting"

    def test_standing_when_legs_straight_and_feet_under_hips(self):
        # Tall bbox 200×600. Hips at y=300 (mid), knees y=450, ankles y=600.
        # All straight vertical line → knee angle ≈ 180°. Hip ratio = 300/600 = 0.5.
        # Ankles directly under hips → stride_ratio ≈ 1.0 (= hip_dx).
        person = _person(
            bbox=(100, 0, 300, 600),
            shoulders=((180, 100), (220, 100)),
            hips=((180, 300), (220, 300)),
            knees=((180, 450), (220, 450)),
            ankles=((180, 600), (220, 600)),
        )

        assert classify_activity(person) == "standing"

    def test_walking_when_stride_wider_than_hip_width(self):
        # Same vertical posture as standing but ankles spread to ~1.6× hip width.
        # hip_dx = 40, ankle_dx = 70 → stride_ratio = 1.75 (> 1.3, < 2.0)
        person = _person(
            bbox=(100, 0, 300, 600),
            shoulders=((180, 100), (220, 100)),
            hips=((180, 300), (220, 300)),
            knees=((175, 450), (225, 450)),
            ankles=((165, 600), (235, 600)),
        )

        assert classify_activity(person) == "walking"

    def test_running_when_wide_stride_and_torso_leans_forward(self):
        # Stride well above 2.0× hip width AND torso leans ~20° forward.
        # hip_dx = 40, ankle_dx = 100 → stride_ratio = 2.5
        # Shoulders shifted forward of hips by 80px over ~200px torso → ~21°.
        person = _person(
            bbox=(50, 0, 350, 600),
            shoulders=((260, 100), (300, 100)),  # forward of hips
            hips=((180, 300), (220, 300)),
            knees=((170, 450), (230, 450)),
            ankles=((150, 600), (250, 600)),
        )

        assert classify_activity(person) == "running"


class TestClassifyActivityFallback:
    def test_falls_back_to_standing_for_tall_bbox_when_lower_body_invisible(self):
        # No keypoints visible at all → must use bbox aspect ratio.
        # Tall bbox (h/w = 600/200 = 3.0) → standing
        person = _person(bbox=(100, 0, 300, 600))

        assert classify_activity(person) == "standing"

    def test_falls_back_to_sitting_for_short_bbox_when_lower_body_invisible(self):
        # Short bbox (h/w = 200/300 ≈ 0.67) → sitting
        person = _person(bbox=(100, 100, 400, 300))

        assert classify_activity(person) == "sitting"

    def test_falls_back_when_only_one_ankle_visible(self):
        # Hips + knees visible, but only ONE ankle visible → still incomplete.
        # Tall bbox so fallback should yield standing.
        person = _person(
            bbox=(100, 0, 300, 600),
            shoulders=((180, 100), (220, 100)),
            hips=((180, 300), (220, 300)),
            knees=((180, 450), (220, 450)),
        )
        # add only the left ankle
        person.keypoints[KP_L_ANKLE] = Keypoint(180, 600, 0.9)

        assert classify_activity(person) == "standing"

    def test_returns_unknown_for_degenerate_zero_area_bbox(self):
        person = _person(bbox=(100, 100, 100, 100))

        assert classify_activity(person) == "unknown"
