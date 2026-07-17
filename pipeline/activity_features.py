"""Versioned per-person pose features shared by MLP training and runtime."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from pipeline.postprocessing import Detection

ACTIVITY_CLASSES = ("sitting", "standing", "walking", "running")
FEATURE_SCHEMA_VERSION = "activity-mlp-features-v1"

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

JOINT_ANGLES = (
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
)

SEGMENTS = (
    "left_upper_arm",
    "right_upper_arm",
    "left_forearm",
    "right_forearm",
    "left_torso",
    "right_torso",
    "left_thigh",
    "right_thigh",
    "left_shin",
    "right_shin",
    "shoulder_width",
    "hip_width",
)

POSTURE_VECTORS = (
    "shoulders_to_hips",
    "hips_to_knees",
    "knees_to_ankles",
)

_JOINT_ANGLE_INDICES = (
    (5, 7, 9),
    (6, 8, 10),
    (7, 5, 11),
    (8, 6, 12),
    (5, 11, 13),
    (6, 12, 14),
    (11, 13, 15),
    (12, 14, 16),
)
_SEGMENT_INDICES = (
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    (5, 6),
    (11, 12),
)
_POSTURE_GROUPS = (
    ((5, 6), (11, 12)),
    ((11, 12), (13, 14)),
    ((13, 14), (15, 16)),
)

_raw_names = tuple(
    feature_name
    for keypoint_name in COCO_KEYPOINT_NAMES
    for feature_name in (
        f"{keypoint_name}.x_bbox",
        f"{keypoint_name}.y_bbox",
        f"{keypoint_name}.visibility",
    )
)
_angle_names = tuple(
    feature_name
    for joint_name in JOINT_ANGLES
    for feature_name in (
        f"{joint_name}.angle_0_1",
        f"{joint_name}.visibility_min",
    )
)
_segment_names = tuple(
    feature_name
    for segment_name in SEGMENTS
    for feature_name in (
        f"{segment_name}.length_bbox",
        f"{segment_name}.visibility_min",
    )
)
_posture_names = tuple(
    feature_name
    for vector_name in POSTURE_VECTORS
    for feature_name in (
        f"{vector_name}.dx_bbox",
        f"{vector_name}.dy_bbox",
        f"{vector_name}.length_bbox",
        f"{vector_name}.visibility_min",
    )
)
_global_names = (
    "bbox.aspect_width_height",
    "shoulders.spread_bbox",
    "hips.spread_bbox",
    "knees.spread_bbox",
    "ankles.spread_bbox",
    "shoulders.center_y_bbox",
    "hips.center_y_bbox",
    "knees.center_y_bbox",
    "ankles.center_y_bbox",
    "visibility.mean_all",
    "visibility.mean_upper",
    "visibility.mean_lower",
)

FEATURE_NAMES = _raw_names + _angle_names + _segment_names + _posture_names + _global_names
FEATURE_DIM = len(FEATURE_NAMES)


def feature_schema_manifest() -> dict:
    """Return the release metadata that binds preprocessing to model weights."""
    checksum_payload = {
        "class_order": list(ACTIVITY_CLASSES),
        "features": list(FEATURE_NAMES),
        "schema_version": FEATURE_SCHEMA_VERSION,
    }
    schema_sha256 = hashlib.sha256(
        json.dumps(checksum_payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {
        **checksum_payload,
        "input_dimension": FEATURE_DIM,
        "schema_sha256": schema_sha256,
    }


def _angle_0_1(
    first: tuple[float, float],
    vertex: tuple[float, float],
    third: tuple[float, float],
) -> float:
    left = (first[0] - vertex[0], first[1] - vertex[1])
    right = (third[0] - vertex[0], third[1] - vertex[1])
    left_length = math.hypot(*left)
    right_length = math.hypot(*right)
    if left_length <= 1e-12 or right_length <= 1e-12:
        return 0.0
    cosine = max(
        -1.0,
        min(1.0, (left[0] * right[0] + left[1] * right[1]) / (left_length * right_length)),
    )
    return math.acos(cosine) / math.pi


def extract_activity_features(detection: Detection) -> np.ndarray:
    """Return the feature-schema-v1 vector for one person detection."""
    if len(detection.keypoints) != len(COCO_KEYPOINT_NAMES):
        raise ValueError(
            f"expected exactly {len(COCO_KEYPOINT_NAMES)} COCO keypoints; "
            f"found {len(detection.keypoints)}"
        )
    x1, y1, x2, y2 = detection.bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(
            f"bbox width and height must be positive; found width={width}, height={height}"
        )
    values = [
        value
        for keypoint in detection.keypoints
        for value in (
            (keypoint.x - x1) / width,
            (keypoint.y - y1) / height,
            keypoint.vis,
        )
    ]
    normalized_points = [
        ((keypoint.x - x1) / width, (keypoint.y - y1) / height) for keypoint in detection.keypoints
    ]
    for first, vertex, third in _JOINT_ANGLE_INDICES:
        values.extend(
            (
                _angle_0_1(
                    normalized_points[first],
                    normalized_points[vertex],
                    normalized_points[third],
                ),
                min(
                    detection.keypoints[first].vis,
                    detection.keypoints[vertex].vis,
                    detection.keypoints[third].vis,
                ),
            )
        )
    for first, second in _SEGMENT_INDICES:
        first_point = normalized_points[first]
        second_point = normalized_points[second]
        values.extend(
            (
                math.hypot(
                    second_point[0] - first_point[0],
                    second_point[1] - first_point[1],
                ),
                min(detection.keypoints[first].vis, detection.keypoints[second].vis),
            )
        )
    for start_indices, end_indices in _POSTURE_GROUPS:
        start = tuple(
            sum(normalized_points[index][axis] for index in start_indices) / 2 for axis in (0, 1)
        )
        end = tuple(
            sum(normalized_points[index][axis] for index in end_indices) / 2 for axis in (0, 1)
        )
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        values.extend(
            (
                dx,
                dy,
                math.hypot(dx, dy),
                min(detection.keypoints[index].vis for index in (*start_indices, *end_indices)),
            )
        )
    body_pairs = ((5, 6), (11, 12), (13, 14), (15, 16))
    visibilities = [keypoint.vis for keypoint in detection.keypoints]
    values.extend(
        (
            width / height,
            *(
                abs(normalized_points[left][0] - normalized_points[right][0])
                for left, right in body_pairs
            ),
            *(
                (normalized_points[left][1] + normalized_points[right][1]) / 2
                for left, right in body_pairs
            ),
            sum(visibilities) / len(visibilities),
            sum(visibilities[:11]) / 11,
            sum(visibilities[11:]) / 6,
        )
    )
    return np.asarray(values, dtype=np.float32)
