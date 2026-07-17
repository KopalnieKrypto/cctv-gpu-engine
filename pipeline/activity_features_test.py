"""Behavior tests for the frozen activity-MLP feature contract.

Feature assumptions approved before RED for issue #34:

* Runtime input is one ``Detection`` with an xyxy bbox and exactly 17 COCO
  keypoints. Dataset xywh conversion stays in the training adapter.
* Output is one deterministic, finite, float32 vector. Translation and uniform
  scale of the person box must not change it.
* Visibility is continuous and never thresholded. Low/missing confidence
  remains represented explicitly; it does not select a fallback schema.
* Degenerate boxes and wrong keypoint counts are errors. Clipping poses or
  inventing coordinates for invisible joints is intentionally out of scope.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.activity_features import (
    ACTIVITY_CLASSES,
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract_activity_features,
    feature_schema_manifest,
)
from pipeline.postprocessing import Detection, Keypoint


def test_feature_schema_v1_has_a_frozen_order_and_class_contract() -> None:
    assert FEATURE_SCHEMA_VERSION == "activity-mlp-features-v1"
    assert ACTIVITY_CLASSES == ("sitting", "standing", "walking", "running")
    assert FEATURE_DIM == 115
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert FEATURE_NAMES[:6] == (
        "nose.x_bbox",
        "nose.y_bbox",
        "nose.visibility",
        "left_eye.x_bbox",
        "left_eye.y_bbox",
        "left_eye.visibility",
    )
    assert FEATURE_NAMES[48:55] == (
        "right_ankle.x_bbox",
        "right_ankle.y_bbox",
        "right_ankle.visibility",
        "left_elbow.angle_0_1",
        "left_elbow.visibility_min",
        "right_elbow.angle_0_1",
        "right_elbow.visibility_min",
    )
    assert FEATURE_NAMES[-12:] == (
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


def test_feature_schema_manifest_checksum_binds_order_and_class_labels() -> None:
    manifest = feature_schema_manifest()
    checksum_payload = {
        "class_order": list(ACTIVITY_CLASSES),
        "features": list(FEATURE_NAMES),
        "schema_version": FEATURE_SCHEMA_VERSION,
    }
    expected_checksum = hashlib.sha256(
        json.dumps(checksum_payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()

    assert manifest == {
        **checksum_payload,
        "input_dimension": FEATURE_DIM,
        "schema_sha256": expected_checksum,
    }


def test_committed_feature_schema_matches_runtime_preprocessing_exactly() -> None:
    schema_path = Path(__file__).parents[1] / "training/activity-mlp/feature-schema.json"

    assert json.loads(schema_path.read_text()) == feature_schema_manifest()


def test_extract_features_starts_with_bbox_normalized_coco_values() -> None:
    detection = Detection(
        bbox=[10.0, 20.0, 110.0, 220.0],
        confidence=0.9,
        keypoints=[
            Keypoint(x=10.0 + index, y=20.0 + 2 * index, vis=index / 16) for index in range(17)
        ],
    )

    features = extract_activity_features(detection)

    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features[:6], [0.0, 0.0, 0.0, 0.01, 0.01, 1 / 16])
    np.testing.assert_allclose(features[48:51], [0.16, 0.16, 1.0])
    assert np.isfinite(features).all()


def test_joint_angles_are_continuous_and_carry_minimum_visibility() -> None:
    keypoints = [Keypoint(x=0.0, y=0.0, vis=1.0) for _ in range(17)]
    keypoints[5] = Keypoint(x=2.0, y=2.0, vis=0.4)  # left shoulder
    keypoints[7] = Keypoint(x=4.0, y=2.0, vis=0.3)  # left elbow
    keypoints[9] = Keypoint(x=4.0, y=6.0, vis=0.2)  # left wrist
    detection = Detection(bbox=[0.0, 0.0, 10.0, 10.0], confidence=1.0, keypoints=keypoints)

    features = extract_activity_features(detection)

    np.testing.assert_allclose(features[51:53], [0.5, 0.2], atol=1e-6)


def test_segment_lengths_are_bbox_normalized_and_visibility_weightable() -> None:
    keypoints = [Keypoint(x=0.0, y=0.0, vis=1.0) for _ in range(17)]
    keypoints[5] = Keypoint(x=2.0, y=2.0, vis=0.4)  # left shoulder
    keypoints[7] = Keypoint(x=4.0, y=2.0, vis=0.3)  # left elbow
    detection = Detection(bbox=[0.0, 0.0, 10.0, 10.0], confidence=1.0, keypoints=keypoints)

    features = extract_activity_features(detection)

    np.testing.assert_allclose(features[67:69], [0.2, 0.3], atol=1e-6)


def test_posture_vectors_connect_continuous_joint_pair_midpoints() -> None:
    keypoints = [Keypoint(x=0.0, y=0.0, vis=1.0) for _ in range(17)]
    keypoints[5] = Keypoint(x=2.0, y=2.0, vis=0.5)
    keypoints[6] = Keypoint(x=4.0, y=2.0, vis=0.4)
    keypoints[11] = Keypoint(x=2.0, y=6.0, vis=0.3)
    keypoints[12] = Keypoint(x=4.0, y=6.0, vis=0.2)
    detection = Detection(bbox=[0.0, 0.0, 10.0, 10.0], confidence=1.0, keypoints=keypoints)

    features = extract_activity_features(detection)

    np.testing.assert_allclose(features[91:95], [0.0, 0.4, 0.4, 0.2], atol=1e-6)


def test_global_posture_and_visibility_summaries_complete_the_schema() -> None:
    keypoints = [Keypoint(x=0.0, y=0.0, vis=index / 16) for index in range(17)]
    for left, right, y in ((5, 6, 4.0), (11, 12, 10.0), (13, 14, 14.0), (15, 16, 18.0)):
        keypoints[left].x, keypoints[right].x = 3.0, 5.0
        keypoints[left].y = keypoints[right].y = y
    keypoints[5].x = 2.0
    keypoints[6].x = 6.0
    detection = Detection(bbox=[0.0, 0.0, 10.0, 20.0], confidence=1.0, keypoints=keypoints)

    features = extract_activity_features(detection)

    np.testing.assert_allclose(
        features[103:],
        [0.5, 0.4, 0.2, 0.2, 0.2, 0.2, 0.5, 0.7, 0.9, 0.5, 0.3125, 0.84375],
        atol=1e-6,
    )


def test_all_features_are_invariant_to_bbox_translation_and_uniform_scale() -> None:
    keypoints = [
        Keypoint(x=20.0 + 3 * index, y=40.0 + 5 * index, vis=0.25 + index / 32)
        for index in range(17)
    ]
    original = Detection(bbox=[10.0, 20.0, 110.0, 220.0], confidence=1.0, keypoints=keypoints)
    transformed = Detection(
        bbox=[130.0, 10.0, 430.0, 610.0],
        confidence=1.0,
        keypoints=[
            Keypoint(x=3 * point.x + 100.0, y=3 * point.y - 50.0, vis=point.vis)
            for point in keypoints
        ],
    )

    np.testing.assert_allclose(
        extract_activity_features(original),
        extract_activity_features(transformed),
        atol=1e-6,
    )


def test_low_visibility_is_continuous_and_does_not_switch_feature_geometry() -> None:
    def detection_with_visibility(visibility: float) -> Detection:
        return Detection(
            bbox=[0.0, 0.0, 100.0, 200.0],
            confidence=1.0,
            keypoints=[
                Keypoint(x=10.0 + index, y=20.0 + 5 * index, vis=visibility) for index in range(17)
            ],
        )

    visible = extract_activity_features(detection_with_visibility(1.0))
    low_visibility = extract_activity_features(detection_with_visibility(0.01))
    visibility_mask = np.asarray(["visibility" in name for name in FEATURE_NAMES])

    np.testing.assert_array_equal(low_visibility[~visibility_mask], visible[~visibility_mask])
    assert np.all(low_visibility[visibility_mask] < visible[visibility_mask])


def test_feature_preprocessing_is_bitwise_deterministic() -> None:
    detection = Detection(
        bbox=[1.0, 2.0, 101.0, 202.0],
        confidence=0.75,
        keypoints=[
            Keypoint(x=3.0 + index, y=5.0 + 2 * index, vis=index / 17) for index in range(17)
        ],
    )

    first = extract_activity_features(detection)
    second = extract_activity_features(detection)

    np.testing.assert_array_equal(first, second)


def test_feature_preprocessing_rejects_non_coco_keypoint_counts() -> None:
    detection = Detection(
        bbox=[0.0, 0.0, 100.0, 200.0],
        confidence=1.0,
        keypoints=[Keypoint(x=1.0, y=2.0, vis=1.0) for _ in range(16)],
    )

    with pytest.raises(ValueError, match="expected exactly 17 COCO keypoints; found 16"):
        extract_activity_features(detection)


def test_feature_preprocessing_rejects_degenerate_person_boxes() -> None:
    detection = Detection(
        bbox=[10.0, 20.0, 10.0, 100.0],
        confidence=1.0,
        keypoints=[Keypoint(x=10.0, y=20.0, vis=1.0) for _ in range(17)],
    )

    with pytest.raises(ValueError, match="bbox width and height must be positive"):
        extract_activity_features(detection)
