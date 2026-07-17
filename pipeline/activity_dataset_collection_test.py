"""Behavior tests for the issue #33 GPU pre-label collection tool."""

from __future__ import annotations

import numpy as np

from pipeline.activity_dataset_collection import (
    activity_at_timestamp,
    candidate_from_detection,
    resize_full_frame,
    select_primary_detection,
)
from pipeline.postprocessing import Detection, Keypoint


def test_detection_becomes_one_full_frame_candidate() -> None:
    """The collector serializes the existing YOLO pose contract without cropping."""
    detection = Detection(
        bbox=[-5.0, 10.0, 210.0, 90.0],
        confidence=0.91,
        keypoints=[Keypoint(x=25.0, y=30.0, vis=0.8) for _ in range(17)],
        activity="standing",
    )

    candidate = candidate_from_detection(
        detection,
        activity="walking",
        camera_geometry_id="factory-a7b76f41",
        frame_height=100,
        frame_path="factory-a7b76f41/frame.jpg",
        frame_sha256="a" * 64,
        frame_width=200,
        sample_id="factory-a7b76f41-source-001000",
        source_id="camera-buffer-20260717-a7b76f41",
        source_timestamp_s=1.0,
        source_video_sha256="b" * 64,
    )

    assert candidate == {
        "activity": "walking",
        "bbox": [0.0, 10.0, 200.0, 80.0],
        "camera_geometry_id": "factory-a7b76f41",
        "frame_height": 100,
        "frame_path": "factory-a7b76f41/frame.jpg",
        "frame_sha256": "a" * 64,
        "frame_width": 200,
        "keypoints": [{"x": 25.0, "y": 30.0, "vis": 0.8} for _ in range(17)],
        "pose_confidence": 0.91,
        "review_status": "pending",
        "sample_id": "factory-a7b76f41-source-001000",
        "source_id": "camera-buffer-20260717-a7b76f41",
        "source_timestamp_s": 1.0,
        "source_video_sha256": "b" * 64,
        "split": "unassigned",
        "synthetic": False,
    }


def test_out_of_frame_keypoint_is_clamped_and_marked_invisible() -> None:
    """Letterbox-edge predictions remain valid without pretending to be visible."""
    keypoints = [Keypoint(x=25.0, y=30.0, vis=0.8) for _ in range(17)]
    keypoints[0] = Keypoint(x=-4.0, y=105.0, vis=0.9)
    detection = Detection(
        bbox=[10.0, 10.0, 50.0, 90.0],
        confidence=0.91,
        keypoints=keypoints,
    )

    candidate = candidate_from_detection(
        detection,
        activity="standing",
        camera_geometry_id="geometry-1",
        frame_height=100,
        frame_path="geometry-1/frame.jpg",
        frame_sha256="a" * 64,
        frame_width=200,
        sample_id="sample-1",
        source_id="source-1",
        source_timestamp_s=1.0,
        source_video_sha256="b" * 64,
    )

    assert candidate["keypoints"][0] == {"x": 0.0, "y": 99.0, "vis": 0.0}


def test_primary_detection_filters_tiny_people_then_prefers_confidence() -> None:
    """Exactly one reviewable person is selected from each source frame."""

    def detection(confidence: float, height: float) -> Detection:
        return Detection(
            bbox=[10.0, 10.0, 50.0, 10.0 + height],
            confidence=confidence,
            keypoints=[Keypoint(x=25.0, y=30.0, vis=0.8) for _ in range(17)],
        )

    tiny = detection(confidence=0.99, height=20.0)
    lower_confidence = detection(confidence=0.72, height=80.0)
    primary = detection(confidence=0.85, height=70.0)

    assert (
        select_primary_detection(
            [tiny, lower_confidence, primary],
            min_bbox_height=40.0,
            min_confidence=0.5,
        )
        is primary
    )


def test_activity_intervals_are_start_inclusive_and_end_exclusive() -> None:
    """Boundary frames receive one deterministic reviewed activity."""
    intervals = [
        {"start_s": 10.0, "end_s": 20.0, "activity": "walking"},
        {"start_s": 20.0, "end_s": 25.0, "activity": "standing"},
    ]

    assert activity_at_timestamp(intervals, 9.999) is None
    assert activity_at_timestamp(intervals, 10.0) == "walking"
    assert activity_at_timestamp(intervals, 19.999) == "walking"
    assert activity_at_timestamp(intervals, 20.0) == "standing"
    assert activity_at_timestamp(intervals, 25.0) is None


def test_full_frame_is_downscaled_without_cropping_or_upscaling() -> None:
    """Large 4K frames keep their geometry at the model's useful resolution."""
    large = np.zeros((100, 200, 3), dtype=np.uint8)
    small = np.zeros((50, 80, 3), dtype=np.uint8)

    assert resize_full_frame(large, max_width=100).shape == (50, 100, 3)
    assert resize_full_frame(small, max_width=100) is small
