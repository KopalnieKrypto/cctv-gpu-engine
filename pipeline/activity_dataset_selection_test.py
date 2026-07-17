"""Behavior tests for deterministic issue #33 candidate selection."""

from __future__ import annotations

from pipeline.activity_dataset_selection import select_evenly_spaced, select_from_quota_plan


def _candidate(index: int, *, confidence: float = 0.8, frame_hash: str | None = None) -> dict:
    return {
        "sample_id": f"sample-{index}",
        "source_video_sha256": "video-a",
        "source_timestamp_s": float(index),
        "frame_sha256": frame_hash or f"hash-{index}",
        "pose_confidence": confidence,
    }


def test_selection_deduplicates_frames_and_spreads_across_time() -> None:
    """A quota is filled by unique source frames across the available interval."""
    candidates = [_candidate(index) for index in range(4)]
    candidates.append(_candidate(10, confidence=0.95, frame_hash="hash-1"))

    selected = select_evenly_spaced(candidates, quota=3, used_frame_hashes=set())

    assert [candidate["sample_id"] for candidate in selected] == [
        "sample-0",
        "sample-2",
        "sample-3",
    ]
    assert len({candidate["frame_sha256"] for candidate in selected}) == 3


def test_quota_plan_assigns_splits_without_frame_leakage() -> None:
    """Train and validation quotas share candidates but never a source frame."""
    candidates = []
    for index in range(5):
        candidate = _candidate(index)
        candidate.update({"camera_geometry_id": "geometry-1", "activity": "walking"})
        candidates.append(candidate)
    plan = [
        {
            "camera_geometry_id": "geometry-1",
            "activity": "walking",
            "split": "train",
            "count": 2,
        },
        {
            "camera_geometry_id": "geometry-1",
            "activity": "walking",
            "split": "validation",
            "count": 1,
        },
    ]

    selected = select_from_quota_plan(candidates, plan)

    assert [candidate["split"] for candidate in selected] == [
        "train",
        "train",
        "validation",
    ]
    assert len({candidate["frame_sha256"] for candidate in selected}) == 3
    assert all(candidate["review_status"] == "pending" for candidate in selected)
