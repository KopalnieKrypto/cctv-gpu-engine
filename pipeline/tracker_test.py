"""Tests for within-video person tracking (issue #32).

The tracker sits between :mod:`pipeline.pose_detector` and the activity
classifier: it receives the per-frame Detection list and stamps each one with
a stable ``track_id`` so downstream aggregation can count *people* rather than
*person-frames*.

Association is by appearance (OSNet Re-ID cosine similarity), not IoU — at
1 fps a person moves too far between frames for box overlap to mean anything
(Andrew Ng advisory 2026-05-27).

These tests inject a fake embedder. The real embedder wraps an ONNX session,
which is a genuine system boundary; the tracker's own association logic is
never mocked — it is exactly what is under test here.
"""

from __future__ import annotations

import numpy as np

from pipeline.postprocessing import Detection, Keypoint
from pipeline.tracker import PersonTracker


def _unit(*components: float) -> np.ndarray:
    """An L2-normalized appearance vector — what a real embedder returns."""
    vec = np.array(components, dtype=np.float32)
    return vec / np.linalg.norm(vec)


# Two mutually orthogonal appearance vectors: cosine similarity 1.0 against
# themselves, 0.0 against each other. "Same person" vs "different person" with
# no model in the loop.
PERSON_A = _unit(1.0, 0.0)
PERSON_B = _unit(0.0, 1.0)


class FakeEmbedder:
    """Stand-in for the OSNet ONNX session (a system boundary).

    Yields a caller-scripted list of appearance vectors per ``embed`` call —
    one per detection, in detection order — so a test can simply *state* who
    is in each frame.
    """

    def __init__(self, vectors_per_frame: list[list[np.ndarray]]) -> None:
        self._queue = list(vectors_per_frame)

    def embed(self, frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]:
        assert self._queue, "embedder called more times than the test scripted"
        vectors = self._queue.pop(0)
        assert len(vectors) == len(detections), (
            f"test scripted {len(vectors)} vectors but frame has {len(detections)} detections"
        )
        return vectors


def _frame(h: int = 8, w: int = 8) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _det(bbox=(0.0, 0.0, 10.0, 20.0), confidence: float = 0.9) -> Detection:
    return Detection(
        bbox=list(bbox),
        confidence=confidence,
        keypoints=[Keypoint(0.0, 0.0, 0.0)] * 17,
    )


class TestTrackIdentity:
    def test_same_person_in_consecutive_frames_keeps_one_track_id(self):
        tracker = PersonTracker(embedder=FakeEmbedder([[PERSON_A], [PERSON_A]]))

        first = tracker.update(_frame(), [_det()], timestamp_s=0.0)
        second = tracker.update(_frame(), [_det()], timestamp_s=1.0)

        assert first[0].track_id is not None
        assert first[0].track_id == second[0].track_id

    def test_two_people_in_one_frame_never_share_a_track_id(self):
        # Both detections embed identically — the worst case for body-only
        # Re-ID, and the normal case at a bending station where everyone wears
        # the same hi-vis. They are still two people in one frame, so one
        # track may claim at most one of them.
        tracker = PersonTracker(embedder=FakeEmbedder([[PERSON_A], [PERSON_A, PERSON_A]]))

        tracker.update(_frame(), [_det()], timestamp_s=0.0)
        second = tracker.update(_frame(), [_det(), _det()], timestamp_s=1.0)

        assert second[0].track_id != second[1].track_id


class TestTrackAge:
    def test_returner_after_the_default_two_minute_age_gets_a_new_track_id(self):
        # Body-only Re-ID decays with the gap (~80% at 2 min, <50% past 5 min),
        # so a stale track is retired rather than trusted. Splitting one person
        # into two tracks costs an absence gap in the report; merging two people
        # into one silently corrupts the person-minutes. We take the split.
        tracker = PersonTracker(embedder=FakeEmbedder([[PERSON_A], [PERSON_A]]))

        first = tracker.update(_frame(), [_det()], timestamp_s=0.0)
        second = tracker.update(_frame(), [_det()], timestamp_s=181.0)

        assert second[0].track_id != first[0].track_id

    def test_returner_within_the_age_window_resumes_the_same_track_id(self):
        # The other side of the boundary: a worker who steps out of frame for a
        # minute is still the same worker, and must not fragment into a second
        # person on the report.
        tracker = PersonTracker(embedder=FakeEmbedder([[PERSON_A], [PERSON_A]]))

        first = tracker.update(_frame(), [_det()], timestamp_s=0.0)
        second = tracker.update(_frame(), [_det()], timestamp_s=60.0)

        assert second[0].track_id == first[0].track_id

    def test_max_track_age_is_configurable(self):
        tracker = PersonTracker(
            embedder=FakeEmbedder([[PERSON_A], [PERSON_A]]),
            max_track_age_s=300.0,
        )

        first = tracker.update(_frame(), [_det()], timestamp_s=0.0)
        second = tracker.update(_frame(), [_det()], timestamp_s=181.0)

        assert second[0].track_id == first[0].track_id


class TestLowConfidenceDetections:
    """ByteTrack's two-stage association, expressed through track_id.

    A faint box is worth keeping when it continues someone we already track,
    and worth nothing when it doesn't — that asymmetry is what stops sporadic
    junk detections from becoming people.
    """

    def test_low_confidence_detection_alone_does_not_spawn_a_track(self):
        tracker = PersonTracker(embedder=FakeEmbedder([[PERSON_A]]))

        detections = tracker.update(_frame(), [_det(confidence=0.3)], timestamp_s=0.0)

        assert detections[0].track_id is None

    def test_low_confidence_detection_continues_an_established_track(self):
        # The person is briefly half-occluded and YOLO's confidence sags. The
        # track should carry through the dip rather than break in two.
        tracker = PersonTracker(embedder=FakeEmbedder([[PERSON_A], [PERSON_A]]))

        first = tracker.update(_frame(), [_det(confidence=0.9)], timestamp_s=0.0)
        second = tracker.update(_frame(), [_det(confidence=0.3)], timestamp_s=1.0)

        assert second[0].track_id == first[0].track_id
