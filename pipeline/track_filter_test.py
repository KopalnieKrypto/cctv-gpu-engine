"""Tests for the min-track-length filter (issue #32).

Sits between :mod:`pipeline.tracker` and :mod:`pipeline.aggregator`: a track
must prove itself over consecutive frames before any of its detections are
allowed to count. This is the mandated bench-on-wheels defence — the advisory
(round 8) ruled out both a person/not-person classifier and a YOLO-confidence
cutoff, leaving temporal persistence to do the job: "ByteTrack's temporal
persistence and minimum track length will likely filter out those sporadic
bench hits for free".

Because a track only proves itself on its 3rd consecutive frame, the filter is
a short delay line — it holds frames back until their fate is decided, then
releases them in order. Tests drive it through push/flush and assert on what
comes out.
"""

from __future__ import annotations

import numpy as np

from pipeline.postprocessing import Detection, Keypoint
from pipeline.track_filter import MinTrackLengthFilter


def _frame(h: int = 8, w: int = 8) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _det(track_id: int | None, activity: str = "walking") -> Detection:
    det = Detection(
        bbox=[0.0, 0.0, 10.0, 20.0],
        confidence=0.9,
        keypoints=[Keypoint(0.0, 0.0, 0.0)] * 17,
        track_id=track_id,
    )
    det.activity = activity
    return det


def _run(filter_: MinTrackLengthFilter, script: list[tuple[float, list[Detection]]]):
    """Push a whole video through the filter and collect everything emitted."""
    emitted = []
    for timestamp_s, detections in script:
        emitted.extend(filter_.push(timestamp_s, _frame(), detections))
    emitted.extend(filter_.flush())
    return emitted


class TestMinTrackLength:
    def test_track_seen_in_only_two_frames_has_its_detections_dropped(self):
        # The bench-on-wheels: YOLO calls it a person for a frame or two, then
        # it is gone. The frames themselves genuinely happened, so they must
        # still reach the aggregator — video duration and frame counts stay
        # honest — but the phantom person must not survive.
        filter_ = MinTrackLengthFilter()

        emitted = _run(filter_, [(0.0, [_det(track_id=7)]), (1.0, [_det(track_id=7)])])

        assert [f.timestamp_s for f in emitted] == [0.0, 1.0]
        assert all(f.detections == [] for f in emitted)

    def test_track_seen_in_three_consecutive_frames_counts_from_its_first_frame(self):
        # Proof arrives on frame 3, but the person was there for all three. The
        # two withheld frames must be released with the detection intact —
        # forfeiting them would under-count every person by 2 s on arrival.
        filter_ = MinTrackLengthFilter()

        emitted = _run(filter_, [(float(t), [_det(track_id=7)]) for t in range(3)])

        assert [f.timestamp_s for f in emitted] == [0.0, 1.0, 2.0]
        assert [d.track_id for f in emitted for d in f.detections] == [7, 7, 7]

    def test_flickering_person_counts_despite_a_missed_frame(self):
        # Real YOLO output at 1 fps is not continuous. Measured on cctv-vps: a
        # genuinely-present person was detected in frames 0, 2, 3, 5, 6 — never
        # three *consecutive*. Demanding consecutive frames threw such people
        # away wholesale (a 7-detection track scored zero person-minutes),
        # trading the client's over-count for a worse under-count. Three
        # sightings inside five frames is a person.
        filter_ = MinTrackLengthFilter()

        emitted = _run(
            filter_,
            [
                (0.0, [_det(track_id=7)]),
                (1.0, []),
                (2.0, [_det(track_id=7)]),
                (3.0, [_det(track_id=7)]),
            ],
        )

        assert [d.track_id for f in emitted for d in f.detections] == [7, 7, 7]

    def test_sporadic_artifact_never_accumulates_its_way_in(self):
        # The bench-on-wheels: YOLO calls it a person twice, it vanishes for
        # ages, then twice more. Four hits in total, but never three close
        # together — so never a person. This is what the window must still
        # reject now that "consecutive" no longer guards the door.
        filter_ = MinTrackLengthFilter()

        script = [(0.0, [_det(track_id=7)]), (1.0, [_det(track_id=7)])]
        script += [(float(t), []) for t in range(2, 10)]
        script += [(10.0, [_det(track_id=7)]), (11.0, [_det(track_id=7)])]

        emitted = _run(filter_, script)

        assert all(f.detections == [] for f in emitted)

    def test_confirmed_track_survives_a_later_gap(self):
        # A worker proves they exist, steps behind a machine for a second, and
        # comes back. They must not have to re-earn their identity — that would
        # punch a hole in their person-minutes on every occlusion.
        filter_ = MinTrackLengthFilter()

        emitted = _run(
            filter_,
            [
                (0.0, [_det(track_id=7)]),
                (1.0, [_det(track_id=7)]),
                (2.0, [_det(track_id=7)]),
                (3.0, []),
                (4.0, [_det(track_id=7)]),
            ],
        )

        assert [d.track_id for d in emitted[4].detections] == [7]

    def test_untracked_detections_never_count(self):
        # track_id is None when the tracker declined to give a faint box an
        # identity. No identity, no proof, no person.
        filter_ = MinTrackLengthFilter()

        emitted = _run(filter_, [(float(t), [_det(track_id=None)]) for t in range(5)])

        assert all(f.detections == [] for f in emitted)
