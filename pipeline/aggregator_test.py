"""Tests for video-level aggregation: per-frame detections → ReportData.

The aggregator is the heart of issue #4: it accumulates results from each
processed frame (person counts, activities, candidate keyframes) and at the
end produces a fully-populated ReportData ready for the report renderer.

These tests cover behaviour, not implementation — they pass real Detection
objects through the public ``add_frame`` / ``build_report_data`` interface
so the internal storage can change without breaking the tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.aggregator import MAX_KEYFRAME_CANDIDATES, Aggregator, ReportData, ZoneReport
from pipeline.postprocessing import Detection, Keypoint
from pipeline.zones import Zone


def _det(activity: str, bbox=(0.0, 0.0, 10.0, 20.0), confidence: float = 0.9) -> Detection:
    det = Detection(
        bbox=list(bbox),
        confidence=confidence,
        keypoints=[Keypoint(0.0, 0.0, 0.0)] * 17,
    )
    det.activity = activity
    return det


def _zoned_det(activity: str, zone_id: str | None) -> Detection:
    det = _det(activity)
    det.zone_id = zone_id
    return det


def _frame(h: int = 4, w: int = 4) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestAggregatorEmptyState:
    def test_empty_aggregator_produces_zero_report_data(self):
        agg = Aggregator(fps=1)

        report = agg.build_report_data()

        assert isinstance(report, ReportData)
        assert report.total_frames == 0
        assert report.video_duration_s == 0.0
        assert report.peak_persons == 0
        assert report.avg_persons == 0.0
        assert report.dominant_activity == "none"
        assert report.person_minutes == {
            "sitting": pytest.approx(0.0),
            "standing": pytest.approx(0.0),
            "walking": pytest.approx(0.0),
            "running": pytest.approx(0.0),
        }
        assert report.timeline == []
        assert report.keyframes == []


class TestAggregatorPersonMinutes:
    def test_60_frames_with_one_walking_person_at_1fps_is_one_walking_minute(self):
        agg = Aggregator(fps=1)
        for t in range(60):
            agg.add_frame(timestamp_s=float(t), frame=_frame(), detections=[_det("walking")])

        report = agg.build_report_data()

        assert report.person_minutes["walking"] == pytest.approx(1.0)
        assert report.person_minutes["standing"] == pytest.approx(0.0)
        assert report.person_minutes["sitting"] == pytest.approx(0.0)
        assert report.person_minutes["running"] == pytest.approx(0.0)
        assert report.total_frames == 60

    def test_two_persons_per_frame_doubles_person_minutes(self):
        # 30 frames × 2 standing persons = 60 standing person-frames = 1.0 standing-min
        agg = Aggregator(fps=1)
        for t in range(30):
            agg.add_frame(
                timestamp_s=float(t),
                frame=_frame(),
                detections=[_det("standing"), _det("standing")],
            )

        report = agg.build_report_data()

        assert report.person_minutes["standing"] == pytest.approx(1.0)
        assert report.peak_persons == 2


class TestAggregatorTimeline:
    def test_90_frames_at_1fps_fall_into_two_one_minute_bins(self):
        # 60 walking frames in minute 0, 30 standing frames in minute 1
        agg = Aggregator(fps=1)
        for t in range(60):
            agg.add_frame(timestamp_s=float(t), frame=_frame(), detections=[_det("walking")])
        for t in range(60, 90):
            agg.add_frame(
                timestamp_s=float(t),
                frame=_frame(),
                detections=[_det("standing"), _det("standing")],
            )

        report = agg.build_report_data()

        assert len(report.timeline) == 2
        bin0, bin1 = report.timeline
        assert bin0.minute == 0
        assert bin0.walking == 60
        assert bin0.standing == 0
        assert bin1.minute == 1
        assert bin1.walking == 0
        assert bin1.standing == 60  # 30 frames × 2 persons


class TestAggregatorPeakAndAverage:
    def test_peak_persons_is_max_across_frames(self):
        agg = Aggregator(fps=1)
        # frame 0: 1 person, frame 1: 5 persons, frame 2: 2 persons
        agg.add_frame(0.0, _frame(), [_det("standing")])
        agg.add_frame(1.0, _frame(), [_det("standing")] * 5)
        agg.add_frame(2.0, _frame(), [_det("walking")] * 2)

        report = agg.build_report_data()

        assert report.peak_persons == 5
        # avg = (1 + 5 + 2) / 3
        assert report.avg_persons == pytest.approx(8 / 3)

    def test_average_excludes_division_by_zero_for_empty_aggregator(self):
        report = Aggregator().build_report_data()

        assert report.avg_persons == 0.0  # not NaN, not ZeroDivisionError


class TestAggregatorDominantActivity:
    def test_dominant_activity_picks_max_person_frames(self):
        agg = Aggregator(fps=1)
        # 3 walking frames, 2 standing frames, 1 sitting frame
        for t in range(3):
            agg.add_frame(float(t), _frame(), [_det("walking")])
        for t in range(3, 5):
            agg.add_frame(float(t), _frame(), [_det("standing")])
        agg.add_frame(5.0, _frame(), [_det("sitting")])

        report = agg.build_report_data()

        assert report.dominant_activity == "walking"

    def test_dominant_activity_is_none_when_no_persons_detected(self):
        agg = Aggregator(fps=1)
        for t in range(10):
            agg.add_frame(float(t), _frame(), [])  # frames processed but no people

        report = agg.build_report_data()

        assert report.dominant_activity == "none"
        assert report.total_frames == 10


class TestAggregatorKeyframes:
    def test_selects_top_k_frames_by_person_count(self):
        # 3 frames spaced far apart with different person counts
        agg = Aggregator(fps=1, keyframe_count=2, keyframe_min_spacing_s=0)
        agg.add_frame(0.0, _frame(), [_det("standing")])  # 1 person
        agg.add_frame(200.0, _frame(), [_det("standing")] * 5)  # 5 persons (top)
        agg.add_frame(400.0, _frame(), [_det("standing")] * 3)  # 3 persons (2nd)

        report = agg.build_report_data()

        assert len(report.keyframes) == 2
        # ordered by person_count descending
        assert report.keyframes[0].person_count == 5
        assert report.keyframes[0].timestamp_s == pytest.approx(200.0)
        assert report.keyframes[1].person_count == 3
        assert report.keyframes[1].timestamp_s == pytest.approx(400.0)
        # Each keyframe carries the raw frame for the renderer to annotate later
        assert isinstance(report.keyframes[0].frame, np.ndarray)
        assert len(report.keyframes[0].detections) == 5

    def test_skips_frames_with_zero_persons(self):
        agg = Aggregator(fps=1, keyframe_count=5, keyframe_min_spacing_s=0)
        agg.add_frame(0.0, _frame(), [])
        agg.add_frame(10.0, _frame(), [])
        agg.add_frame(20.0, _frame(), [_det("walking")])

        report = agg.build_report_data()

        assert len(report.keyframes) == 1
        assert report.keyframes[0].timestamp_s == pytest.approx(20.0)

    def test_enforces_min_spacing_between_keyframes(self):
        # 4 candidate frames; the 2nd-best is too close to the best and must
        # be skipped in favour of frames further away.
        agg = Aggregator(fps=1, keyframe_count=2, keyframe_min_spacing_s=120.0)
        agg.add_frame(0.0, _frame(), [_det("standing")])  # 1 person
        agg.add_frame(60.0, _frame(), [_det("standing")] * 9)  # 9 — best
        agg.add_frame(100.0, _frame(), [_det("standing")] * 8)  # 8 but only 40s away → skip
        agg.add_frame(300.0, _frame(), [_det("standing")] * 5)  # 5, far enough → keep

        report = agg.build_report_data()

        timestamps = [k.timestamp_s for k in report.keyframes]
        assert timestamps == pytest.approx([60.0, 300.0])

    def test_returns_fewer_than_count_for_short_video(self):
        # Only 2 frames have people; cannot return 5
        agg = Aggregator(fps=1, keyframe_count=5, keyframe_min_spacing_s=0)
        agg.add_frame(0.0, _frame(), [_det("walking")])
        agg.add_frame(5.0, _frame(), [_det("walking")] * 2)

        report = agg.build_report_data()

        assert len(report.keyframes) == 2

    def test_includes_at_least_one_keyframe_per_activity(self):
        """Each activity that appears in the video gets a representative keyframe.

        Without per-activity coverage, the report could legitimately ship 5
        ``walking`` keyframes (because walking dominates by frame count) and
        zero ``sitting`` even though sitting is present in the timeline —
        the operator scanning the keyframes section gets a misleading
        impression of what happened.
        """
        agg = Aggregator(fps=1, keyframe_count=8, keyframe_min_spacing_s=10.0)
        # Walking dominates by count; one short sitting + one short standing.
        for t in range(20):
            agg.add_frame(float(t), _frame(), [_det("walking")])
        agg.add_frame(50.0, _frame(), [_det("sitting")])
        agg.add_frame(80.0, _frame(), [_det("standing")])

        activities = {kf.detections[0].activity for kf in agg.build_report_data().keyframes}

        assert {"walking", "sitting", "standing"}.issubset(activities)

    def test_keyframes_returned_in_chronological_order(self):
        """Sorting by timestamp produces a narrative tour through the video."""
        agg = Aggregator(fps=1, keyframe_count=4, keyframe_min_spacing_s=0)
        agg.add_frame(100.0, _frame(), [_det("walking")] * 3)
        agg.add_frame(10.0, _frame(), [_det("sitting")])
        agg.add_frame(200.0, _frame(), [_det("standing")] * 2)

        timestamps = [kf.timestamp_s for kf in agg.build_report_data().keyframes]

        assert timestamps == sorted(timestamps)


class TestAggregatorCandidateBuffer:
    """Issue #49: the candidate keyframe buffer must be provably bounded.

    ``add_frame`` used to append ``frame.copy()`` for *every* frame with a
    person, with no cap — a 1 h 1080p video of a continuously-present person
    accumulates ~3600 frames ≈ 21 GiB RSS and OOMs the gpu-service container.
    This violates the CLAUDE.md hard rule "frame-by-frame processing, never
    full video in RAM". The buffer only ever needs ~keyframe_count frames.
    """

    def test_candidate_buffer_is_bounded_by_constant(self):
        # A person in every one of 500 frames must NOT retain 500 raw frames.
        agg = Aggregator(fps=1)
        for t in range(500):
            agg.add_frame(float(t), _frame(64, 64), [_det("standing")])

        assert agg.candidate_count <= MAX_KEYFRAME_CANDIDATES

    def test_sole_occurrence_of_an_activity_survives_eviction_flood(self):
        """Eviction must not drop the ONLY frame of a rare activity.

        Regression guard against a naive "evict lowest person_count" bound: an
        early single-person ``running`` frame is the lowest-value candidate,
        yet it is the sole evidence of running in the whole video. Per-activity
        coverage (an existing keyframe guarantee) must survive bounding, so the
        report still shows what happened rather than only the crowded frames.
        """
        agg = Aggregator(fps=1, keyframe_count=8, keyframe_min_spacing_s=0.0)
        # One low-value running frame very early...
        agg.add_frame(0.0, _frame(64, 64), [_det("running")])
        # ...then a flood of higher-person-count walking frames.
        for t in range(1, 400):
            agg.add_frame(float(t), _frame(64, 64), [_det("walking")] * 3)

        activities = {kf.detections[0].activity for kf in agg.build_report_data().keyframes}

        assert "running" in activities
        assert "walking" in activities

    def test_memory_footprint_stays_flat_regardless_of_video_length(self):
        """Retained keyframe bytes stay bounded as frame count scales (AC #2).

        The OOM risk in issue #49 comes entirely from retained ``frame.copy()``
        buffers, so summing ``kf.frame.nbytes`` over the candidate buffer is the
        memory that actually matters. (We assert on nbytes rather than
        ``tracemalloc`` because numpy allocates its data buffers outside
        CPython's allocator, so tracemalloc does not see the frame bytes — the
        exact allocation that causes the OOM.) On the old unbounded code, 1000
        frames of 640×360 retain ~1.2 GiB; the bound caps it near 42 MiB.

        Linear-extrapolation assumption: bytes-per-frame is constant, so a bound
        that holds at 1000 frames holds for a 1 h (3600-frame) video too.
        """
        frame_bytes = 360 * 640 * 3  # 640×360 BGR uint8 ≈ 0.66 MiB
        agg = Aggregator(fps=1)
        for t in range(1000):
            agg.add_frame(float(t), _frame(360, 640), [_det("walking")])

        retained_bytes = sum(c.frame.nbytes for c in agg._candidates)

        assert retained_bytes <= 2 * MAX_KEYFRAME_CANDIDATES * frame_bytes


class TestAggregatorPerZone:
    """Issue #78 — person-minutes bucketed per ROI zone.

    Without a zone config the aggregator behaves exactly as before (no zone
    breakdown). With one, each detection accrues to the zone stamped on its
    ``zone_id``; the global totals are unchanged so existing consumers keep
    working, and every configured zone always emits a full four-bucket row.
    """

    def _zones(self) -> list[Zone]:
        tri = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        return [
            Zone(id="bending-1", name="Giętarka 1", polygon=tri, rules={}),
            Zone(id="welding-1", name="Spawalnia 1", polygon=tri, rules={}),
        ]

    def test_no_zone_breakdown_without_config(self):
        agg = Aggregator(fps=1)
        for t in range(60):
            agg.add_frame(float(t), _frame(), [_det("walking")])

        assert agg.build_report_data().zones == []

    def test_two_detections_in_different_zones_aggregate_separately(self):
        agg = Aggregator(fps=1, zones=self._zones())
        # 60 frames: one worker sitting in bending-1, one standing in welding-1.
        for t in range(60):
            agg.add_frame(
                float(t),
                _frame(),
                [_zoned_det("sitting", "bending-1"), _zoned_det("standing", "welding-1")],
            )

        by_id = {z.zone_id: z for z in agg.build_report_data().zones}
        assert set(by_id) == {"bending-1", "welding-1"}
        assert isinstance(by_id["bending-1"], ZoneReport)
        assert by_id["bending-1"].name == "Giętarka 1"
        assert by_id["bending-1"].person_minutes["sitting"] == pytest.approx(1.0)
        assert by_id["bending-1"].person_minutes["standing"] == pytest.approx(0.0)
        assert by_id["welding-1"].person_minutes["standing"] == pytest.approx(1.0)
        assert by_id["welding-1"].person_minutes["sitting"] == pytest.approx(0.0)

    def test_every_configured_zone_emits_all_four_buckets_even_when_empty(self):
        agg = Aggregator(fps=1, zones=self._zones())
        agg.add_frame(0.0, _frame(), [_zoned_det("sitting", "bending-1")])

        by_id = {z.zone_id: z for z in agg.build_report_data().zones}
        assert set(by_id["bending-1"].person_minutes) == {
            "sitting",
            "standing",
            "walking",
            "running",
        }
        # welding-1 saw nobody but still reports a zero row.
        assert by_id["welding-1"].person_minutes["sitting"] == pytest.approx(0.0)

    def test_unzoned_detection_counts_globally_but_in_no_zone(self):
        agg = Aggregator(fps=1, zones=self._zones())
        for t in range(60):
            agg.add_frame(float(t), _frame(), [_zoned_det("walking", None)])

        report = agg.build_report_data()
        assert report.person_minutes["walking"] == pytest.approx(1.0)  # global unchanged
        by_id = {z.zone_id: z for z in report.zones}
        assert by_id["bending-1"].person_minutes["walking"] == pytest.approx(0.0)
        assert by_id["welding-1"].person_minutes["walking"] == pytest.approx(0.0)
