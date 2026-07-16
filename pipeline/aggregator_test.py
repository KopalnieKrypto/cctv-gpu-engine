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

from pipeline.aggregator import (
    MAX_KEYFRAME_CANDIDATES,
    Aggregator,
    ReportData,
    ZoneReport,
)
from pipeline.postprocessing import Detection, Keypoint
from pipeline.zones import ShiftSchedule, Zone


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


class TestAggregatorShiftGating:
    """Issue #79 — only frames inside an active shift window are analysed.

    With a :class:`ShiftSchedule`, a frame whose wall-clock falls outside every
    working window (or inside a break) is dropped from *all* analysis counters —
    person-minutes, timeline, peak/average, keyframes — while the raw video
    length (``video_duration_s``) still reflects the whole recording. The report
    carries a summary of the analysed windows plus the total excluded duration.
    Without a schedule the aggregator behaves exactly as before.
    """

    def _schedule(self) -> ShiftSchedule:
        # Recording anchored at 06:00; window 07:00–15:00 minus an 11:00–11:20
        # break. So timestamp 3600 s → 07:00 (in), 0 s → 06:00 (before),
        # 18000 s → 11:00 (break), 36000 s → 16:00 (after).
        return ShiftSchedule.from_config(
            "2026-07-16T06:00:00+02:00",
            {"windows": [["07:00", "15:00"]], "breaks": [["11:00", "11:20"]]},
        )

    def test_frames_before_the_window_are_excluded_from_person_minutes(self):
        agg = Aggregator(fps=1, shift=self._schedule())
        for t in range(60):  # 06:00:00–06:00:59, before the shift → dropped
            agg.add_frame(float(t), _frame(), [_det("walking")])
        for t in range(3600, 3660):  # 07:00:00–07:00:59, in shift → 1.0 min
            agg.add_frame(float(t), _frame(), [_det("walking")])

        report = agg.build_report_data()

        assert report.person_minutes["walking"] == pytest.approx(1.0)
        assert report.total_frames == 60

    def test_frames_inside_a_break_are_excluded(self):
        agg = Aggregator(fps=1, shift=self._schedule())
        for t in range(18000, 18060):  # 11:00, inside the break → dropped
            agg.add_frame(float(t), _frame(), [_det("sitting")])

        report = agg.build_report_data()

        assert report.person_minutes["sitting"] == pytest.approx(0.0)
        assert report.total_frames == 0

    def test_excluded_frames_produce_no_timeline_bins(self):
        agg = Aggregator(fps=1, shift=self._schedule())
        for t in range(60):  # 06:00, excluded
            agg.add_frame(float(t), _frame(), [_det("walking")])
        for t in range(3600, 3660):  # 07:00 → minute 60
            agg.add_frame(float(t), _frame(), [_det("standing")])

        minutes = [b.minute for b in agg.build_report_data().timeline]

        assert minutes == [60]  # the 06:00 minute never appears

    def test_excluded_frames_do_not_affect_peak_or_average(self):
        agg = Aggregator(fps=1, shift=self._schedule())
        agg.add_frame(0.0, _frame(), [_det("standing")] * 10)  # 06:00, crowded, excluded
        agg.add_frame(3600.0, _frame(), [_det("standing")])  # 07:00, one person

        report = agg.build_report_data()

        assert report.peak_persons == 1
        assert report.avg_persons == pytest.approx(1.0)

    def test_excluded_frames_are_not_selected_as_keyframes(self):
        agg = Aggregator(
            fps=1, keyframe_count=8, keyframe_min_spacing_s=0.0, shift=self._schedule()
        )
        agg.add_frame(0.0, _frame(), [_det("standing")] * 5)  # 06:00, excluded
        agg.add_frame(3600.0, _frame(), [_det("walking")])  # 07:00, included

        timestamps = [kf.timestamp_s for kf in agg.build_report_data().keyframes]

        assert timestamps == pytest.approx([3600.0])

    def test_report_carries_shift_summary_with_windows_breaks_and_excluded_duration(self):
        agg = Aggregator(fps=1, shift=self._schedule())
        for t in range(60):  # 06:00 → 60 s excluded
            agg.add_frame(float(t), _frame(), [_det("walking")])
        for t in range(3600, 3660):  # 07:00 included
            agg.add_frame(float(t), _frame(), [_det("walking")])

        summary = agg.build_report_data().shift

        assert summary is not None
        assert summary.windows == [("07:00", "15:00")]
        assert summary.breaks == [("11:00", "11:20")]
        assert summary.excluded_duration_s == pytest.approx(60.0)

    def test_video_duration_reflects_full_length_including_an_excluded_tail(self):
        agg = Aggregator(fps=1, shift=self._schedule())
        agg.add_frame(3600.0, _frame(), [_det("walking")])  # 07:00, included
        agg.add_frame(36000.0, _frame(), [])  # 16:00, after shift → excluded, last frame

        report = agg.build_report_data()

        assert report.video_duration_s == pytest.approx(36000.0)
        assert report.total_frames == 1

    def test_without_a_schedule_the_summary_is_none_and_all_frames_count(self):
        agg = Aggregator(fps=1)
        for t in range(60):
            agg.add_frame(float(t), _frame(), [_det("walking")])

        report = agg.build_report_data()

        assert report.shift is None
        assert report.person_minutes["walking"] == pytest.approx(1.0)


def _tracked_det(track_id: int, zone_id: str, foot_x: float = 100.0) -> Detection:
    """A zoned detection with a track id and a foot point at ``foot_x``."""
    det = Detection(
        bbox=[foot_x - 5.0, 180.0, foot_x + 5.0, 200.0],  # foot = (foot_x, 200)
        confidence=0.9,
        keypoints=[Keypoint(0.0, 0.0, 0.0)] * 17,
    )
    det.activity = "standing"
    det.track_id = track_id
    det.zone_id = zone_id
    return det


class TestAggregatorZonePresence:
    """Issue #80 — per-zone anchored-worker presence attached to each ZoneReport.

    The aggregator feeds each shift-active frame's in-zone ``{track_id:
    foot_point}`` to a per-zone presence analyzer, so ``ZoneReport.presence``
    reports who the worker is (longest dwell) and when they were away — never a
    passer-by.
    """

    def _zone(self, rules=None) -> Zone:
        tri = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0)]
        return Zone(id="bending-1", name="Giętarka 1", polygon=tri, rules=rules or {})

    def test_presence_anchors_on_the_long_dwelling_track(self):
        agg = Aggregator(fps=1, zones=[self._zone()])
        # Track 1 works the station for 60 frames; track 2 just passes through.
        for t in range(60):
            dets = [_tracked_det(1, "bending-1")]
            if t in (30, 31):
                dets.append(_tracked_det(2, "bending-1", foot_x=400.0))
            agg.add_frame(float(t), _frame(), dets)

        presence = agg.build_report_data().zones[0].presence

        assert presence is not None
        assert presence.anchored_track_id == 1

    def test_anchor_is_none_when_detections_carry_no_track_ids(self):
        # --no-tracker leaves track_id None, so nothing can anchor.
        agg = Aggregator(fps=1, zones=[self._zone()])
        for t in range(10):
            agg.add_frame(float(t), _frame(), [_zoned_det("standing", "bending-1")])

        presence = agg.build_report_data().zones[0].presence

        assert presence is not None
        assert presence.anchored_track_id is None

    def test_zone_rules_configure_the_absence_flag_threshold(self):
        agg = Aggregator(fps=1, zones=[self._zone(rules={"absence": {"flag_after_s": 60.0}})])
        # Present 0–9 s, away, back 110–119 s: a 101 s gap the default (180 s)
        # would pass but the configured 60 s threshold flags.
        for t in list(range(10)) + list(range(110, 120)):
            agg.add_frame(float(t), _frame(), [_tracked_det(1, "bending-1")])

        (absence,) = agg.build_report_data().zones[0].presence.absence_intervals

        assert absence.duration_s == pytest.approx(101.0)
        assert absence.flagged is True

    def test_zone_rules_configure_min_move_px(self):
        agg = Aggregator(fps=1, zones=[self._zone(rules={"work": {"min_move_px": 10.0}})])
        # Foot point advances 20 px/frame — below the default 40 but above the
        # configured 10, so it registers as working.
        for t in range(10):
            agg.add_frame(float(t), _frame(), [_tracked_det(1, "bending-1", foot_x=t * 20.0)])

        work = agg.build_report_data().zones[0].presence.work_intervals

        assert len(work) == 1

    def test_presence_only_reflects_shift_active_frames(self):
        schedule = ShiftSchedule.from_config(
            "2026-07-16T06:00:00+02:00", {"windows": [["07:00", "15:00"]]}
        )
        agg = Aggregator(fps=1, zones=[self._zone()], shift=schedule)
        # 06:00 (before the shift) the worker is in the zone but excluded; the
        # analysed presence must start at 07:00, not 06:00.
        for t in range(60):  # 06:00:00–06:00:59, excluded
            agg.add_frame(float(t), _frame(), [_tracked_det(1, "bending-1")])
        for t in range(3600, 3660):  # 07:00:00–07:00:59, in shift
            agg.add_frame(float(t), _frame(), [_tracked_det(1, "bending-1")])

        presence = agg.build_report_data().zones[0].presence

        assert presence.anchored_track_id == 1
        assert presence.presence_intervals[0].start_s == 3600.0
