"""Tests for anchored-worker presence/absence + work derivation (issue #80).

A zone sees many tracks over a shift — the worker at the station plus
passers-by crossing it. The :class:`ZonePresenceAnalyzer` anchors on the one
track that actually *works* the station (longest cumulative dwell), then reports
when that worker was present, absent, and moving — never a passer-by.

These tests pin the public behaviour through ``observe`` / ``result``: feed one
``{track_id: foot_point}`` sighting map per analysed frame, then read the
:class:`ZonePresence`. Nothing here touches the pipeline, ONNX, or numpy — the
analyzer is pure temporal logic over sightings.
"""

from __future__ import annotations

from pipeline.presence import ZonePresenceAnalyzer


class TestAnchoredTrack:
    """The anchor is the track with the longest cumulative dwell in the zone."""

    def test_long_dwelling_worker_anchors_over_a_transiting_passerby(self):
        analyzer = ZonePresenceAnalyzer(sampling_step_s=1.0)

        # Track 1 works the station for 60 frames; track 2 merely crosses it for
        # two frames on its way past. The worker must anchor, not the passer-by.
        for t in range(60):
            sightings = {1: (100.0, 200.0)}
            if t in (30, 31):
                sightings[2] = (400.0, 200.0)
            analyzer.observe(float(t), sightings)

        assert analyzer.result().anchored_track_id == 1


class TestEmptyState:
    def test_no_observations_yields_no_anchor_and_no_intervals(self):
        result = ZonePresenceAnalyzer().result()

        assert result.anchored_track_id is None
        assert result.presence_intervals == ()
        assert result.absence_intervals == ()


class TestPresenceIntervals:
    def test_contiguous_sightings_form_one_presence_interval(self):
        analyzer = ZonePresenceAnalyzer(sampling_step_s=1.0)

        # The worker is seen every frame from 10 s to 19 s — one unbroken stay.
        for t in range(10, 20):
            analyzer.observe(float(t), {1: (100.0, 200.0)})

        intervals = analyzer.result().presence_intervals

        assert len(intervals) == 1
        assert intervals[0].start_s == 10.0
        assert intervals[0].end_s == 19.0
        assert intervals[0].duration_s == 9.0


class TestAbsenceIntervals:
    def test_gap_between_two_stays_is_one_absence_spanning_the_gap(self):
        analyzer = ZonePresenceAnalyzer(sampling_step_s=1.0)

        # Present 10–19 s, gone, present again 200–209 s: one absence in between.
        for t in list(range(10, 20)) + list(range(200, 210)):
            analyzer.observe(float(t), {1: (100.0, 200.0)})

        result = analyzer.result()

        assert len(result.presence_intervals) == 2
        assert len(result.absence_intervals) == 1
        absence = result.absence_intervals[0]
        # "Time away" = from last seen before the gap to first seen after it.
        assert absence.start_s == 19.0
        assert absence.end_s == 200.0
        assert absence.duration_s == 181.0
        # Presence meets absence with no overlap or double counting.
        assert result.presence_intervals[0].end_s == absence.start_s
        assert result.presence_intervals[1].start_s == absence.end_s


class TestAbsenceFlagging:
    def _absence_of(self, gap_s: float, **kwargs) -> object:
        analyzer = ZonePresenceAnalyzer(sampling_step_s=1.0, **kwargs)
        for t in range(10, 20):
            analyzer.observe(float(t), {1: (0.0, 0.0)})
        resume = 19.0 + gap_s
        for t in range(int(resume), int(resume) + 10):
            analyzer.observe(float(t), {1: (0.0, 0.0)})
        (absence,) = analyzer.result().absence_intervals
        return absence

    def test_absence_longer_than_threshold_is_flagged(self):
        absence = self._absence_of(gap_s=181.0)  # default flag_after_s = 180

        assert absence.duration_s == 181.0
        assert absence.flagged is True

    def test_absence_within_threshold_is_not_flagged(self):
        absence = self._absence_of(gap_s=100.0)

        assert absence.flagged is False

    def test_flag_threshold_is_configurable(self):
        # A stricter 60 s threshold flags a 100 s absence the default would pass.
        absence = self._absence_of(gap_s=100.0, flag_after_s=60.0)

        assert absence.flagged is True

    def test_passerby_during_the_gap_does_not_fill_the_anchor_absence(self):
        analyzer = ZonePresenceAnalyzer(sampling_step_s=1.0)
        for t in range(10, 20):
            analyzer.observe(float(t), {1: (0.0, 0.0)})
        # A different track crosses the zone mid-gap — irrelevant to the worker.
        for t in (100, 101):
            analyzer.observe(float(t), {2: (500.0, 200.0)})
        for t in range(200, 210):
            analyzer.observe(float(t), {1: (0.0, 0.0)})

        result = analyzer.result()

        assert result.anchored_track_id == 1
        (absence,) = result.absence_intervals
        assert absence.start_s == 19.0
        assert absence.end_s == 200.0


class TestWorkMode:
    """``work`` = anchored worker present AND moving between foot-point samples."""

    def test_stationary_presence_is_not_work(self):
        analyzer = ZonePresenceAnalyzer(sampling_step_s=1.0)
        # Present but planted at one spot — standing/idle, not working.
        for t in range(10):
            analyzer.observe(float(t), {1: (100.0, 200.0)})

        assert analyzer.result().work_intervals == ()

    def test_pacing_worker_produces_a_work_interval(self):
        analyzer = ZonePresenceAnalyzer(sampling_step_s=1.0, min_move_px=40.0)
        # Foot point advances 50 px/frame (> 40) — the worker paces the station.
        for t in range(10):
            analyzer.observe(float(t), {1: (t * 50.0, 200.0)})

        work = analyzer.result().work_intervals

        assert len(work) == 1
        assert work[0].start_s == 0.0
        assert work[0].end_s == 9.0

    def test_min_move_px_is_configurable(self):
        # 20 px/frame is below the default 40 but above a 10 px threshold.
        def work_for(min_move_px):
            analyzer = ZonePresenceAnalyzer(sampling_step_s=1.0, min_move_px=min_move_px)
            for t in range(10):
                analyzer.observe(float(t), {1: (t * 20.0, 200.0)})
            return analyzer.result().work_intervals

        assert work_for(40.0) == ()  # too small to count as movement
        assert len(work_for(10.0)) == 1  # now it registers as pacing

    def test_only_the_anchors_movement_counts_as_work(self):
        analyzer = ZonePresenceAnalyzer(sampling_step_s=1.0, min_move_px=40.0)
        # The worker stands still; a passer-by streaks across — not the worker's
        # movement, so no work is credited.
        for t in range(20):
            sightings = {1: (100.0, 200.0)}
            if t in (5, 6):
                sightings[2] = (t * 300.0, 200.0)
            analyzer.observe(float(t), sightings)

        result = analyzer.result()

        assert result.anchored_track_id == 1
        assert result.work_intervals == ()
