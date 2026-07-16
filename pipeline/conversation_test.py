"""Tests for zone conversation-mode detection (issue #81, slice 3).

A *conversation* is two people standing together in the zone: ≥2 stable tracks
whose foot points are mutually close and who are both barely moving. It completes
the ``work`` / ``conversation`` / ``absent`` zone-mode set alongside the
anchored-worker presence analysis (#80).

These tests pin the public behaviour through ``observe`` / ``result``: feed one
``{track_id: foot_point}`` sighting map per analysed frame, then read the
:class:`ZoneConversation`. Like the presence analyzer, nothing here touches the
pipeline, ONNX, or numpy — it is pure geometric/temporal logic over sightings.
"""

from __future__ import annotations

from pipeline.conversation import ConversationAnalyzer


class TestConversation:
    """Two stationary, proximate tracks read as a conversation interval."""

    def test_two_stationary_proximate_tracks_form_a_conversation(self):
        analyzer = ConversationAnalyzer(sampling_step_s=1.0, proximity_px=150.0)
        # Two workers stand shoulder to shoulder (50 px apart), unmoving, 0–9 s.
        for t in range(10):
            analyzer.observe(float(t), {1: (100.0, 200.0), 2: (150.0, 200.0)})

        result = analyzer.result()

        assert len(result.intervals) == 1
        assert result.intervals[0].start_s == 0.0
        assert result.intervals[0].end_s == 9.0
        assert result.conversation_s == 9.0

    def test_a_moving_partner_is_working_not_conversing(self):
        analyzer = ConversationAnalyzer(sampling_step_s=1.0, proximity_px=150.0, min_move_px=40.0)
        # Track 1 stands idle; track 2 stays near it but jitters 50 px/frame —
        # moving more than the work threshold, so it is working, not conversing.
        for t in range(10):
            near = 150.0 + (t % 2) * 50.0  # 150↔200: distance ≤100 (proximate)
            analyzer.observe(float(t), {1: (100.0, 200.0), 2: (near, 200.0)})

        result = analyzer.result()

        assert result.intervals == ()
        assert result.conversation_s == 0.0

    def test_two_idle_but_distant_tracks_are_not_conversing(self):
        analyzer = ConversationAnalyzer(sampling_step_s=1.0, proximity_px=150.0)
        # Both planted and idle, but 300 px apart — across the zone, not together.
        for t in range(10):
            analyzer.observe(float(t), {1: (100.0, 200.0), 2: (400.0, 200.0)})

        assert analyzer.result().intervals == ()

    def test_proximity_px_is_configurable(self):
        # Foot points 200 px apart: beyond the default 150, within a wider 250.
        def conversing_at(proximity_px: float) -> bool:
            analyzer = ConversationAnalyzer(sampling_step_s=1.0, proximity_px=proximity_px)
            for t in range(10):
                analyzer.observe(float(t), {1: (100.0, 200.0), 2: (300.0, 200.0)})
            return bool(analyzer.result().intervals)

        assert conversing_at(150.0) is False  # too far apart to be together
        assert conversing_at(250.0) is True  # a looser zone counts them as one

    def test_min_move_px_is_configurable(self):
        # Both drift 20 px/frame in step, staying 50 px apart: under the default
        # 40 they read as idle-and-together, under a stricter 10 they are moving.
        def conversing_at(min_move_px: float) -> bool:
            analyzer = ConversationAnalyzer(
                sampling_step_s=1.0, proximity_px=150.0, min_move_px=min_move_px
            )
            for t in range(10):
                analyzer.observe(float(t), {1: (t * 20.0, 200.0), 2: (t * 20.0 + 50.0, 200.0)})
            return bool(analyzer.result().intervals)

        assert conversing_at(40.0) is True  # 20 px drift counts as standing still
        assert conversing_at(10.0) is False  # now that drift reads as movement


class TestConversationEmptyState:
    def test_no_observations_yields_no_conversation(self):
        result = ConversationAnalyzer().result()

        assert result.intervals == ()
        assert result.conversation_s == 0.0

    def test_a_lone_track_never_converses(self):
        analyzer = ConversationAnalyzer(sampling_step_s=1.0)
        # One worker at the station the whole time — nobody to talk to.
        for t in range(20):
            analyzer.observe(float(t), {1: (100.0, 200.0)})

        assert analyzer.result().intervals == ()


class TestConversationTimeline:
    """Conversation resolves into split intervals — the report's mode timeline."""

    def test_partner_leaving_and_returning_splits_into_two_intervals(self):
        analyzer = ConversationAnalyzer(sampling_step_s=1.0, proximity_px=150.0, min_move_px=40.0)
        # Together 0–2 s, partner walks off across the zone 3–5 s, back 6–9 s.
        far = {3: (400.0, 200.0), 4: (400.0, 200.0), 5: (400.0, 200.0)}
        for t in range(10):
            partner = far.get(t, (150.0, 200.0))
            analyzer.observe(float(t), {1: (100.0, 200.0), 2: partner})

        intervals = analyzer.result().intervals

        assert len(intervals) == 2
        assert (intervals[0].start_s, intervals[0].end_s) == (0.0, 2.0)
        # Partner is away through 5 s and back at 6 s, so the pair (6, 7) is the
        # first to re-converse — the second interval opens at 6, not 5.
        assert (intervals[1].start_s, intervals[1].end_s) == (6.0, 9.0)

    def test_detection_dropout_does_not_bridge_two_conversations(self):
        analyzer = ConversationAnalyzer(sampling_step_s=1.0, proximity_px=150.0)
        # Two idle proximate tracks, a long detection blackout, then again.
        for t in list(range(5)) + list(range(200, 205)):
            analyzer.observe(float(t), {1: (100.0, 200.0), 2: (150.0, 200.0)})

        intervals = analyzer.result().intervals

        assert len(intervals) == 2
        assert (intervals[0].start_s, intervals[0].end_s) == (0.0, 4.0)
        assert (intervals[1].start_s, intervals[1].end_s) == (200.0, 204.0)


class TestConversationWithBystanders:
    def test_a_moving_passerby_does_not_disturb_a_conversation(self):
        analyzer = ConversationAnalyzer(sampling_step_s=1.0, proximity_px=150.0, min_move_px=40.0)
        # Two workers chat idle and close; a third track streaks across the zone.
        for t in range(10):
            sightings = {1: (100.0, 200.0), 2: (150.0, 200.0)}
            sightings[3] = (t * 300.0, 500.0)  # far and fast — a passer-by
            analyzer.observe(float(t), sightings)

        result = analyzer.result()

        assert len(result.intervals) == 1
        assert (result.intervals[0].start_s, result.intervals[0].end_s) == (0.0, 9.0)
