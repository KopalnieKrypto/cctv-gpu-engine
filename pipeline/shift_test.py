"""Tests for shift-window gating (issue #79, zone pilot slice 1).

A *shift schedule* maps a frame's ``timestamp_s`` (seconds into the recording)
onto a wall-clock time, then decides whether that moment falls inside an active
working window — inside a configured window and outside every break. Only
active frames are analysed; the report reflects the working period the client
defines, not the whole recording.

These tests pin the public behaviour through :class:`ShiftSchedule`: the
timestamp→wall-clock mapping is tz-aware and DST-safe, and gating is exercised
inside a window / outside a window / inside a break, on the half-open
boundaries, and across a day rollover. Malformed schedules raise the same typed
:class:`ZoneConfigError` the rest of the zones config uses, not a bare error
deep in the pipeline.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from pipeline.zones import ShiftSchedule, ZoneConfig, ZoneConfigError


def _schedule() -> ShiftSchedule:
    """A 07:00–15:00 shift with an 11:00–11:20 break, recording from 06:00.

    ``recording_start`` is 06:00 local, so ``timestamp_s`` maps cleanly:
    3600 s → 07:00 (window start), 18000 s → 11:00 (break start), etc.
    """
    return ShiftSchedule.from_config(
        "2026-07-16T06:00:00+02:00",
        {"windows": [["07:00", "15:00"]], "breaks": [["11:00", "11:20"]]},
    )


class TestIsActive:
    def test_timestamp_inside_window_is_active(self):
        schedule = _schedule()

        # 06:00 + 3600 s = 07:00 — the window opens.
        assert schedule.is_active(3600.0) is True

    def test_timestamp_before_window_is_inactive(self):
        schedule = _schedule()

        # 06:00, an hour before the shift starts — nothing to count yet.
        assert schedule.is_active(0.0) is False

    def test_timestamp_after_window_is_inactive(self):
        schedule = _schedule()

        # 06:00 + 10 h = 16:00, an hour after the shift ends.
        assert schedule.is_active(10 * 3600.0) is False

    def test_timestamp_inside_break_is_excluded(self):
        schedule = _schedule()

        # 06:00 + 5 h = 11:00 → 11:10 is inside the 11:00–11:20 break.
        assert schedule.is_active(5 * 3600.0 + 600.0) is False

    def test_window_start_is_inclusive_but_end_is_exclusive(self):
        schedule = _schedule()

        # 07:00 exactly counts as working; 15:00 exactly does not.
        assert schedule.is_active(3600.0) is True  # 07:00
        assert schedule.is_active(9 * 3600.0) is False  # 15:00

    def test_break_start_is_excluded_but_break_end_returns_to_work(self):
        schedule = _schedule()

        # 11:00 exactly is on break; 11:20 exactly is back at the station.
        assert schedule.is_active(5 * 3600.0) is False  # 11:00
        assert schedule.is_active(5 * 3600.0 + 20 * 60.0) is True  # 11:20

    def test_window_recurs_the_next_day(self):
        # A recording spanning midnight reuses the same daily schedule: the
        # window is a time-of-day range, not a one-shot absolute interval.
        schedule = _schedule()

        # 06:00 + 25 h = 07:00 the following day — inside the window again.
        assert schedule.is_active(25 * 3600.0) is True
        # 06:00 + 24 h = 06:00 next day — still before that day's window.
        assert schedule.is_active(24 * 3600.0) is False


class TestWallClockMapping:
    def test_maps_elapsed_seconds_to_local_wall_clock(self):
        schedule = _schedule()

        wall = schedule.wall_clock_at(3600.0)

        assert wall.strftime("%H:%M") == "07:00"
        assert wall.utcoffset() == timedelta(hours=2)  # +02:00 as configured

    def test_mapping_is_dst_safe_across_the_spring_forward(self):
        # Poland springs forward 2026-03-29 02:00 → 03:00. A recording that
        # starts at 01:30 (winter, +01:00) and runs one real hour must read
        # 03:30 on the wall — the clock jumped an hour mid-recording. A frozen
        # offset would wrongly read 02:30, a wall-clock time that never existed.
        schedule = ShiftSchedule.from_config(
            "2026-03-29T01:30:00+01:00",
            {"timezone": "Europe/Warsaw", "windows": [["00:00", "23:59"]]},
        )

        wall = schedule.wall_clock_at(3600.0)  # +1 real hour

        assert wall.strftime("%H:%M") == "03:30"
        assert wall.utcoffset() == timedelta(hours=2)  # now on summer time


class TestValidation:
    def test_malformed_recording_start_raises_typed_error(self):
        with pytest.raises(ZoneConfigError):
            ShiftSchedule.from_config("not-a-timestamp", {"windows": [["07:00", "15:00"]]})

    def test_unknown_timezone_raises_typed_error(self):
        with pytest.raises(ZoneConfigError):
            ShiftSchedule.from_config(
                "2026-07-16T06:00:00+02:00",
                {"timezone": "Mars/Olympus", "windows": [["07:00", "15:00"]]},
            )

    def test_malformed_window_time_raises_typed_error(self):
        with pytest.raises(ZoneConfigError):
            ShiftSchedule.from_config("2026-07-16T06:00:00+02:00", {"windows": [["7am", "15:00"]]})

    def test_window_with_end_not_after_start_raises_typed_error(self):
        with pytest.raises(ZoneConfigError):
            ShiftSchedule.from_config(
                "2026-07-16T06:00:00+02:00", {"windows": [["15:00", "07:00"]]}
            )

    def test_naive_recording_start_without_timezone_is_rejected(self):
        # Without an offset or an IANA zone the anchor is ambiguous in absolute
        # time — refuse it rather than guess a timezone.
        with pytest.raises(ZoneConfigError):
            ShiftSchedule.from_config("2026-07-16T06:00:00", {"windows": [["07:00", "15:00"]]})

    def test_naive_recording_start_is_localized_by_timezone(self):
        schedule = ShiftSchedule.from_config(
            "2026-07-16T06:00:00",
            {"timezone": "Europe/Warsaw", "windows": [["07:00", "15:00"]]},
        )

        # 06:00 local + 1 h = 07:00 → inside the window.
        assert schedule.is_active(3600.0) is True


class TestZoneConfigShiftSchedule:
    """The shift schedule is reachable off a loaded ZoneConfig (issue #79)."""

    def _config(self, **extra) -> ZoneConfig:
        data: dict = {"zones": [{"id": "z", "name": "z", "polygon": [[0, 0], [1, 0], [1, 1]]}]}
        data.update(extra)
        return ZoneConfig.from_dict(data)

    def test_config_without_shift_has_no_schedule(self):
        assert self._config().shift_schedule is None

    def test_config_with_shift_builds_a_working_schedule(self):
        config = self._config(
            recording_start="2026-07-16T06:00:00+02:00",
            shift={"windows": [["07:00", "15:00"]], "breaks": [["11:00", "11:20"]]},
        )

        schedule = config.shift_schedule
        assert isinstance(schedule, ShiftSchedule)
        assert schedule.is_active(3600.0) is True  # 07:00, working
        assert schedule.is_active(5 * 3600.0 + 600.0) is False  # 11:10, on break

    def test_shift_without_recording_start_is_rejected_at_load(self):
        with pytest.raises(ZoneConfigError):
            self._config(shift={"windows": [["07:00", "15:00"]]})
