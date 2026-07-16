"""Anchored-worker presence/absence + ``work`` derivation (issue #80, slice 2).

A zone sees many tracks over a shift: the worker who staffs the station plus
passers-by who merely cross it. :class:`ZonePresenceAnalyzer` consumes one
``{track_id: foot_point}`` sighting map per analysed frame and, at the end,
picks the **anchored track** — the one with the longest cumulative dwell in the
zone — then reports that worker's presence/absence intervals and when they were
working. A passer-by never anchors, so a body crossing the zone can't be
mistaken for the worker.

Intervals are endpoint-based: a presence interval spans ``[first_seen,
last_seen]`` of an unbroken run of sightings, and the absence between two runs
spans ``[last_seen_before, first_seen_after]`` — so an absence's ``duration_s``
is exactly the wall time the worker was away, and presence.end meets
absence.start with no double counting. Two consecutive sightings belong to the
same run while their gap stays within a small tolerance of the sampling step;
any wider gap opens an absence (a brief detection dropout thus shows as a short,
unflagged absence rather than being silently bridged).

The analyzer is pure temporal logic: it knows nothing about ONNX, numpy, or the
pipeline. Callers (the aggregator) resolve which detections fall in the zone and
hand over ``{track_id: foot_point}`` per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]

# A run of sightings stays unbroken while consecutive gaps are within this
# multiple of the sampling step — 1.5 absorbs float jitter in the timestamps
# without bridging a genuine missing frame (gap ≈ 2× the step).
PRESENCE_MERGE_FACTOR = 1.5

# Absences longer than this (seconds) are flagged for the report. The client
# gave no bending-specific threshold, so 3 min is the default; per-zone config
# (``rules.absence.flag_after_s``) overrides it (issue #80).
DEFAULT_FLAG_AFTER_S = 180.0

# Minimum foot-point displacement (pixels) between consecutive sightings for the
# worker to count as *moving* — i.e. working the station rather than standing
# idle. Per-zone config (``rules.work.min_move_px``) overrides it (issue #80).
DEFAULT_MIN_MOVE_PX = 40.0


@dataclass(frozen=True)
class Interval:
    """A half-open-agnostic ``[start_s, end_s]`` span in recording seconds."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class Absence:
    """A stretch the anchored worker was away, flagged if it ran long."""

    start_s: float
    end_s: float
    flagged: bool

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class ZonePresence:
    """Derived presence facts for one zone's anchored worker."""

    anchored_track_id: int | None
    presence_intervals: tuple[Interval, ...] = ()
    absence_intervals: tuple[Absence, ...] = ()
    work_intervals: tuple[Interval, ...] = ()

    @property
    def present_s(self) -> float:
        """Total seconds the anchored worker was present in the zone."""
        return sum(iv.duration_s for iv in self.presence_intervals)

    @property
    def absent_s(self) -> float:
        """Total seconds the anchored worker was away between presences."""
        return sum(a.duration_s for a in self.absence_intervals)

    @property
    def work_s(self) -> float:
        """Total seconds the anchored worker was present and moving (working)."""
        return sum(iv.duration_s for iv in self.work_intervals)


class ZonePresenceAnalyzer:
    """Accumulates per-frame zone sightings into a :class:`ZonePresence`."""

    def __init__(
        self,
        sampling_step_s: float = 1.0,
        flag_after_s: float = DEFAULT_FLAG_AFTER_S,
        min_move_px: float = DEFAULT_MIN_MOVE_PX,
    ) -> None:
        self._sampling_step_s = sampling_step_s
        self._merge_gap_s = sampling_step_s * PRESENCE_MERGE_FACTOR
        self._flag_after_s = flag_after_s
        self._min_move_px = min_move_px
        # track_id → ordered (timestamp_s, foot_point) sightings inside the zone.
        self._sightings: dict[int, list[tuple[float, Point]]] = {}

    def observe(self, timestamp_s: float, sightings: dict[int, Point]) -> None:
        """Record which tracks stood in the zone at ``timestamp_s`` and where."""
        for track_id, foot_point in sightings.items():
            self._sightings.setdefault(track_id, []).append((timestamp_s, foot_point))

    def result(self) -> ZonePresence:
        """Resolve the anchored track and its presence/absence intervals."""
        anchor = self._anchored_track_id()
        if anchor is None:
            return ZonePresence(anchored_track_id=None)

        runs = self._runs(sorted(self._sightings[anchor]))
        presence = tuple(Interval(run[0][0], run[-1][0]) for run in runs)
        absence = tuple(
            self._absence(runs[i][-1][0], runs[i + 1][0][0]) for i in range(len(runs) - 1)
        )
        return ZonePresence(
            anchored_track_id=anchor,
            presence_intervals=presence,
            absence_intervals=absence,
            work_intervals=self._work_intervals(runs),
        )

    def _absence(self, start_s: float, end_s: float) -> Absence:
        return Absence(start_s, end_s, flagged=end_s - start_s > self._flag_after_s)

    def _work_intervals(self, runs: list[list[tuple[float, Point]]]) -> tuple[Interval, ...]:
        """Sub-spans of presence where the anchor's foot point actually moved.

        Movement is judged per consecutive sighting pair; adjacent moving pairs
        merge into one work interval. A stationary stretch inside a presence run
        splits the work around it, and a work interval never crosses an absence
        because each run is scanned on its own.
        """
        work: list[Interval] = []
        for run in runs:
            seg_start: float | None = None
            seg_end = 0.0
            for (t_prev, f_prev), (t_cur, f_cur) in zip(run, run[1:], strict=False):
                if math.dist(f_prev, f_cur) > self._min_move_px:
                    seg_start = t_prev if seg_start is None else seg_start
                    seg_end = t_cur
                elif seg_start is not None:
                    work.append(Interval(seg_start, seg_end))
                    seg_start = None
            if seg_start is not None:
                work.append(Interval(seg_start, seg_end))
        return tuple(work)

    def _anchored_track_id(self) -> int | None:
        """Track with the longest cumulative dwell (most sightings), or None."""
        if not self._sightings:
            return None
        return max(self._sightings, key=lambda tid: len(self._sightings[tid]))

    def _runs(self, sightings: list[tuple[float, Point]]) -> list[list[tuple[float, Point]]]:
        """Split ordered sightings into unbroken runs at gaps past the tolerance."""
        runs: list[list[tuple[float, Point]]] = [[sightings[0]]]
        for prev, cur in zip(sightings, sightings[1:], strict=False):
            if cur[0] - prev[0] <= self._merge_gap_s:
                runs[-1].append(cur)
            else:
                runs.append([cur])
        return runs
