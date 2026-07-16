"""Zone configuration and foot-point assignment (issue #78, slice 0).

A *zone* is a named ROI polygon over a workstation (e.g. a bending station),
plus a per-zone rule set and — for later slices — a shift schedule. Zones are
supplied as a config file (``--zones zones.json``), never hardcoded, so a
future welding zone can carry different rules (#82) without a code change.

Assignment is by **foot point**: a detection belongs to the zone whose polygon
contains the midpoint of its bounding-box bottom edge — where the person is
standing on the floor, not where the (perspective-skewed) box center sits. A
detection outside every zone has ``zone_id`` ``None``.

The polygon test treats a point exactly on an edge or vertex as *inside*: a
worker with a foot planted on the zone boundary is at the station, and edge
flicker between "in" and "out" would corrupt person-minute totals.

Only the pieces slice 0 needs are validated here. ``recording_start`` and
``shift`` are parsed into a :class:`ShiftSchedule` for the shift-gating slice
(#79) that maps a frame's ``timestamp_s`` onto wall-clock time and decides
whether that moment falls inside an active working window; per-zone ``rules``
are still carried through untyped until the rule-abstraction slice (#82).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pipeline.postprocessing import Detection

Point = tuple[float, float]
# A shift window or break: a half-open [start, end) time-of-day interval.
Interval = tuple[time, time]


class ZoneConfigError(ValueError):
    """Raised when a zones config is missing required fields or malformed.

    A subclass of :class:`ValueError` so callers already catching ``ValueError``
    on bad input keep working, while callers that want to single out zone-config
    problems can catch this specific type.
    """


@dataclass(frozen=True)
class Zone:
    """One named ROI polygon plus its (untyped-for-now) rule set."""

    id: str
    name: str
    polygon: list[Point]
    rules: dict[str, Any] = field(default_factory=dict)

    def contains(self, x: float, y: float) -> bool:
        """Return whether point ``(x, y)`` lies inside this zone's polygon.

        On-boundary points (on an edge or vertex) count as inside — see the
        module docstring for why.
        """
        return _point_in_polygon(x, y, self.polygon)


@dataclass(frozen=True)
class ShiftSchedule:
    """Wall-clock gating for the configured working period (issue #79).

    Built from ``recording_start`` (the wall-clock anchor of ``timestamp_s ==
    0``) plus a ``shift`` block of working ``windows`` and ``breaks``. A frame's
    ``timestamp_s`` (elapsed real seconds into the recording) maps to a
    wall-clock instant, and :meth:`is_active` reports whether that instant falls
    inside a window and outside every break — the only frames the analysis
    counts.

    The mapping is DST-safe: elapsed seconds are added on the absolute UTC
    timeline, then projected back into ``project_tz``. With an IANA
    ``shift.timezone`` (e.g. ``"Europe/Warsaw"``) a recording that crosses a DST
    transition shifts its wall clock by the hour exactly as real clocks do;
    without one it falls back to ``recording_start``'s fixed offset (correct as
    long as the recording does not cross a transition).

    Windows and breaks are half-open ``[start, end)`` time-of-day intervals that
    recur every day, so a recording spanning a day rollover reuses the same
    schedule.
    """

    start_instant: datetime  # tz-aware absolute instant of timestamp_s == 0
    project_tz: tzinfo  # zone used to project wall-clock (DST-aware if IANA)
    windows: tuple[Interval, ...]
    breaks: tuple[Interval, ...]

    @classmethod
    def from_config(cls, recording_start: str, shift: dict[str, Any]) -> ShiftSchedule:
        """Build a :class:`ShiftSchedule` from the raw config fields.

        Raises:
            ZoneConfigError: on a malformed ``recording_start``, an unknown
                ``shift.timezone``, or a window/break that is not a list of
                ``[start, end]`` ``HH:MM`` pairs with ``start < end``.
        """
        if not isinstance(shift, dict):
            raise ZoneConfigError(f"shift must be a JSON object, got {type(shift).__name__}")
        project_tz = _parse_timezone(shift.get("timezone"))
        start = _parse_recording_start(recording_start, project_tz)
        if project_tz is None:
            project_tz = start.tzinfo
        windows = _parse_intervals(shift.get("windows"), "windows")
        breaks = _parse_intervals(shift.get("breaks"), "breaks")
        return cls(start_instant=start, project_tz=project_tz, windows=windows, breaks=breaks)

    def wall_clock_at(self, timestamp_s: float) -> datetime:
        """Map ``timestamp_s`` (elapsed real seconds) to a wall-clock datetime.

        DST-safe: the elapsed seconds are added on the UTC timeline (which has
        no offset jumps), then the instant is projected into ``project_tz``.
        """
        instant = self.start_instant.astimezone(UTC) + timedelta(seconds=timestamp_s)
        return instant.astimezone(self.project_tz)

    def is_active(self, timestamp_s: float) -> bool:
        """Whether the frame at ``timestamp_s`` falls inside an active window.

        Active = inside some working window and outside every break. Interval
        bounds are half-open ``[start, end)``: a frame exactly at a window's
        start counts as working, one exactly at its end does not; a frame at a
        break's start is excluded, one at its end is back to work.
        """
        tod = self.wall_clock_at(timestamp_s).time()
        if not any(start <= tod < end for start, end in self.windows):
            return False
        return not any(start <= tod < end for start, end in self.breaks)

    @property
    def window_labels(self) -> list[tuple[str, str]]:
        """Working windows as ``(start, end)`` ``HH:MM`` label pairs for reports."""
        return [_interval_labels(iv) for iv in self.windows]

    @property
    def break_labels(self) -> list[tuple[str, str]]:
        """Breaks as ``(start, end)`` ``HH:MM`` label pairs for reports."""
        return [_interval_labels(iv) for iv in self.breaks]


@dataclass
class ZoneConfig:
    """Loaded zones config: the zone list plus opaque shift metadata."""

    zones: list[Zone]
    recording_start: str | None = None
    shift: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZoneConfig:
        """Build a :class:`ZoneConfig` from a parsed JSON dict.

        Raises:
            ZoneConfigError: if ``zones`` is absent/not a list, or any zone is
                missing ``id``/``name`` or carries a degenerate polygon.
        """
        if not isinstance(data, dict):
            raise ZoneConfigError(f"zones config must be a JSON object, got {type(data).__name__}")
        raw_zones = data.get("zones")
        if not isinstance(raw_zones, list):
            raise ZoneConfigError("zones config must have a 'zones' list")

        zones = [_parse_zone(raw) for raw in raw_zones]
        config = cls(
            zones=zones,
            recording_start=data.get("recording_start"),
            shift=data.get("shift"),
        )
        # Build the shift schedule once so a malformed shift block fails at
        # load time, not deep inside aggregation (issue #79).
        _ = config.shift_schedule
        return config

    @classmethod
    def load(cls, path: str | Path) -> ZoneConfig:
        """Read and parse a ``zones.json`` file from disk."""
        text = Path(path).read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ZoneConfigError(f"zones config at {path} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def zone_for_point(self, x: float, y: float) -> str | None:
        """Return the id of the first zone containing ``(x, y)``, else ``None``."""
        for zone in self.zones:
            if zone.contains(x, y):
                return zone.id
        return None

    def zone_for_detection(self, det: Detection) -> str | None:
        """Return the zone id for ``det``'s foot point, or ``None`` if outside all."""
        fx, fy = foot_point(det)
        return self.zone_for_point(fx, fy)

    @property
    def shift_schedule(self) -> ShiftSchedule | None:
        """The parsed shift-gating schedule, or ``None`` when no shift is set.

        Raises:
            ZoneConfigError: if a ``shift`` block is present without a
                ``recording_start`` anchor, or the shift block is malformed.
        """
        if self.shift is None:
            return None
        if self.recording_start is None:
            raise ZoneConfigError("a 'shift' schedule requires a 'recording_start' anchor")
        return ShiftSchedule.from_config(self.recording_start, self.shift)


def foot_point(det: Detection) -> Point:
    """Midpoint of a detection's bbox bottom edge — where the person stands."""
    x1, _y1, x2, y2 = det.bbox
    return ((x1 + x2) / 2.0, y2)


def _parse_zone(raw: Any) -> Zone:
    if not isinstance(raw, dict):
        raise ZoneConfigError(f"each zone must be a JSON object, got {type(raw).__name__}")
    zone_id = raw.get("id")
    if not isinstance(zone_id, str) or not zone_id:
        raise ZoneConfigError("each zone must have a non-empty string 'id'")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ZoneConfigError(f"zone {zone_id!r} must have a non-empty string 'name'")
    polygon = _parse_polygon(raw.get("polygon"), zone_id)
    rules = raw.get("rules") or {}
    if not isinstance(rules, dict):
        raise ZoneConfigError(f"zone {zone_id!r} 'rules' must be an object")
    return Zone(id=zone_id, name=name, polygon=polygon, rules=rules)


def _parse_polygon(raw: Any, zone_id: str) -> list[Point]:
    if not isinstance(raw, list) or len(raw) < 3:
        raise ZoneConfigError(f"zone {zone_id!r} polygon must have at least 3 points")
    points: list[Point] = []
    for pt in raw:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ZoneConfigError(f"zone {zone_id!r} polygon points must be [x, y] pairs")
        x, y = pt
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ZoneConfigError(f"zone {zone_id!r} polygon coordinates must be numbers")
        points.append((float(x), float(y)))
    return points


def _parse_timezone(raw: Any) -> tzinfo | None:
    """Resolve an optional IANA ``shift.timezone`` name to a tzinfo, or None."""
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ZoneConfigError("shift 'timezone' must be a non-empty IANA name string")
    try:
        return ZoneInfo(raw)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ZoneConfigError(f"shift 'timezone' {raw!r} is not a known IANA zone") from exc


def _parse_recording_start(raw: Any, project_tz: tzinfo | None) -> datetime:
    """Parse ``recording_start`` into a tz-aware instant.

    A naive timestamp is localized into ``project_tz`` when one is supplied;
    otherwise it is rejected — the anchor must be unambiguous in absolute time.
    """
    if not isinstance(raw, str):
        raise ZoneConfigError("recording_start must be an ISO 8601 timestamp string")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ZoneConfigError(f"recording_start {raw!r} is not a valid ISO 8601 timestamp") from exc
    if dt.tzinfo is None:
        if project_tz is None:
            raise ZoneConfigError(
                "recording_start needs a UTC offset or a shift.timezone to anchor it in time"
            )
        dt = dt.replace(tzinfo=project_tz)
    return dt


def _parse_intervals(raw: Any, field_name: str) -> tuple[Interval, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ZoneConfigError(f"shift {field_name!r} must be a list of [start, end] pairs")
    intervals: list[Interval] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ZoneConfigError(f"shift {field_name!r} entries must be [start, end] pairs")
        start = _parse_hhmm(pair[0], field_name)
        end = _parse_hhmm(pair[1], field_name)
        if not start < end:
            raise ZoneConfigError(
                f"shift {field_name!r} interval {list(pair)} must have start earlier than end"
            )
        intervals.append((start, end))
    return tuple(intervals)


def _interval_labels(interval: Interval) -> tuple[str, str]:
    start, end = interval
    return (start.strftime("%H:%M"), end.strftime("%H:%M"))


def _parse_hhmm(raw: Any, field_name: str) -> time:
    if not isinstance(raw, str):
        raise ZoneConfigError(f"shift {field_name!r} times must be 'HH:MM' strings")
    try:
        return datetime.strptime(raw, "%H:%M").time()
    except ValueError as exc:
        raise ZoneConfigError(
            f"shift {field_name!r} time {raw!r} is not a valid 'HH:MM' time"
        ) from exc


def _on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> bool:
    """Whether point ``P`` lies on segment ``AB`` (collinear and within bounds)."""
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > 1e-9:
        return False
    return min(ax, bx) - 1e-9 <= px <= max(ax, bx) + 1e-9 and (
        min(ay, by) - 1e-9 <= py <= max(ay, by) + 1e-9
    )


def _point_in_polygon(x: float, y: float, polygon: list[Point]) -> bool:
    """Ray-casting point-in-polygon with on-boundary treated as inside.

    Boundary points are checked explicitly first (ray casting is ambiguous on
    edges/vertices), then the standard even-odd crossing test decides interior
    points.
    """
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        if _on_segment(x, y, ax, ay, bx, by):
            return True

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside
