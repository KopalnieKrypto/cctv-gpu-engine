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

``recording_start`` and ``shift`` are parsed into a :class:`ShiftSchedule` for
the shift-gating slice (#79) that maps a frame's ``timestamp_s`` onto wall-clock
time and decides whether that moment falls inside an active working window.

Per-zone ``rules`` are dispatched on ``rules.type`` (#82): the type selects a
*ruleset* — the set of semantic modes derived for the zone. ``bending`` (slices
2–3: presence/work + conversation) is the only implemented type and the default
when ``type`` is omitted, so no pre-#82 config changes meaning; an unknown type
is rejected at load time with :class:`UnsupportedRuleTypeError`. A future welding
station registers its own ruleset in ``_RULESETS`` without touching the bending
analyzers. :func:`build_zone_ruleset` is the dispatch the aggregator routes
through.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pipeline.conversation import (
    DEFAULT_PROXIMITY_PX,
    ConversationAnalyzer,
    ZoneConversation,
)
from pipeline.postprocessing import Detection
from pipeline.presence import (
    DEFAULT_FLAG_AFTER_S,
    DEFAULT_MIN_MOVE_PX,
    ZonePresence,
    ZonePresenceAnalyzer,
)

Point = tuple[float, float]
# One analysed frame's in-zone sightings: which tracks stood in a zone and where
# their foot points were. The unit both the presence and conversation analyzers
# consume, so the rule dispatch (#82) passes it straight through.
Sightings = dict[int, Point]
# A shift window or break: a half-open [start, end) time-of-day interval.
Interval = tuple[time, time]

# Per-zone rule dispatch (#82). A zone's ``rules.type`` selects its ruleset — the
# set of semantic modes derived for it. ``bending`` (slices 2–3: presence/work +
# conversation) is the only implemented type; a future ``welding`` station adds
# its own type to the registry below. A zone that omits ``type`` defaults to
# ``bending`` so no pre-#82 config changes meaning.
DEFAULT_RULE_TYPE = "bending"


class ZoneConfigError(ValueError):
    """Raised when a zones config is missing required fields or malformed.

    A subclass of :class:`ValueError` so callers already catching ``ValueError``
    on bad input keep working, while callers that want to single out zone-config
    problems can catch this specific type.
    """


class UnsupportedRuleTypeError(ZoneConfigError):
    """Raised when a zone's ``rules.type`` names an unimplemented ruleset (#82).

    A subclass of :class:`ZoneConfigError` (so callers already catching bad-config
    errors keep working) that the rule dispatch raises for an unknown type — e.g.
    a ``welding`` zone before that ruleset ships. Its message lists the supported
    types so the operator sees what they should have written.
    """


@dataclass(frozen=True)
class Zone:
    """One named ROI polygon plus its rule set (dispatched on ``rules.type``, #82)."""

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
class InferenceROI:
    """One semantic zone whose bounding box focuses pose inference (#86)."""

    zone_id: str
    margin_px: float


@dataclass
class ZoneConfig:
    """Loaded zones config: the zone list plus opaque shift metadata."""

    zones: list[Zone]
    recording_start: str | None = None
    shift: dict[str, Any] | None = None
    inference_roi: InferenceROI | None = None
    # Mask the headline tallies to the polygons (#96). The platform's opt-in
    # (gpu-exchange#162): off (or absent), zones only ADD a per-zone breakdown
    # and the whole-frame totals still count everyone; on, only in-zone
    # detections reach the global counters. Absent means off, so every pre-#96
    # config keeps its whole-frame numbers.
    restrict_to_zones: bool = False

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
        inference_roi = _parse_inference_roi(data.get("inference_roi"), zones)
        config = cls(
            zones=zones,
            recording_start=data.get("recording_start"),
            shift=data.get("shift"),
            inference_roi=inference_roi,
            restrict_to_zones=data.get("restrict_to_zones") is True,
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

    def inference_bounds(
        self, frame_width: int, frame_height: int
    ) -> tuple[int, int, int, int] | None:
        """Return the clipped inference crop as ``(x1, y1, x2, y2)``."""
        if self.inference_roi is None:
            return None
        zone = next(zone for zone in self.zones if zone.id == self.inference_roi.zone_id)
        xs = [point[0] for point in zone.polygon]
        ys = [point[1] for point in zone.polygon]
        margin = self.inference_roi.margin_px
        bounds = (
            math.floor(max(0.0, min(xs) - margin)),
            math.floor(max(0.0, min(ys) - margin)),
            math.ceil(min(float(frame_width), max(xs) + margin)),
            math.ceil(min(float(frame_height), max(ys) + margin)),
        )
        x1, y1, x2, y2 = bounds
        if x1 >= x2 or y1 >= y2:
            raise ZoneConfigError(
                f"inference_roi zone {self.inference_roi.zone_id!r} has no pixels "
                f"inside the frame ({frame_width}x{frame_height})"
            )
        return bounds

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


def _rule_section(rules: dict[str, Any], name: str) -> dict:
    """A named sub-object of ``rules`` (e.g. ``work``), or ``{}`` if absent/malformed."""
    section = rules.get(name)
    return section if isinstance(section, dict) else {}


@dataclass(frozen=True)
class ZoneModes:
    """The semantic modes a ruleset derived for one zone, for the report (#82).

    The bending ruleset fills ``presence`` (slice 2) and ``conversation`` (slice
    3); a future welding ruleset would populate its own mode fields here without
    touching the bending analyzers.
    """

    presence: ZonePresence | None = None
    conversation: ZoneConversation | None = None


class BendingRuleset:
    """Bending-station ruleset (#78–#81): presence/absence + work + conversation.

    Owns the presence and conversation analyzers a bending zone derives, built
    from the zone's ``rules``: ``absence.flag_after_s`` flags long absences,
    ``work.min_move_px`` is the moving/idle threshold (shared with conversation,
    so a track moving enough to be *working* can never simultaneously *converse*),
    and ``conversation.proximity_px`` is how close two tracks stand to read as
    together. Fed one :data:`Sightings` map per shift-active frame; :meth:`result`
    yields the zone's derived modes. A missing rule section falls back to the
    module default, so a zone with no explicit rules still gets sensible analysis.
    """

    type = "bending"

    def __init__(self, rules: dict[str, Any], sampling_step_s: float) -> None:
        min_move_px = _rule_section(rules, "work").get("min_move_px", DEFAULT_MIN_MOVE_PX)
        self._presence = ZonePresenceAnalyzer(
            sampling_step_s=sampling_step_s,
            flag_after_s=_rule_section(rules, "absence").get("flag_after_s", DEFAULT_FLAG_AFTER_S),
            min_move_px=min_move_px,
        )
        self._conversation = ConversationAnalyzer(
            sampling_step_s=sampling_step_s,
            proximity_px=_rule_section(rules, "conversation").get(
                "proximity_px", DEFAULT_PROXIMITY_PX
            ),
            min_move_px=min_move_px,
        )

    def observe(self, timestamp_s: float, sightings: Sightings) -> None:
        """Feed one frame's in-zone ``{track_id: foot_point}`` to each mode analyzer."""
        self._presence.observe(timestamp_s, sightings)
        self._conversation.observe(timestamp_s, sightings)

    def result(self) -> ZoneModes:
        """Resolve the accumulated sightings into this zone's derived modes."""
        return ZoneModes(
            presence=self._presence.result(),
            conversation=self._conversation.result(),
        )


# The rule registry: ``rules.type`` → ruleset class. Adding a welding station is a
# new entry here plus its own ruleset — the bending analyzers stay untouched.
_RULESETS: dict[str, type[BendingRuleset]] = {BendingRuleset.type: BendingRuleset}

# Rule types this build implements, derived from the registry (single source of
# truth) so the constant and the dispatch can never drift apart.
SUPPORTED_RULE_TYPES: tuple[str, ...] = tuple(_RULESETS)


def _resolve_rule_type(rules: dict[str, Any], zone_id: str) -> str:
    """Return the ruleset type for ``rules``, defaulting to ``bending`` (#82).

    A zone that omits ``rules.type`` keeps the pre-#82 behaviour (the bending
    ruleset of slices 2–3), so no existing config changes meaning.

    Raises:
        UnsupportedRuleTypeError: if ``rules.type`` names a type this build does
            not implement; the message lists the supported types.
    """
    rule_type = rules.get("type", DEFAULT_RULE_TYPE)
    if rule_type not in _RULESETS:
        supported = ", ".join(SUPPORTED_RULE_TYPES)
        raise UnsupportedRuleTypeError(
            f"zone {zone_id!r} has unsupported rules.type {rule_type!r}; "
            f"supported types: {supported}"
        )
    return rule_type


def build_zone_ruleset(zone: Zone, sampling_step_s: float) -> BendingRuleset:
    """Instantiate the ruleset selected by ``zone.rules['type']`` (#82).

    The dispatch the aggregator routes through: it resolves the zone's rule type
    (defaulting to ``bending``) and constructs that ruleset from the zone's
    ``rules``. Raises :class:`UnsupportedRuleTypeError` for an unknown type — the
    welding seam. (The return type widens to a shared ruleset protocol once a
    second ruleset lands.)
    """
    rule_type = _resolve_rule_type(zone.rules, zone.id)
    return _RULESETS[rule_type](zone.rules, sampling_step_s)


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
    # Fail fast at load time on an unknown rules.type, rather than untyped deep
    # inside aggregation (#82).
    _resolve_rule_type(rules, zone_id)
    return Zone(id=zone_id, name=name, polygon=polygon, rules=rules)


def _parse_inference_roi(raw: Any, zones: list[Zone]) -> InferenceROI | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ZoneConfigError(f"inference_roi must be a JSON object, got {type(raw).__name__}")
    zone_id = raw.get("zone_id")
    if not isinstance(zone_id, str) or not zone_id:
        raise ZoneConfigError("inference_roi 'zone_id' must be a non-empty string")
    if "margin_px" not in raw:
        raise ZoneConfigError("inference_roi must have an explicit 'margin_px'")
    margin = raw["margin_px"]
    if (
        not isinstance(margin, (int, float))
        or isinstance(margin, bool)
        or not math.isfinite(margin)
        or margin < 0
    ):
        raise ZoneConfigError("inference_roi 'margin_px' must be a finite non-negative number")
    zone = next((zone for zone in zones if zone.id == zone_id), None)
    if zone is None:
        raise ZoneConfigError(f"inference_roi 'zone_id' must name an existing zone: {zone_id!r}")
    doubled_area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                zone.polygon,
                zone.polygon[1:] + zone.polygon[:1],
                strict=True,
            )
        )
    )
    if doubled_area == 0:
        raise ZoneConfigError(f"inference_roi zone {zone_id!r} polygon must have non-zero area")
    return InferenceROI(zone_id=zone_id, margin_px=float(margin))


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
