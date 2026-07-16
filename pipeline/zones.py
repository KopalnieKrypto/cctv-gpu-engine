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
``shift`` are parsed and retained verbatim for the shift-gating slice (#79) but
are otherwise opaque; per-zone ``rules`` are likewise carried through untyped
until the rule-abstraction slice (#82).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.postprocessing import Detection

Point = tuple[float, float]


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
        return cls(
            zones=zones,
            recording_start=data.get("recording_start"),
            shift=data.get("shift"),
        )

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
