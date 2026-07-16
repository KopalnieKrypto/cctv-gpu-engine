"""Zone conversation-mode detection (issue #81, slice 3).

Two people standing together in a zone — ≥2 stable tracks whose foot points are
mutually close and who are both barely moving — read as a *conversation*, the
third zone mode alongside ``work`` and ``absent`` (#80). Where
:class:`~pipeline.presence.ZonePresenceAnalyzer` anchors on the single worker who
staffs the station, conversation is inherently *multi-track*: it asks whether any
two proximate tracks are both idle at the same moment, so the anchored worker
chatting with a visitor and two workers pausing together both surface the same.

:class:`ConversationAnalyzer` consumes the same ``{track_id: foot_point}`` sighting
map per analysed frame the presence analyzer does. Movement is judged per
consecutive sighting pair (a track present in both frames whose foot point shifts
no more than ``min_move_px`` is *stationary* — reusing the ``work`` threshold, so a
track moving enough to be "working" can never also be "conversing"); a frame pair
is a conversation segment when two stationary tracks sit within ``proximity_px``,
and adjacent segments merge into one interval. A moving passer-by is ignored, and
a detection-dropout gap wider than the sampling tolerance never bridges two
intervals.

Like the presence analyzer, this is pure geometry/temporal logic — it knows
nothing about ONNX, numpy, or the pipeline. Callers (the aggregator) resolve which
detections fall in the zone and hand over ``{track_id: foot_point}`` per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from pipeline.presence import Interval

Point = tuple[float, float]

# A run of conversation segments stays unbroken while consecutive frame gaps are
# within this multiple of the sampling step — mirrors PRESENCE_MERGE_FACTOR so a
# detection dropout (gap ≈ 2× the step) opens a break rather than being bridged.
CONVERSATION_MERGE_FACTOR = 1.5

# Maximum foot-point distance (pixels) between two tracks for them to count as
# "standing together". The client gave no bending-specific value, so this is the
# default; per-zone config (``rules.conversation.proximity_px``) overrides it.
DEFAULT_PROXIMITY_PX = 150.0

# Reused from presence: a track whose foot point moved more than this between
# samples is *moving* (working), so it cannot be conversing. Per-zone config
# (``rules.work.min_move_px``) overrides it — one movement threshold per zone.
DEFAULT_MIN_MOVE_PX = 40.0


@dataclass(frozen=True)
class ZoneConversation:
    """Derived conversation intervals for one zone."""

    intervals: tuple[Interval, ...] = ()

    @property
    def conversation_s(self) -> float:
        """Total seconds two idle, proximate tracks stood together in the zone."""
        return sum(iv.duration_s for iv in self.intervals)


class ConversationAnalyzer:
    """Accumulates per-frame zone sightings into a :class:`ZoneConversation`."""

    def __init__(
        self,
        sampling_step_s: float = 1.0,
        proximity_px: float = DEFAULT_PROXIMITY_PX,
        min_move_px: float = DEFAULT_MIN_MOVE_PX,
    ) -> None:
        self._merge_gap_s = sampling_step_s * CONVERSATION_MERGE_FACTOR
        self._proximity_px = proximity_px
        self._min_move_px = min_move_px
        # Ordered (timestamp_s, {track_id: foot_point}) — a full per-frame map,
        # because conversation is a relation between tracks, not a per-track fact.
        self._frames: list[tuple[float, dict[int, Point]]] = []

    def observe(self, timestamp_s: float, sightings: dict[int, Point]) -> None:
        """Record which tracks stood in the zone at ``timestamp_s`` and where."""
        self._frames.append((timestamp_s, dict(sightings)))

    def result(self) -> ZoneConversation:
        """Resolve conversation segments and merge adjacent ones into intervals."""
        # Sort by timestamp only — the sighting dicts are not orderable, so a key
        # avoids a TypeError should two frames ever share a timestamp.
        frames = sorted(self._frames, key=lambda frame: frame[0])
        segments: list[tuple[float, float]] = []
        for (t_prev, prev), (t_cur, cur) in zip(frames, frames[1:], strict=False):
            if t_cur - t_prev > self._merge_gap_s:
                continue  # detection dropout — no movement judgement across it
            if self._is_conversation(prev, cur):
                segments.append((t_prev, t_cur))
        return ZoneConversation(intervals=self._merge(segments))

    def _is_conversation(self, prev: dict[int, Point], cur: dict[int, Point]) -> bool:
        """Whether two stationary tracks stood within proximity across this pair."""
        stationary = [
            tid
            for tid, foot in cur.items()
            if tid in prev and math.dist(prev[tid], foot) <= self._min_move_px
        ]
        return any(
            math.dist(cur[a], cur[b]) <= self._proximity_px for a, b in combinations(stationary, 2)
        )

    def _merge(self, segments: list[tuple[float, float]]) -> tuple[Interval, ...]:
        """Fuse touching frame-pair segments; a non-conversation pair splits them."""
        merged: list[list[float]] = []
        for start, end in segments:
            if merged and start <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return tuple(Interval(start, end) for start, end in merged)
