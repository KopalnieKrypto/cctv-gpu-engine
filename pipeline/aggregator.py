"""Per-frame aggregation into a video-level ReportData.

The :class:`Aggregator` is the central accumulator for full-video analysis
(issue #4). It receives one ``(timestamp_s, frame, detections)`` tuple per
processed frame and tracks just enough state to build a complete
:class:`ReportData` at the end:

* total / peak / average person counts
* per-activity person-frame counters → person-minutes
* 1-minute timeline bins (per-activity)
* a bounded buffer of "best" candidate keyframes for the report

The aggregator never retains more than ``MAX_KEYFRAME_CANDIDATES`` raw frames
in memory (evicting the lowest-value unprotected candidate on overflow), so RSS
stays flat regardless of video length (issue #49).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.postprocessing import Detection
from pipeline.presence import (
    DEFAULT_FLAG_AFTER_S,
    DEFAULT_MIN_MOVE_PX,
    ZonePresence,
    ZonePresenceAnalyzer,
)
from pipeline.zones import ShiftSchedule, Zone, foot_point

ACTIVITIES = ("sitting", "standing", "walking", "running")

# Hard cap on the number of raw candidate frames retained for keyframe
# selection. The selector only ever needs ~keyframe_count (8) frames; 64 gives
# ample slack for per-activity coverage + min-spacing fill while keeping RSS
# flat regardless of video length (issue #49). At 1080p BGR (~5.9 MiB/frame)
# this bounds the candidate buffer at ~380 MiB instead of unbounded ~21 GiB/h.
MAX_KEYFRAME_CANDIDATES = 64


@dataclass
class TimelineBin:
    """Per-minute counts of how many person-detections fell in this minute."""

    minute: int
    sitting: int = 0
    standing: int = 0
    walking: int = 0
    running: int = 0


@dataclass
class Keyframe:
    """One annotated keyframe selected for the report."""

    timestamp_s: float
    person_count: int
    frame: np.ndarray  # raw BGR frame; annotation happens in the renderer
    detections: list[Detection] = field(default_factory=list)


@dataclass
class ZoneReport:
    """Per-zone posture breakdown (issue #78).

    A named ROI's slice of the video: person-minutes per activity for the
    detections whose foot point fell inside this zone. Slice 0 carries the
    posture breakdown only; semantic modes / timelines arrive in later slices.
    """

    zone_id: str
    name: str
    person_minutes: dict[str, float]
    # Anchored-worker presence/absence + work (issue #80). ``None`` before slice
    # 2 wired it or when no zone config is active; otherwise a ZonePresence whose
    # ``anchored_track_id`` is ``None`` if no track ever dwelt here (e.g. tracking
    # disabled).
    presence: ZonePresence | None = None


@dataclass
class ShiftSummary:
    """Shift-window gating summary for the report (issue #79).

    ``windows``/``breaks`` are the analysed working windows and the excluded
    breaks as ``(start, end)`` ``HH:MM`` label pairs; ``excluded_duration_s`` is
    the total footage, in seconds, dropped because it fell outside an active
    window. Present only when a shift schedule gated the run.
    """

    windows: list[tuple[str, str]]
    breaks: list[tuple[str, str]]
    excluded_duration_s: float


@dataclass
class ReportData:
    """The full data model passed to :func:`pipeline.report_renderer.render_report`."""

    video_duration_s: float
    total_frames: int
    peak_persons: int
    avg_persons: float
    dominant_activity: str
    person_minutes: dict[str, float]
    timeline: list[TimelineBin]
    keyframes: list[Keyframe]
    # Per-zone posture breakdown (issue #78); empty when no zones config is
    # active. One entry per configured zone, in config order.
    zones: list[ZoneReport] = field(default_factory=list)
    # Shift-window gating summary (issue #79); ``None`` when no shift schedule
    # gated the run. Carries the analysed windows and total excluded duration.
    shift: ShiftSummary | None = None


def _presence_analyzer_for(zone: Zone, sampling_step_s: float) -> ZonePresenceAnalyzer:
    """Build a zone's presence analyzer, honouring its ``rules`` config (#80).

    ``rules.absence.flag_after_s`` and ``rules.work.min_move_px`` override the
    module defaults; a missing or non-object rule section falls back cleanly so a
    zone with no explicit rules still gets sensible presence analysis.
    """

    def _section(name: str) -> dict:
        section = zone.rules.get(name)
        return section if isinstance(section, dict) else {}

    return ZonePresenceAnalyzer(
        sampling_step_s=sampling_step_s,
        flag_after_s=_section("absence").get("flag_after_s", DEFAULT_FLAG_AFTER_S),
        min_move_px=_section("work").get("min_move_px", DEFAULT_MIN_MOVE_PX),
    )


class Aggregator:
    """Accumulates per-frame detections into a final :class:`ReportData`."""

    def __init__(
        self,
        fps: int = 1,
        keyframe_count: int = 8,
        keyframe_min_spacing_s: float = 30.0,
        zones: list[Zone] | None = None,
        shift: ShiftSchedule | None = None,
    ) -> None:
        self.fps = fps
        self.keyframe_count = keyframe_count
        self.keyframe_min_spacing_s = keyframe_min_spacing_s
        # Shift-window gate (issue #79). When set, frames whose wall-clock falls
        # outside an active window are dropped from every analysis counter and
        # only tallied into ``_excluded_frames`` for the report's summary.
        self._shift = shift
        self._excluded_frames = 0
        self._total_frames = 0
        self._person_count_sum = 0
        self._peak_persons = 0
        self._activity_person_frames: dict[str, int] = dict.fromkeys(ACTIVITIES, 0)
        # Per-zone posture accumulation (issue #78). ``_zones`` preserves config
        # order and id→name; ``_zone_person_frames`` mirrors the global
        # per-activity counter but keyed by zone id. Detections with a
        # ``zone_id`` outside this set (or None) accrue only to the global
        # totals, never to a zone.
        self._zones: list[Zone] = list(zones) if zones else []
        self._zone_person_frames: dict[str, dict[str, int]] = {
            zone.id: dict.fromkeys(ACTIVITIES, 0) for zone in self._zones
        }
        # Per-zone anchored-worker presence (issue #80). Each analyzer is fed the
        # in-zone {track_id: foot_point} of every shift-active frame; its rules
        # come from the zone config, falling back to module defaults.
        sampling_step_s = 1.0 / fps
        self._zone_presence: dict[str, ZonePresenceAnalyzer] = {
            zone.id: _presence_analyzer_for(zone, sampling_step_s) for zone in self._zones
        }
        self._last_timestamp_s = 0.0
        self._bins: dict[int, TimelineBin] = {}
        # Frames with ≥1 person are kept as keyframe candidates, capped at
        # MAX_KEYFRAME_CANDIDATES (issue #49). ``_activity_best`` pins the top
        # candidate of each observed activity so eviction can never drop the
        # sole evidence of a rare activity — per-activity report coverage
        # survives bounding. Its values are always objects still in
        # ``_candidates`` (protected frames are never evicted).
        self._candidates: list[Keyframe] = []
        self._activity_best: dict[str, Keyframe] = {}

    def add_frame(
        self,
        timestamp_s: float,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> None:
        """Record one processed frame and its per-person detections."""
        # video_duration_s reflects the whole recording, so advance it even for
        # frames the shift gate is about to drop (issue #79).
        if timestamp_s > self._last_timestamp_s:
            self._last_timestamp_s = timestamp_s
        # Shift gating: a frame whose wall-clock is outside every active window
        # (or inside a break) contributes to nothing but the excluded tally.
        if self._shift is not None and not self._shift.is_active(timestamp_s):
            self._excluded_frames += 1
            return
        self._total_frames += 1
        person_count = len(detections)
        self._person_count_sum += person_count
        if person_count > self._peak_persons:
            self._peak_persons = person_count
        minute = int(timestamp_s // 60)
        bin_ = self._bins.get(minute)
        if bin_ is None:
            bin_ = TimelineBin(minute=minute)
            self._bins[minute] = bin_
        for det in detections:
            if det.activity in self._activity_person_frames:
                self._activity_person_frames[det.activity] += 1
                setattr(bin_, det.activity, getattr(bin_, det.activity) + 1)
                zone_counter = self._zone_person_frames.get(det.zone_id)
                if zone_counter is not None:
                    zone_counter[det.activity] += 1
        self._observe_zone_presence(timestamp_s, detections)
        if person_count > 0:
            # Keep a copy of the frame because the caller will overwrite the
            # buffer on the next ffmpeg read.
            candidate = Keyframe(
                timestamp_s=timestamp_s,
                person_count=person_count,
                frame=frame.copy(),
                detections=list(detections),
            )
            self._candidates.append(candidate)
            self._update_activity_best(candidate)
            self._evict_if_over_capacity()

    def _observe_zone_presence(self, timestamp_s: float, detections: list[Detection]) -> None:
        """Feed this (shift-active) frame's in-zone foot points to each analyzer.

        Only detections carrying both a ``zone_id`` for a configured zone and a
        ``track_id`` count — a foot point without an identity can't be anchored
        (issue #80). Runs after the shift gate, so excluded frames never register
        as presence.
        """
        if not self._zone_presence:
            return
        per_zone: dict[str, dict[int, tuple[float, float]]] = {}
        for det in detections:
            if det.track_id is None or det.zone_id not in self._zone_presence:
                continue
            per_zone.setdefault(det.zone_id, {})[det.track_id] = foot_point(det)
        for zone_id, sightings in per_zone.items():
            self._zone_presence[zone_id].observe(timestamp_s, sightings)

    @property
    def candidate_count(self) -> int:
        """Number of raw frames currently retained (bounded, issue #49)."""
        return len(self._candidates)

    def _update_activity_best(self, candidate: Keyframe) -> None:
        """Pin ``candidate`` as an activity's best if it outranks the incumbent.

        Ranking matches ``_select_keyframes`` pass 1: highest ``person_count``,
        ties broken by earliest timestamp. A frame is a candidate for every
        activity present among its detections.
        """
        key = (candidate.person_count, -candidate.timestamp_s)
        for activity in {d.activity for d in candidate.detections if d.activity in ACTIVITIES}:
            incumbent = self._activity_best.get(activity)
            if incumbent is None or key > (incumbent.person_count, -incumbent.timestamp_s):
                self._activity_best[activity] = candidate

    def _evict_if_over_capacity(self) -> None:
        """Drop the lowest-value *unprotected* candidate over the cap.

        Per-activity bests are protected so bounding never erases the sole
        evidence of a rare activity. Protected frames number at most
        ``len(ACTIVITIES)`` (4) ≪ cap, so an evictable candidate always exists.
        """
        if len(self._candidates) <= MAX_KEYFRAME_CANDIDATES:
            return
        protected = {id(k) for k in self._activity_best.values()}
        evictable = [k for k in self._candidates if id(k) not in protected]
        if not evictable:
            return
        worst = min(evictable, key=lambda k: (k.person_count, k.timestamp_s))
        self._candidates = [k for k in self._candidates if k is not worst]

    def _select_keyframes(self) -> list[Keyframe]:
        """Two-pass selection: one keyframe per detected activity, then fill.

        Pass 1 (per-activity coverage): for each activity that appeared in
        the video, pick the candidate with the highest ``person_count`` (ties
        broken by earliest timestamp). Different activities ⇒ different
        scenes, so the spacing constraint does NOT apply between activities
        — we want every observed activity represented in the report.

        Pass 2 (fill): top up to ``keyframe_count`` from the highest
        ``person_count`` remaining candidates, this time enforcing
        ``keyframe_min_spacing_s`` against everything already selected so the
        report doesn't show 5 nearly-identical frames.

        Final list is sorted chronologically so the renderer gives a
        narrative tour of the video.
        """
        if not self._candidates:
            return []

        selected: list[Keyframe] = []
        chosen_ids: set[int] = set()

        # Pass 1 — at least one keyframe per activity that occurred.
        for activity in ACTIVITIES:
            cands = [
                c for c in self._candidates if any(d.activity == activity for d in c.detections)
            ]
            if not cands:
                continue
            best = max(cands, key=lambda k: (k.person_count, -k.timestamp_s))
            if id(best) not in chosen_ids:
                selected.append(best)
                chosen_ids.add(id(best))

        # Pass 2 — fill remaining slots, this time enforcing min-spacing.
        ranked = sorted(self._candidates, key=lambda k: k.person_count, reverse=True)
        for cand in ranked:
            if len(selected) >= self.keyframe_count:
                break
            if id(cand) in chosen_ids:
                continue
            if all(
                abs(cand.timestamp_s - k.timestamp_s) >= self.keyframe_min_spacing_s
                for k in selected
            ):
                selected.append(cand)
                chosen_ids.add(id(cand))

        selected.sort(key=lambda k: k.timestamp_s)
        return selected

    def build_report_data(self) -> ReportData:
        person_minutes = {
            activity: self._activity_person_frames[activity] / self.fps / 60.0
            for activity in ACTIVITIES
        }

        if self._total_frames == 0:
            avg_persons = 0.0
            dominant = "none"
        else:
            avg_persons = self._person_count_sum / self._total_frames
            dominant = max(self._activity_person_frames.items(), key=lambda kv: kv[1])[0]
            if self._activity_person_frames[dominant] == 0:
                dominant = "none"

        zones = [
            ZoneReport(
                zone_id=zone.id,
                name=zone.name,
                person_minutes={
                    activity: self._zone_person_frames[zone.id][activity] / self.fps / 60.0
                    for activity in ACTIVITIES
                },
                presence=self._zone_presence[zone.id].result(),
            )
            for zone in self._zones
        ]

        shift = None
        if self._shift is not None:
            shift = ShiftSummary(
                windows=self._shift.window_labels,
                breaks=self._shift.break_labels,
                excluded_duration_s=self._excluded_frames / self.fps,
            )

        return ReportData(
            video_duration_s=self._last_timestamp_s,
            total_frames=self._total_frames,
            peak_persons=self._peak_persons,
            avg_persons=avg_persons,
            dominant_activity=dominant,
            person_minutes=person_minutes,
            timeline=[self._bins[m] for m in sorted(self._bins)],
            keyframes=self._select_keyframes(),
            zones=zones,
            shift=shift,
        )
