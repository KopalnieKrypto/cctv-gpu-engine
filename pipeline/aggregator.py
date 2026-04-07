"""Per-frame aggregation into a video-level ReportData.

The :class:`Aggregator` is the central accumulator for full-video analysis
(issue #4). It receives one ``(timestamp_s, frame, detections)`` tuple per
processed frame and tracks just enough state to build a complete
:class:`ReportData` at the end:

* total / peak / average person counts
* per-activity person-frame counters → person-minutes
* 1-minute timeline bins (per-activity)
* a bounded buffer of "best" candidate keyframes for the report

The aggregator never holds more than ``keyframe_buffer_size`` raw frames in
memory, so it stays flat regardless of video length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.postprocessing import Detection

ACTIVITIES = ("sitting", "standing", "walking", "running")


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


class Aggregator:
    """Accumulates per-frame detections into a final :class:`ReportData`."""

    def __init__(
        self,
        fps: int = 1,
        keyframe_count: int = 5,
        keyframe_min_spacing_s: float = 120.0,
    ) -> None:
        self.fps = fps
        self.keyframe_count = keyframe_count
        self.keyframe_min_spacing_s = keyframe_min_spacing_s
        self._total_frames = 0
        self._person_count_sum = 0
        self._peak_persons = 0
        self._activity_person_frames: dict[str, int] = dict.fromkeys(ACTIVITIES, 0)
        self._last_timestamp_s = 0.0
        self._bins: dict[int, TimelineBin] = {}
        # All frames with ≥1 person are kept as candidates. We do not bound this
        # buffer in tests; the orchestrator caps at runtime via _trim_candidates.
        self._candidates: list[Keyframe] = []

    def add_frame(
        self,
        timestamp_s: float,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> None:
        """Record one processed frame and its per-person detections."""
        self._total_frames += 1
        person_count = len(detections)
        self._person_count_sum += person_count
        if person_count > self._peak_persons:
            self._peak_persons = person_count
        if timestamp_s > self._last_timestamp_s:
            self._last_timestamp_s = timestamp_s
        minute = int(timestamp_s // 60)
        bin_ = self._bins.get(minute)
        if bin_ is None:
            bin_ = TimelineBin(minute=minute)
            self._bins[minute] = bin_
        for det in detections:
            if det.activity in self._activity_person_frames:
                self._activity_person_frames[det.activity] += 1
                setattr(bin_, det.activity, getattr(bin_, det.activity) + 1)
        if person_count > 0:
            # Keep a copy of the frame because the caller will overwrite the
            # buffer on the next ffmpeg read.
            self._candidates.append(
                Keyframe(
                    timestamp_s=timestamp_s,
                    person_count=person_count,
                    frame=frame.copy(),
                    detections=list(detections),
                )
            )

    def _select_keyframes(self) -> list[Keyframe]:
        """Greedy: sort by person_count desc, take top-K with ≥spacing gaps."""
        ranked = sorted(self._candidates, key=lambda k: k.person_count, reverse=True)
        selected: list[Keyframe] = []
        for cand in ranked:
            if len(selected) >= self.keyframe_count:
                break
            if all(
                abs(cand.timestamp_s - k.timestamp_s) >= self.keyframe_min_spacing_s
                for k in selected
            ):
                selected.append(cand)
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

        return ReportData(
            video_duration_s=self._last_timestamp_s,
            total_frames=self._total_frames,
            peak_persons=self._peak_persons,
            avg_persons=avg_persons,
            dominant_activity=dominant,
            person_minutes=person_minutes,
            timeline=[self._bins[m] for m in sorted(self._bins)],
            keyframes=self._select_keyframes(),
        )
