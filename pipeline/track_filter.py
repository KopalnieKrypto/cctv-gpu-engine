"""Min-track-length filter: only proven tracks reach aggregation (issue #32).

Sits between :mod:`pipeline.tracker` and :mod:`pipeline.aggregator`. A track
must be seen in ``MIN_TRACK_FRAMES`` consecutive frames before any of its
detections are allowed to count; until then its detections are withheld, and
if it never gets there they are dropped for good.

This is the mandated defence against sporadic false-positive detections — the
bench-on-wheels that Film 1 reported as a second person. The advisory (round 8)
ruled out both a person/not-person classifier and a YOLO-confidence cutoff, on
the grounds that temporal persistence gets it "for free": a real person
persists frame after frame, a phantom does not.

Because a track only proves itself on its 3rd consecutive frame, this is
necessarily a **delay line** — frame N's contents cannot be judged until frame
N+2 has been seen. ``push`` therefore returns the frames whose fate is now
settled (usually the one from two steps back, nothing at all for the first
couple of frames), and ``flush`` drains the tail at end of video. Frames are
always emitted, in order, whether or not anyone in them survived: the frame
happened, so it still counts toward duration and frame totals — only the
phantom people are removed.

Detections with ``track_id is None`` — those the tracker refused to give an
identity, i.e. faint boxes matching nothing — never survive.

Memory is bounded by ``min_frames - 1`` retained frames (~12 MiB at 1080p),
independent of video length, so it upholds the same flat-RSS discipline as the
aggregator's bounded keyframe buffer (issue #49).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.postprocessing import Detection

# Consecutive frames a track must survive before it is believed to be a person.
# 3 frames at 1 fps = 3 seconds of continuous presence, which also implies the
# 2 consistent OSNet matches that linked them.
MIN_TRACK_FRAMES = 3


@dataclass
class ConfirmedFrame:
    """One frame whose track membership is settled, ready for aggregation."""

    timestamp_s: float
    frame: np.ndarray
    detections: list[Detection]


@dataclass
class _PendingFrame:
    """A frame held back until its tracks are confirmed or abandoned."""

    timestamp_s: float
    frame: np.ndarray
    detections: list[Detection]


class MinTrackLengthFilter:
    """Withholds detections until their track proves it persists."""

    def __init__(self, min_frames: int = MIN_TRACK_FRAMES) -> None:
        self._min_frames = min_frames
        self._pending: list[_PendingFrame] = []
        self._run_length: dict[int, int] = {}
        self._confirmed: set[int] = set()
        self._previous_ids: set[int] = set()

    def push(
        self,
        timestamp_s: float,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[ConfirmedFrame]:
        """Feed one tracked frame; return any frames whose fate is now settled."""
        self._observe(detections)
        # Copy: the caller reuses its frame buffer on the next ffmpeg read, and
        # we hold this one across reads.
        self._pending.append(_PendingFrame(timestamp_s, frame.copy(), detections))

        released: list[ConfirmedFrame] = []
        while len(self._pending) >= self._min_frames:
            released.append(self._release(self._pending.pop(0)))
        return released

    def flush(self) -> list[ConfirmedFrame]:
        """Drain the tail at end of video, judged on everything seen so far."""
        released = [self._release(pending) for pending in self._pending]
        self._pending = []
        return released

    def _observe(self, detections: list[Detection]) -> None:
        """Update per-track consecutive run lengths from one frame."""
        present = {d.track_id for d in detections if d.track_id is not None}

        for track_id in present:
            if track_id in self._confirmed:
                # Confirmation is permanent: a person who has proven they exist
                # does not have to re-prove it after stepping out of frame.
                continue
            if track_id in self._previous_ids:
                self._run_length[track_id] = self._run_length.get(track_id, 1) + 1
            else:
                self._run_length[track_id] = 1
            if self._run_length[track_id] >= self._min_frames:
                self._confirmed.add(track_id)
                self._run_length.pop(track_id, None)

        # A gap breaks an unproven track's run — it must start counting over.
        for track_id in list(self._run_length):
            if track_id not in present:
                del self._run_length[track_id]

        self._previous_ids = present

    def _release(self, pending: _PendingFrame) -> ConfirmedFrame:
        return ConfirmedFrame(
            timestamp_s=pending.timestamp_s,
            frame=pending.frame,
            # `None in self._confirmed` is False, so untracked detections drop
            # out here too.
            detections=[d for d in pending.detections if d.track_id in self._confirmed],
        )
