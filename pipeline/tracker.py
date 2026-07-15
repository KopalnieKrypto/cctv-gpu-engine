"""Within-video person tracking by appearance Re-ID (issue #32).

Runs between :mod:`pipeline.pose_detector` and the activity classifier: takes
the per-frame Detection list and stamps each detection with a stable
``track_id``, so downstream aggregation can count *people* instead of
*person-frames*.

Association is by OSNet appearance similarity rather than IoU: at 1 fps a
person can cross most of the frame between samples, so box overlap carries
almost no signal (Andrew Ng advisory 2026-05-27). Embeddings come from an
injected embedder — see :mod:`pipeline.reid` for the ONNX implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pipeline.postprocessing import Detection

# Minimum cosine similarity for a detection to join an existing track.
#
# PROVISIONAL — must be tuned on real bending-station footage before the pilot
# is trusted. OSNet similarities are not spread over [0,1]: on a real frame,
# two *completely unrelated* crops (a person and a patch of ceiling) already
# score ~0.49, and the same crop scores ~1.0. Everything meaningful therefore
# lives in a narrow high band, and 0.5 sits at the "totally unrelated" floor —
# it is permissive, not conservative, and it errs toward *merging* two people
# into one track. That is the damaging direction for person-minute reporting
# (a merge silently corrupts totals; a split only shows an absence gap).
#
# Raise this once we can measure same-person vs different-person similarity on
# real crops from the pilot camera. Published OSNet work typically separates
# identities nearer 0.7.
MATCH_THRESHOLD = 0.5

# Minimum YOLO confidence for a detection to *start* a new track.
#
# This is ByteTrack's two-stage association, not a person/not-person filter —
# the advisory (round 8) ruled out thresholding detections away, because real
# people in difficult frames score anywhere from 0.62 to 0.92 and no clean
# cutoff exists. Nothing is discarded here: a faint box still extends a track
# it matches. It just cannot *found* an identity on its own, so a one-off
# low-confidence artifact never becomes a person. A real person entering during
# a difficult frame simply starts their track a frame or two later.
TRACK_BIRTH_CONFIDENCE = 0.5

# How long a track may go unseen and still be re-matched, in seconds.
#
# Body-only OSNet similarity decays as the gap grows — roughly 95% re-match at
# 30 s, 80% at 2 min, 60-70% at 5 min, under 50% beyond — because lighting,
# pose and occlusion all drift. 120 s is the conservative knee of that curve:
# raise it to re-match more returners at the cost of cross-person ID confusion,
# lower it to split more aggressively. Splitting is the safer failure for
# person-minute reporting, so the default errs low.
DEFAULT_MAX_TRACK_AGE_S = 120.0


class Embedder(Protocol):
    """Produces one L2-normalized appearance vector per detection."""

    def embed(self, frame: np.ndarray, detections: list[Detection]) -> list[np.ndarray]: ...


@dataclass
class Track:
    """One person's identity as it persists across frames."""

    track_id: int
    embedding: np.ndarray
    last_seen_s: float


class PersonTracker:
    """Assigns stable ``track_id``s to per-frame detections."""

    def __init__(
        self,
        embedder: Embedder,
        match_threshold: float = MATCH_THRESHOLD,
        max_track_age_s: float = DEFAULT_MAX_TRACK_AGE_S,
        track_birth_confidence: float = TRACK_BIRTH_CONFIDENCE,
    ) -> None:
        self._embedder = embedder
        self._match_threshold = match_threshold
        self._max_track_age_s = max_track_age_s
        self._track_birth_confidence = track_birth_confidence
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        timestamp_s: float,
    ) -> list[Detection]:
        """Stamp ``track_id`` on each detection and return the same list."""
        if not detections:
            return detections

        self._expire(timestamp_s)
        embeddings = self._embedder.embed(frame, detections)
        matched = self._match(embeddings)
        for index, (det, embedding) in enumerate(zip(detections, embeddings, strict=True)):
            track = matched.get(index)
            if track is None:
                if det.confidence < self._track_birth_confidence:
                    # Matched nothing and too faint to found an identity — left
                    # untracked, and so ignored by aggregation.
                    det.track_id = None
                    continue
                track = Track(
                    track_id=self._next_id,
                    embedding=embedding,
                    last_seen_s=timestamp_s,
                )
                self._next_id += 1
                self._tracks.append(track)
            else:
                track.embedding = embedding
                track.last_seen_s = timestamp_s
            det.track_id = track.track_id
        return detections

    def _expire(self, timestamp_s: float) -> None:
        """Retire tracks unseen for longer than ``max_track_age_s``.

        Expired tracks leave the matching pool for good, so a person returning
        after a long absence starts a fresh identity instead of being matched
        on an embedding too stale to trust.
        """
        self._tracks = [
            track
            for track in self._tracks
            if timestamp_s - track.last_seen_s <= self._max_track_age_s
        ]

    def _match(self, embeddings: list[np.ndarray]) -> dict[int, Track]:
        """Greedily match detections to tracks, one-to-one, by similarity.

        Every (detection, track) pair scoring above ``match_threshold`` is a
        candidate; candidates are consumed strongest-first, so the most
        confident pairing wins a contested track. Because a track is retired
        from the pool once it claims a detection, two detections in the same
        frame can never land on one identity however alike they embed.

        Returns a ``detection index → track`` map; unmatched indices are
        absent and become new tracks.
        """
        candidates: list[tuple[float, int, Track]] = []
        for index, embedding in enumerate(embeddings):
            for track in self._tracks:
                # Both vectors are L2-normalized, so the dot product is cosine.
                similarity = float(np.dot(track.embedding, embedding))
                if similarity > self._match_threshold:
                    candidates.append((similarity, index, track))
        # Key on similarity alone — Track is not orderable, and a stable sort
        # keeps ties deterministic in detection order.
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        matched: dict[int, Track] = {}
        claimed: set[int] = set()
        for _similarity, index, track in candidates:
            if index in matched or track.track_id in claimed:
                continue
            matched[index] = track
            claimed.add(track.track_id)
        return matched
