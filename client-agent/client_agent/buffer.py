"""Rolling recording buffer for the client appliance (issue #27).

The appliance records continuously into a ring of 1-hour chunks per camera
under ``base_dir/{camera_id}/chunk_NNN.mp4``. When the platform sends a
task with a historical ``[start_time, end_time]`` window, the task poller
asks the buffer for the chunks that overlap that window and hands them to
:mod:`ffmpeg_trim`.

A chunk's covered time range is inferred from its mtime: ffmpeg's segment
muxer finalizes (closes) a chunk file at the boundary between segments, so
``mtime`` ≈ end of the recording window, and ``mtime - segment_seconds`` ≈
start. The approximation is fine because downstream ffmpeg trim re-anchors
using exact ``-ss``/``-to`` offsets — the buffer's job is only to narrow
the candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class BufferChunk:
    """A chunk file with its inferred wallclock range."""

    path: Path
    start: datetime
    end: datetime


class RollingBuffer:
    """Per-camera rolling buffer rooted at ``base_dir/{camera_id}/``."""

    def __init__(
        self,
        *,
        base_dir: Path,
        buffer_hours: int,
        segment_seconds: int = 3600,
    ) -> None:
        self._base_dir = base_dir
        self._buffer_hours = buffer_hours
        self._segment_seconds = segment_seconds

    def set_buffer_hours(self, buffer_hours: int) -> None:
        """Re-point the retention window at runtime (issue #85).

        The maintenance thread holds one long-lived buffer, so an admin
        editing ``buffer_hours`` in the platform UI must change retention in
        place — the next :meth:`trim_old_chunks` uses the new window. A larger
        value keeps more history *going forward* (already-trimmed chunks are
        gone); a smaller one deletes more on the next pass. No effect on the
        recorder's ffmpeg — retention is purely a delete-side policy."""
        self._buffer_hours = buffer_hours

    def chunks_in_range(
        self, camera_id: str, *, start: datetime, end: datetime
    ) -> list[BufferChunk]:
        """Return chunks whose time range overlaps ``[start, end]``.

        Sorted by start time so the downstream concat sees them in
        wallclock order."""
        cam_dir = self._base_dir / camera_id
        if not cam_dir.exists():
            return []

        seg = timedelta(seconds=self._segment_seconds)
        overlapping: list[BufferChunk] = []
        for path in cam_dir.glob("chunk_*.mp4"):
            chunk_end = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            chunk_start = chunk_end - seg
            if chunk_start <= end and chunk_end >= start:
                overlapping.append(BufferChunk(path=path, start=chunk_start, end=chunk_end))
        overlapping.sort(key=lambda c: c.start)
        return overlapping

    def trim_old_chunks(self, camera_id: str, *, now: datetime) -> int:
        """Delete chunks whose end is older than ``buffer_hours`` ago.

        Returns the number of files deleted so a maintenance thread can
        emit a metric without a second stat pass. ``now`` is injected so
        the periodic worker can pin a single clock reading across cameras."""
        cam_dir = self._base_dir / camera_id
        if not cam_dir.exists():
            return 0
        cutoff = now - timedelta(hours=self._buffer_hours)
        deleted = 0
        for path in cam_dir.glob("chunk_*.mp4"):
            chunk_end = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            # ``<=`` so a chunk whose end lands exactly on the cutoff is
            # treated as stale (its content predates the retention window).
            if chunk_end <= cutoff:
                path.unlink()
                deleted += 1
        return deleted

    def trim_all_cameras(self, *, now: datetime) -> int:
        """Trim every camera dir under the buffer root, return total deleted.

        The buffer owns its root, so it — not the caller — enumerates the
        per-camera dirs and applies :meth:`trim_old_chunks` to each with a
        single pinned ``now``. This is the production caller for the
        retention policy: a maintenance thread ticks this on an interval so
        ``BUFFER_HOURS`` is actually enforced and the appliance disk stays
        bounded (issue #51). A missing root (no recorder has written yet)
        is "nothing to trim", not an error."""
        if not self._base_dir.exists():
            return 0
        deleted = 0
        for cam_dir in self._base_dir.iterdir():
            if cam_dir.is_dir():
                deleted += self.trim_old_chunks(cam_dir.name, now=now)
        return deleted

    def has_recorded(self, camera_id: str) -> bool:
        """``True`` iff the camera has at least one chunk on disk.

        Used by the task poller to distinguish "no recorder ever ran"
        (empty buffer → operator should wait) from "recorder ran but
        history doesn't cover the request" (stale → operator should
        extend ``BUFFER_HOURS`` or request a more recent window). The
        camera dir alone doesn't count — the recorder creates it before
        ffmpeg writes anything."""
        cam_dir = self._base_dir / camera_id
        if not cam_dir.exists():
            return False
        return any(cam_dir.glob("chunk_*.mp4"))
