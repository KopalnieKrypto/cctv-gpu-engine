"""Tests for the rolling recording buffer (issue #27, Slice 1c.2).

The buffer is the appliance-side ring of recorded chunks the task poller
trims from when the platform asks for a historical time range. Two
behaviors live here:

* :meth:`RollingBuffer.chunks_in_range` — given a camera id and a
  ``[start, end]`` window, return the chunks that overlap, with their
  inferred time bounds. The downstream :mod:`ffmpeg_trim` helper uses the
  bounds to compute ``-ss``/``-to`` offsets.
* :meth:`RollingBuffer.trim_old_chunks` — delete chunks whose end is
  older than ``buffer_hours``. Called periodically by a maintenance
  thread to keep the buffer bounded.

Filesystem is real (``tmp_path``) — the buffer is just files on disk,
mocking ``Path.glob`` / ``Path.stat`` would be testing the mock. mtime is
set explicitly via ``os.utime`` so each test pins a deterministic clock.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _make_chunk(path: Path, *, end: datetime) -> None:
    """Create a chunk file with mtime set to ``end``.

    ffmpeg's segment muxer finalizes a chunk by closing the file when the
    next segment opens; the close time becomes the file's mtime and stands
    in for "wallclock at end of recording window" everywhere downstream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-mp4")
    ts = end.timestamp()
    os.utime(path, (ts, ts))


# ----- 1. chunks_in_range returns only overlapping chunks -----


def test_chunks_in_range_returns_only_overlapping_chunks(tmp_path: Path) -> None:
    """Three back-to-back 1h chunks for one camera. Query a 30-min window
    inside the middle chunk → only that chunk comes back. Bound metadata
    on the returned ``BufferChunk`` is what :mod:`ffmpeg_trim` will use to
    compute the ``-ss`` offset, so we assert it here too."""
    from client_agent.buffer import RollingBuffer

    base = tmp_path / "cctv-buffer"
    cam = "cam-1"
    # Three chunks ending at 10:00, 11:00, 12:00 (segment_seconds=3600 →
    # ranges [09:00, 10:00], [10:00, 11:00], [11:00, 12:00]).
    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    t11 = t10 + timedelta(hours=1)
    t12 = t10 + timedelta(hours=2)
    _make_chunk(base / cam / "chunk_000.mp4", end=t10)
    _make_chunk(base / cam / "chunk_001.mp4", end=t11)
    _make_chunk(base / cam / "chunk_002.mp4", end=t12)

    buffer = RollingBuffer(base_dir=base, buffer_hours=4, segment_seconds=3600)

    # Query window 10:15 → 10:45 — fully inside the middle chunk.
    result = buffer.chunks_in_range(
        cam,
        start=datetime(2026, 5, 15, 10, 15, 0, tzinfo=UTC),
        end=datetime(2026, 5, 15, 10, 45, 0, tzinfo=UTC),
    )

    assert len(result) == 1
    assert result[0].path.name == "chunk_001.mp4"
    assert result[0].start == t10  # 10:00 = end - segment_seconds
    assert result[0].end == t11


# ----- 2. chunks_in_range straddling boundary returns both chunks, sorted -----


def test_chunks_in_range_straddling_boundary_returns_both_chunks(tmp_path: Path) -> None:
    """Query 10:45 → 11:15 — overlaps the end of chunk_001 and the start of
    chunk_002. Both chunks come back, sorted by start time so the downstream
    concat sees them in wallclock order regardless of glob order."""
    from client_agent.buffer import RollingBuffer

    base = tmp_path / "cctv-buffer"
    cam = "cam-1"
    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    t11 = t10 + timedelta(hours=1)
    t12 = t10 + timedelta(hours=2)
    # Write in reverse order to prove sort happens (glob() order is
    # filesystem-dependent and not guaranteed to match write order).
    _make_chunk(base / cam / "chunk_002.mp4", end=t12)
    _make_chunk(base / cam / "chunk_001.mp4", end=t11)
    _make_chunk(base / cam / "chunk_000.mp4", end=t10)

    buffer = RollingBuffer(base_dir=base, buffer_hours=4, segment_seconds=3600)

    result = buffer.chunks_in_range(
        cam,
        start=datetime(2026, 5, 15, 10, 45, 0, tzinfo=UTC),
        end=datetime(2026, 5, 15, 11, 15, 0, tzinfo=UTC),
    )

    assert [c.path.name for c in result] == ["chunk_001.mp4", "chunk_002.mp4"]


# ----- 3. chunks_in_range on unknown camera returns empty (no FileNotFoundError) -----


def test_chunks_in_range_on_unknown_camera_returns_empty(tmp_path: Path) -> None:
    """First task ever for a camera the recorder hasn't booted yet → the
    cam dir doesn't exist on disk. The poller treats this as "empty
    buffer", not as an OS error to crash on."""
    from client_agent.buffer import RollingBuffer

    base = tmp_path / "cctv-buffer"
    base.mkdir()
    buffer = RollingBuffer(base_dir=base, buffer_hours=4, segment_seconds=3600)

    result = buffer.chunks_in_range(
        "cam-never-recorded",
        start=datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC),
        end=datetime(2026, 5, 15, 11, 0, 0, tzinfo=UTC),
    )

    assert result == []


# ----- 4. trim_old_chunks keeps last buffer_hours, deletes the rest -----


def test_trim_old_chunks_deletes_chunks_older_than_buffer_hours(tmp_path: Path) -> None:
    """25 mock chunks spaced 1h apart, buffer_hours=1 → only the last chunk
    survives (every chunk older than 1h ago relative to ``now`` gets
    deleted). The clock is injected so the test is deterministic; the
    issue body calls for this exact "~24 deleted" sanity check."""
    from client_agent.buffer import RollingBuffer

    base = tmp_path / "cctv-buffer"
    cam = "cam-1"
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    # 25 chunks ending at now - 24h … now (one per hour).
    for i in range(25):
        end = now - timedelta(hours=24 - i)
        _make_chunk(base / cam / f"chunk_{i:03d}.mp4", end=end)

    buffer = RollingBuffer(base_dir=base, buffer_hours=1, segment_seconds=3600)
    deleted = buffer.trim_old_chunks(cam, now=now)

    survivors = sorted((base / cam).glob("chunk_*.mp4"))
    assert len(survivors) == 1
    assert survivors[0].name == "chunk_024.mp4"
    assert deleted == 24


# ----- 5. has_recorded distinguishes "no history" from "stale history" -----


def test_has_recorded_true_when_camera_dir_has_chunks(tmp_path: Path) -> None:
    """The poller uses ``has_recorded`` to pick between two failure
    messages ("buffer empty" vs. "time range outside buffer"). The
    semantic is "at any point this camera produced at least one chunk",
    so even a single stale chunk in the dir counts."""
    from client_agent.buffer import RollingBuffer

    base = tmp_path / "cctv-buffer"
    t = datetime(2026, 5, 15, 0, 0, 0, tzinfo=UTC)
    _make_chunk(base / "cam-1" / "chunk_000.mp4", end=t)
    # cam-2 has its directory but no chunks (recorder booted and died
    # before writing) — treat that the same as "no recordings".
    (base / "cam-2").mkdir(parents=True)

    buffer = RollingBuffer(base_dir=base, buffer_hours=1, segment_seconds=3600)

    assert buffer.has_recorded("cam-1") is True
    assert buffer.has_recorded("cam-2") is False
    assert buffer.has_recorded("cam-never") is False
