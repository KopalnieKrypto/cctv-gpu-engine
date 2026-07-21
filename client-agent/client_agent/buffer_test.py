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


# ----- 4b. trim_all_cameras trims every camera dir under the buffer root -----


def test_trim_all_cameras_trims_every_camera_dir(tmp_path: Path) -> None:
    """The maintenance thread can't enumerate cameras itself — the buffer
    owns the root, so it discovers every ``base_dir/{camera_id}/`` dir and
    applies ``trim_old_chunks`` to each with a single pinned ``now``.

    Two cameras, each with one stale chunk (2h old, outside a 1h window)
    and one fresh chunk (30 min old). After one pass every stale chunk is
    gone and every fresh chunk survives — for *both* cameras without the
    caller naming them. The returned count is the total deleted across
    cameras so a maintenance thread can emit one metric per sweep."""
    from client_agent.buffer import RollingBuffer

    base = tmp_path / "cctv-buffer"
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    for cam in ("cam-1", "cam-2"):
        _make_chunk(base / cam / "chunk_stale.mp4", end=now - timedelta(hours=2))
        _make_chunk(base / cam / "chunk_fresh.mp4", end=now - timedelta(minutes=30))

    buffer = RollingBuffer(base_dir=base, buffer_hours=1, segment_seconds=3600)
    deleted = buffer.trim_all_cameras(now=now)

    assert deleted == 2
    for cam in ("cam-1", "cam-2"):
        survivors = sorted(p.name for p in (base / cam).glob("chunk_*.mp4"))
        assert survivors == ["chunk_fresh.mp4"]


def test_trim_all_cameras_on_missing_root_returns_zero(tmp_path: Path) -> None:
    """First maintenance tick can fire before any recorder created the
    buffer root (platform mode boots the poller + maintenance threads up
    front, recorders spawn only once the operator approves a camera). A
    missing root is "nothing to trim", not an ``OSError`` that kills the
    daemon thread."""
    from client_agent.buffer import RollingBuffer

    buffer = RollingBuffer(
        base_dir=tmp_path / "never-created", buffer_hours=1, segment_seconds=3600
    )

    assert buffer.trim_all_cameras(now=datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)) == 0


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


# ----- 6. set_buffer_hours re-widens the retention window at runtime (#85) -----


def test_set_buffer_hours_changes_retention_window_for_next_trim(tmp_path: Path) -> None:
    """The platform lets an admin edit ``buffer_hours`` at runtime (#85). The
    trim cron holds one long-lived buffer, so retention must be mutable in
    place — no rebuild, no ffmpeg restart. After ``set_buffer_hours(6)`` the
    very next ``trim_old_chunks`` keeps 6h of history instead of 1h."""
    from client_agent.buffer import RollingBuffer

    base = tmp_path / "cctv-buffer"
    cam = "cam-1"
    now = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)
    # 10 chunks ending at now-9h … now (one per hour).
    for i in range(10):
        end = now - timedelta(hours=9 - i)
        _make_chunk(base / cam / f"chunk_{i:03d}.mp4", end=end)

    buffer = RollingBuffer(base_dir=base, buffer_hours=1, segment_seconds=3600)

    buffer.set_buffer_hours(6)
    deleted = buffer.trim_old_chunks(cam, now=now)

    # cutoff = now - 6h; trim deletes chunks whose end is <= cutoff. Ends run
    # now-9h…now, so now-9h/-8h/-7h/-6h go (the -6h chunk lands on the cutoff
    # and counts as stale) → 4 deleted, 6 survive. With the seed buffer_hours=1
    # only the final chunk would have survived — proof the setter took effect.
    survivors = sorted(p.name for p in (base / cam).glob("chunk_*.mp4"))
    assert deleted == 4
    assert survivors == [f"chunk_{i:03d}.mp4" for i in range(4, 10)]


# ----- 5. oldest_chunk_at: how far back the buffer actually reaches (#92) -----


def test_oldest_chunk_at_returns_oldest_chunk_mtime(tmp_path: Path) -> None:
    """The platform cannot see what the box holds, so the heartbeat reports
    the oldest buffered chunk per camera (#92 / gpu-exchange#157 Tier 2).

    Raw mtime, not the inferred ``mtime - segment_seconds`` start: the
    platform column is literally ``buffer_oldest_chunk_at`` and its
    ``cameraBufferDepth`` derivation does ``now - value`` against a
    deliberately generous half-window floor, which absorbs the sub-segment
    difference. Reporting the inferred start would over-claim history
    whenever the oldest chunk is a respawn-truncated partial."""
    from client_agent.buffer import RollingBuffer

    base = tmp_path / "cctv-buffer"
    cam = "cam-1"
    now = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)
    oldest = now - timedelta(hours=5)
    # Written out of order so the result cannot come from glob iteration order.
    _make_chunk(base / cam / "chunk_b.mp4", end=now - timedelta(hours=1))
    _make_chunk(base / cam / "chunk_a.mp4", end=oldest)
    _make_chunk(base / cam / "chunk_c.mp4", end=now - timedelta(hours=3))

    buffer = RollingBuffer(base_dir=base, buffer_hours=6, segment_seconds=3600)

    assert buffer.oldest_chunk_at(cam) == oldest


# ----- 6. buffer_depths: whole-root map for the heartbeat (#92) -----


def test_buffer_depths_maps_every_camera_and_omits_empty_dirs(tmp_path: Path) -> None:
    """The heartbeat needs one map for the whole box, not a per-camera loop
    at the call site — so the buffer owns the root walk, exactly as
    :meth:`trim_all_cameras` already does.

    A camera whose directory exists but holds no chunk is *omitted* rather
    than sent as ``null``: the platform stores NULL for "unknown depth" and
    its schema is ``z.record(z.string(), z.string().datetime())``, so an
    explicit null would fail validation for the entire beat."""
    from client_agent.buffer import RollingBuffer

    base = tmp_path / "cctv-buffer"
    now = datetime(2026, 7, 21, 12, 0, 0, tzinfo=UTC)
    _make_chunk(base / "cam-a" / "chunk_1.mp4", end=now - timedelta(hours=4))
    _make_chunk(base / "cam-a" / "chunk_2.mp4", end=now - timedelta(hours=1))
    _make_chunk(base / "cam-b" / "chunk_1.mp4", end=now - timedelta(hours=2))
    # Recorder created the dir but ffmpeg never finalized a chunk.
    (base / "cam-empty").mkdir(parents=True)
    # A stray file at the root is not a camera.
    (base / "notes.txt").write_bytes(b"x")

    buffer = RollingBuffer(base_dir=base, buffer_hours=6, segment_seconds=3600)

    assert buffer.buffer_depths() == {
        "cam-a": now - timedelta(hours=4),
        "cam-b": now - timedelta(hours=2),
    }


def test_buffer_depths_empty_when_root_missing(tmp_path: Path) -> None:
    """No recorder has ever written — "nothing buffered", not a crash. This
    runs inside the boot heartbeat, before any recorder spawns, so raising
    here would take down registration on every cold start."""
    from client_agent.buffer import RollingBuffer

    buffer = RollingBuffer(base_dir=tmp_path / "never-created", buffer_hours=6)

    assert buffer.buffer_depths() == {}
