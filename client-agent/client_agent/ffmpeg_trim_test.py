"""Tests for the ffmpeg trim/concat helper (issue #27, Slice 1c.2).

Two code paths:

* **Single chunk** — the requested ``[start, end]`` window lives entirely
  inside one buffer chunk. ffmpeg is invoked with ``-ss``/``-to`` offsets
  relative to that chunk and ``-c copy`` (no re-encode).
* **Multi chunk** — the window crosses chunk boundaries. We write a
  temporary concat-demuxer file list and invoke ffmpeg once with
  ``-f concat -safe 0 -i list.txt -c copy``; ``-ss``/``-to`` then apply
  to the concatenated virtual stream.

The subprocess runner is injected so tests assert ffmpeg argv at the
boundary without forking. The actual mux is exercised by the integration
test against mediamtx (out of scope for unit tests)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from client_agent.buffer import BufferChunk


def _chunk(path: Path, end: datetime, *, segment_s: int = 3600) -> BufferChunk:
    """Cheap BufferChunk factory — no filesystem side effect needed
    because trim_and_concat only reads ``path`` / ``start`` / ``end`` and
    hands the path to ffmpeg, which we mock."""
    return BufferChunk(path=path, start=end - timedelta(seconds=segment_s), end=end)


# ----- 1. single-chunk trim invokes ffmpeg with -ss / -to offsets -----


def test_trim_single_chunk_builds_ss_to_command_with_stream_copy(tmp_path: Path) -> None:
    """Window 10:15 → 10:45 inside chunk_001 (10:00 → 11:00). ``-ss`` /
    ``-to`` express the offset *into the chunk* (15 min / 45 min), and
    ``-c copy`` skips re-encoding so a 30-min trim costs ~ms not minutes.

    Putting ``-ss`` *after* ``-i`` is intentional: in input position ffmpeg
    seeks to the nearest keyframe (fast but imprecise); in output position
    it does decode-and-discard (slower but frame-accurate). Stream-copy
    requires keyframe alignment, so we use the input-position form."""
    from client_agent.ffmpeg_trim import trim_and_concat

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk_path = tmp_path / "chunk_001.mp4"
    chunk_path.write_bytes(b"fake")
    chunk = _chunk(chunk_path, end=t10 + timedelta(hours=1))
    output = tmp_path / "out.mp4"
    calls: list[list[str]] = []

    def runner(cmd, **kwargs):
        calls.append(cmd)

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        output.write_bytes(b"fake-trimmed")
        return _R()

    trim_and_concat(
        chunks=[chunk],
        start=t10 + timedelta(minutes=15),
        end=t10 + timedelta(minutes=45),
        output=output,
        runner=runner,
    )

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "ffmpeg"
    # The chunk path appears via -i, with -ss / -to set to the offsets
    # 900s and 2700s into the chunk (15 min / 45 min).
    assert "-i" in cmd
    assert str(chunk_path) == cmd[cmd.index("-i") + 1]
    assert "-ss" in cmd
    assert cmd[cmd.index("-ss") + 1] == "900"
    assert "-to" in cmd
    assert cmd[cmd.index("-to") + 1] == "2700"
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"
    assert cmd[-1] == str(output)


# ----- 2. multi-chunk trim writes concat list, uses concat demuxer -----


def test_trim_multi_chunk_uses_concat_demuxer_with_file_list(tmp_path: Path) -> None:
    """Window 10:45 → 11:15 crosses the boundary between chunk_001
    (ends 11:00) and chunk_002 (ends 12:00). We write a concat-demuxer
    file list (``file '<path>'`` per chunk, in wallclock order) and
    invoke ffmpeg once with ``-f concat -safe 0 -i list.txt``.

    Offsets are computed against the **first chunk's start** so the
    concatenated virtual stream is sliced consistently — start = 45 min
    into chunk_001 (= 2700 s into the virtual stream), end = 75 min
    into the virtual stream (= 4500 s)."""
    from client_agent.ffmpeg_trim import trim_and_concat

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    c1_path = tmp_path / "chunk_001.mp4"
    c1_path.write_bytes(b"fake1")
    c2_path = tmp_path / "chunk_002.mp4"
    c2_path.write_bytes(b"fake2")
    c1 = _chunk(c1_path, end=t10 + timedelta(hours=1))
    c2 = _chunk(c2_path, end=t10 + timedelta(hours=2))
    output = tmp_path / "out.mp4"
    calls: list[list[str]] = []
    list_contents: list[str] = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        # Read the concat list *here* — this stands in for ffmpeg opening
        # the file during the call. trim_and_concat unlinks it on return,
        # so reading it afterward would race the cleanup.
        list_path = Path(cmd[cmd.index("-i") + 1])
        list_contents.extend(list_path.read_text().splitlines())

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        output.write_bytes(b"fake-concat")
        return _R()

    trim_and_concat(
        chunks=[c1, c2],
        start=t10 + timedelta(minutes=45),
        end=t10 + timedelta(minutes=75),
        output=output,
        runner=runner,
    )

    assert len(calls) == 1
    cmd = calls[0]
    assert "-f" in cmd
    assert cmd[cmd.index("-f") + 1] == "concat"
    assert "-safe" in cmd
    assert cmd[cmd.index("-safe") + 1] == "0"
    # -i should point at the list file — its contents were captured inside
    # the runner (see ``list_contents``), i.e. while ffmpeg would be reading
    # it, because the file is cleaned up once trim_and_concat returns.
    i_idx = cmd.index("-i")
    assert list_contents == [f"file '{c1_path}'", f"file '{c2_path}'"]
    assert cmd[i_idx + 1].endswith(".concat.txt")
    # Offsets land at 2700s / 4500s into the virtual stream.
    assert cmd[cmd.index("-ss") + 1] == "2700"
    assert cmd[cmd.index("-to") + 1] == "4500"
    assert cmd[-1] == str(output)


# ----- 3. empty chunks list raises ValueError (caller misuse guard) -----


def test_trim_empty_chunks_raises_value_error(tmp_path: Path) -> None:
    """Empty buffer should be caught by the poller before getting here,
    but the guard turns a misuse into a clean exception instead of an
    ffmpeg argv error that is much harder to read in journald."""

    from client_agent.ffmpeg_trim import trim_and_concat

    with pytest.raises(ValueError, match="no chunks"):
        trim_and_concat(
            chunks=[],
            start=datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC),
            end=datetime(2026, 5, 15, 11, 0, 0, tzinfo=UTC),
            output=tmp_path / "out.mp4",
            runner=lambda *a, **kw: None,
        )


# ----- 4. multi-chunk path leaves no concat list file behind -----


def test_concat_listfile_cleaned_up(tmp_path: Path) -> None:
    """The concat demuxer needs a temp file list, written with
    ``delete=False`` so ffmpeg can reopen it by name. Nothing removed it —
    every multi-chunk task dropped a ``*.concat.txt`` next to the output
    that never got cleaned, leaking inodes on the appliance (issue #51).
    After ``trim_and_concat`` returns, the output's directory holds no
    ``*.concat.txt``."""
    from client_agent.ffmpeg_trim import trim_and_concat

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    c1_path = tmp_path / "chunk_001.mp4"
    c1_path.write_bytes(b"fake1")
    c2_path = tmp_path / "chunk_002.mp4"
    c2_path.write_bytes(b"fake2")
    out_dir = tmp_path / "trim-out"
    out_dir.mkdir()
    output = out_dir / "out.mp4"

    def runner(cmd, **kwargs):
        # The list file must exist *during* the call — assert that here so
        # cleanup is proven to happen after ffmpeg is done, not before.
        assert Path(cmd[cmd.index("-i") + 1]).exists()
        output.write_bytes(b"fake-concat")

    trim_and_concat(
        chunks=[
            _chunk(c1_path, end=t10 + timedelta(hours=1)),
            _chunk(c2_path, end=t10 + timedelta(hours=2)),
        ],
        start=t10 + timedelta(minutes=45),
        end=t10 + timedelta(minutes=75),
        output=output,
        runner=runner,
    )

    assert list(out_dir.glob("*.concat.txt")) == []


# ----- 5. single-chunk trim raises on non-zero ffmpeg exit -----


def _failing_result(stderr: str):
    """A subprocess-run-shaped result with a non-zero exit and stderr,
    matching what ``subprocess.run(..., text=True)`` returns on failure."""

    class _R:
        returncode = 1
        stdout = ""

    _R.stderr = stderr
    return _R()


def test_single_chunk_trim_raises_on_ffmpeg_failure(tmp_path: Path) -> None:
    """A failed single-chunk trim (unreadable chunk, ENOSPC) currently
    returns normally with no/partial output, so the poller proceeds to
    upload a missing/truncated file. Mirror ``ffmpeg_concat``'s contract:
    a non-zero ffmpeg exit raises ``RuntimeError`` carrying stderr (#57)."""
    from client_agent.ffmpeg_trim import trim_and_concat

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk_path = tmp_path / "chunk_001.mp4"
    chunk_path.write_bytes(b"fake")
    chunk = _chunk(chunk_path, end=t10 + timedelta(hours=1))
    output = tmp_path / "out.mp4"

    def runner(cmd, **kwargs):
        return _failing_result("Invalid data found when processing input")

    with pytest.raises(RuntimeError, match="Invalid data found when processing input"):
        trim_and_concat(
            chunks=[chunk],
            start=t10 + timedelta(minutes=15),
            end=t10 + timedelta(minutes=45),
            output=output,
            runner=runner,
        )


# ----- 6. multi-chunk (concat) trim raises on non-zero ffmpeg exit -----


def test_multi_chunk_trim_raises_on_ffmpeg_failure(tmp_path: Path) -> None:
    """Same contract on the concat branch — a bad concat list or an
    unreadable member chunk must raise, not silently produce nothing.
    The temp concat list is still cleaned up on the failure path (#51)."""
    from client_agent.ffmpeg_trim import trim_and_concat

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    c1_path = tmp_path / "chunk_001.mp4"
    c1_path.write_bytes(b"fake1")
    c2_path = tmp_path / "chunk_002.mp4"
    c2_path.write_bytes(b"fake2")
    out_dir = tmp_path / "trim-out"
    out_dir.mkdir()
    output = out_dir / "out.mp4"

    def runner(cmd, **kwargs):
        return _failing_result("Impossible to open concat list")

    with pytest.raises(RuntimeError, match="Impossible to open concat list"):
        trim_and_concat(
            chunks=[
                _chunk(c1_path, end=t10 + timedelta(hours=1)),
                _chunk(c2_path, end=t10 + timedelta(hours=2)),
            ],
            start=t10 + timedelta(minutes=45),
            end=t10 + timedelta(minutes=75),
            output=output,
            runner=runner,
        )

    # No temp concat list left behind even though the run failed.
    assert list(out_dir.glob("*.concat.txt")) == []


# ----- 7. -ss clamped to zero when the window starts before the chunk -----


def test_ss_clamped_to_zero_when_window_precedes_chunk(tmp_path: Path) -> None:
    """When the task window starts before the first chunk's (mtime-inferred)
    start, the raw offset ``start - chunk.start`` is negative — ffmpeg
    rejects or misbehaves on a negative ``-ss``. The offset must clamp to
    0 so the trim starts at the chunk head (#57)."""
    from client_agent.ffmpeg_trim import trim_and_concat

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk_path = tmp_path / "chunk_001.mp4"
    chunk_path.write_bytes(b"fake")
    chunk = _chunk(chunk_path, end=t10 + timedelta(hours=1))  # chunk.start = 10:00
    output = tmp_path / "out.mp4"
    calls: list[list[str]] = []

    def runner(cmd, **kwargs):
        calls.append(cmd)

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        output.write_bytes(b"fake-trimmed")
        return _R()

    # Window starts 5 min *before* the chunk (09:55) → raw ss would be -300.
    trim_and_concat(
        chunks=[chunk],
        start=t10 - timedelta(minutes=5),
        end=t10 + timedelta(minutes=45),
        output=output,
        runner=runner,
    )

    cmd = calls[0]
    assert cmd[cmd.index("-ss") + 1] == "0"


# ----- 8. covered window reports the requested start back to the caller -----


def _ok_runner(output: Path, calls: list[list[str]] | None = None):
    """A runner that fakes a successful ffmpeg: records argv and writes
    a stand-in output file."""

    def runner(cmd, **kwargs):
        if calls is not None:
            calls.append(cmd)

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        output.write_bytes(b"fake-trimmed")
        return _R()

    return runner


def test_trim_returns_requested_start_when_window_fully_covered(tmp_path: Path) -> None:
    """The platform stamps ``recording_start`` from the *requested*
    ``start_time`` and the engine anchors ``timestamp_s == 0`` to it
    (SPEC.md:173), so the appliance has to say where the delivered clip
    actually begins (#91). When the buffer covers the whole window there
    is no shortfall and the answer is simply the requested start."""
    from client_agent.ffmpeg_trim import trim_and_concat

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk_path = tmp_path / "chunk_001.mp4"
    chunk_path.write_bytes(b"fake")
    chunk = _chunk(chunk_path, end=t10 + timedelta(hours=1))  # covers 10:00 → 11:00
    output = tmp_path / "out.mp4"
    requested_start = t10 + timedelta(minutes=15)

    actual_start = trim_and_concat(
        chunks=[chunk],
        start=requested_start,
        end=t10 + timedelta(minutes=45),
        output=output,
        runner=_ok_runner(output),
    )

    assert actual_start == requested_start


# ----- 9. clamped window reports the chunk's start, not the requested one -----


def test_trim_reports_chunk_start_when_window_precedes_buffer(tmp_path: Path) -> None:
    """The defect behind #91 / gpu-exchange#154. When the buffer does not
    reach back to the requested start the ``-ss`` clamp silently produces a
    clip that begins *later* than asked — but the platform had already
    stamped ``recording_start`` from the request, so every frame's inferred
    wall clock was wrong by the shortfall and shift gating discarded the
    wrong footage. The produced clip starts at the chunk head; that is what
    must be reported."""
    from client_agent.ffmpeg_trim import trim_and_concat

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    chunk_path = tmp_path / "chunk_001.mp4"
    chunk_path.write_bytes(b"fake")
    chunk = _chunk(chunk_path, end=t10 + timedelta(hours=1))  # chunk.start = 10:00
    output = tmp_path / "out.mp4"

    # Requested 09:55, but the buffer only reaches back to 10:00 — a 5 min
    # shortfall that ffmpeg absorbs by clamping -ss to 0.
    actual_start = trim_and_concat(
        chunks=[chunk],
        start=t10 - timedelta(minutes=5),
        end=t10 + timedelta(minutes=45),
        output=output,
        runner=_ok_runner(output),
    )

    assert actual_start == t10


# ----- 10. concat path reports the actual start too -----


def test_trim_multi_chunk_reports_actual_start(tmp_path: Path) -> None:
    """The concat branch carries its own copy of the ``-ss`` clamp, against
    the *first* chunk's start. A window reaching back before the buffer is
    exactly the multi-chunk shape a long request takes (#91), so the same
    reporting contract has to hold here — otherwise the fix only covers
    windows that happen to fit inside one chunk."""
    from client_agent.ffmpeg_trim import trim_and_concat

    t10 = datetime(2026, 5, 15, 10, 0, 0, tzinfo=UTC)
    c1_path = tmp_path / "chunk_001.mp4"
    c1_path.write_bytes(b"fake1")
    c2_path = tmp_path / "chunk_002.mp4"
    c2_path.write_bytes(b"fake2")
    c1 = _chunk(c1_path, end=t10 + timedelta(hours=1))  # 10:00 → 11:00
    c2 = _chunk(c2_path, end=t10 + timedelta(hours=2))  # 11:00 → 12:00
    output = tmp_path / "out.mp4"

    # Requested 09:45; the buffer's oldest chunk only starts at 10:00.
    actual_start = trim_and_concat(
        chunks=[c1, c2],
        start=t10 - timedelta(minutes=15),
        end=t10 + timedelta(minutes=90),
        output=output,
        runner=_ok_runner(output),
    )

    assert actual_start == t10
