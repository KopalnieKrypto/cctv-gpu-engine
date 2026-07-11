"""ffmpeg trim/concat helper for the task poller (issue #27).

Given the chunks the :class:`client_agent.buffer.RollingBuffer` returned
and a ``[start, end]`` window, materialize a single MP4 covering exactly
that window. Two code paths:

* **Single chunk** — ``ffmpeg -ss <off> -to <off> -i chunk.mp4 -c copy out.mp4``.
* **Multi chunk** — write a concat-demuxer file list, then
  ``ffmpeg -f concat -safe 0 -i list.txt -ss <off> -to <off> -c copy out.mp4``.

Stream-copy (no re-encode) keeps a 30-min trim under ~1s; the trade-off
is that ``-ss`` snaps to the previous keyframe (input position) which
typically lands ~2 s before the requested start. The downstream pipeline
treats this as acceptable — the pose detector samples at 1 fps so a
1-2 s lead-in is invisible at the report layer.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from client_agent.buffer import BufferChunk


def trim_and_concat(
    *,
    chunks: list[BufferChunk],
    start: datetime,
    end: datetime,
    output: Path,
    runner: Callable[..., Any],
) -> None:
    """Materialize a single MP4 covering ``[start, end]`` from ``chunks``.

    Raises :class:`ValueError` if ``chunks`` is empty — the caller (poller)
    should have caught "empty buffer" before getting here, but the guard
    keeps a misuse from producing an ffmpeg "no input" error which is
    harder to read in journald."""
    if not chunks:
        raise ValueError("trim_and_concat called with no chunks")

    if len(chunks) == 1:
        chunk = chunks[0]
        # Clamp to 0: when the task window starts before the chunk's
        # (mtime-inferred) start, the raw offset is negative and ffmpeg
        # rejects/misbehaves on a negative -ss. Start at the chunk head.
        ss = max(0, int((start - chunk.start).total_seconds()))
        to = int((end - chunk.start).total_seconds())
        cmd = [
            "ffmpeg",
            "-ss",
            str(ss),
            "-to",
            str(to),
            "-i",
            str(chunk.path),
            "-c",
            "copy",
            str(output),
        ]
        _check(runner(cmd, capture_output=True, text=True))
        return

    # Multi-chunk: write a concat-demuxer file list next to the output, then
    # invoke ffmpeg once with -f concat. Offsets are relative to the first
    # chunk's start so the virtual concatenated stream is sliced consistently.
    #
    # Tolerance note (#57): the offset math assumes a *gapless* concat —
    # ``first.start + elapsed``. Recorder respawns or camera dropouts can
    # leave gaps between chunks, which shift the slice later by the total gap
    # duration. For the 1 fps pose sampler a few seconds of drift is
    # invisible at the report layer; a task spanning a multi-minute recorder
    # outage would need offsets derived from cumulative chunk durations
    # instead (deferred — no such footage in the current test corpus).
    first = chunks[0]
    ss = max(0, int((start - first.start).total_seconds()))
    to = int((end - first.start).total_seconds())
    # ``delete=False`` so ffmpeg can reopen the list by name; we own the
    # cleanup ourselves in the ``finally`` below. Without it every multi-
    # chunk task leaks a ``*.concat.txt`` next to the output (issue #51).
    list_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".concat.txt",
        delete=False,
        dir=output.parent,
    )
    try:
        for c in chunks:
            list_file.write(f"file '{c.path}'\n")
        list_file.close()

        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file.name,
            "-ss",
            str(ss),
            "-to",
            str(to),
            "-c",
            "copy",
            str(output),
        ]
        _check(runner(cmd, capture_output=True, text=True))
    finally:
        list_file.close()
        Path(list_file.name).unlink(missing_ok=True)


def _check(result: Any) -> None:
    """Raise on a non-zero ffmpeg exit, mirroring :func:`ffmpeg_concat`.

    Without this, a failed trim (unreadable chunk, ENOSPC, bad concat
    list) returns normally with no output file or a truncated one, and
    the poller then uploads a missing/partial MP4 as a success (#57).
    The ``runner`` is ``subprocess.run(..., text=True)`` in production so
    ``stderr`` is a ``str``; bytes are decoded defensively for other
    runners."""
    returncode = getattr(result, "returncode", 0)
    if returncode == 0:
        return
    stderr = getattr(result, "stderr", "") or ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    raise RuntimeError(f"ffmpeg trim exited {returncode}: {stderr}")
