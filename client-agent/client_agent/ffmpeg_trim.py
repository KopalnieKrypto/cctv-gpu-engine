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
        ss = int((start - chunk.start).total_seconds())
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
        runner(cmd, capture_output=True, text=True)
        return

    # Multi-chunk: write a concat-demuxer file list next to the output, then
    # invoke ffmpeg once with -f concat. Offsets are relative to the first
    # chunk's start so the virtual concatenated stream is sliced consistently.
    first = chunks[0]
    ss = int((start - first.start).total_seconds())
    to = int((end - first.start).total_seconds())
    list_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".concat.txt",
        delete=False,
        dir=output.parent,
    )
    try:
        for c in chunks:
            list_file.write(f"file '{c.path}'\n")
    finally:
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
    runner(cmd, capture_output=True, text=True)
