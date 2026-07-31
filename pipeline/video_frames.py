"""Streaming frame extraction at a configurable fps via ffmpeg rawvideo pipe.

Phase 1's :mod:`pipeline.frame_extractor` extracts a *single* frame at a
chosen timestamp via fast-seek. For full-video analysis (issue #4) we need a
generator that yields every frame at 1 fps without ever loading the whole
decoded video in RAM.

The strategy:

1. Probe the video dimensions once with ``ffprobe`` so we know the byte size
   of one rawvideo frame (W × H × 3 for ``bgr24``).
2. Spawn ``ffmpeg -i input -vf fps=N -pix_fmt bgr24 -f rawvideo -`` and read
   from its stdout in fixed-size chunks. Each chunk is one decoded BGR frame.
3. Yield ``(timestamp_s, frame)`` tuples to the caller.

Memory: at any moment we hold exactly one decoded frame plus any pending
ffmpeg pipe buffer (a few MB). Stable for arbitrarily long videos.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import numpy as np

# Cap the amount of ffmpeg stderr echoed into a RuntimeError. The sink itself
# is unbounded on disk (never blocks the writer), but a decode-error storm can
# be megabytes — only the tail carries the actionable failure line.
_STDERR_TAIL_BYTES = 8192


def _probe_dimensions(video_path: str | Path) -> tuple[int, int]:
    """Return ``(width, height)`` of the first video stream via ``ffprobe``."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe not found on PATH — install ffmpeg before running the pipeline"
        ) from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffprobe failed for {video_path}: {stderr}")
    payload = json.loads(result.stdout.decode("utf-8", errors="replace"))
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {video_path}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def probe_duration_s(video_path: str | Path) -> float | None:
    """Return the clip length in seconds, or ``None`` if it can't be read.

    Used by :mod:`pipeline.analyze` to turn a within-chunk timestamp into a
    fraction of the job, so ``progress`` climbs during a chunk instead of
    reporting the chunk-start percentage (issue #115).

    Deliberately total: a missing ffprobe, an unreadable file, or a container
    with no duration in its header (a stream-copied RTSP chunk reports ``N/A``)
    all yield ``None``. Progress reporting is a diagnostic — it must never be
    the reason a job fails, and the caller degrades to chunk-boundary progress.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=30)  # noqa: S603
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.decode("utf-8", errors="replace").strip())
    except ValueError:
        return None  # "N/A" — header carries no duration
    if duration <= 0 or duration != duration:  # NaN-safe
        return None
    return duration


def _build_ffmpeg_cmd(video_path: str | Path, fps: int) -> list[str]:
    """Assemble the streaming ffmpeg command (extracted for testability)."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]


def _read_stderr_tail(sink: BinaryIO) -> str:
    """Return the last ``_STDERR_TAIL_BYTES`` of the ffmpeg stderr sink.

    The sink is a regular file (seekable), so we seek near the end rather
    than load a potentially multi-MB decode-error dump into memory.
    """
    try:
        sink.seek(0, 2)  # SEEK_END
        size = sink.tell()
        sink.seek(max(0, size - _STDERR_TAIL_BYTES))
        data = sink.read() or b""
    except Exception:  # pragma: no cover - best effort diagnostics
        return ""
    return data.decode("utf-8", errors="replace").strip()


def iter_frames(
    video_path: str | Path,
    fps: int = 1,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_s, frame_bgr)`` for every frame at ``fps``."""
    width, height = _probe_dimensions(video_path)
    frame_size = width * height * 3  # bgr24

    cmd = _build_ffmpeg_cmd(video_path, fps)
    # Redirect stderr to a real temp file, not a PIPE. A corrupt surveillance
    # MP4 makes ffmpeg emit repeated decode-error lines; with stderr=PIPE and
    # only stdout being drained, the ~64 KiB pipe buffer fills, ffmpeg blocks
    # on its stderr write, stops producing stdout, and our read() blocks
    # forever (issue #60). A temp file never applies backpressure to the
    # writer, so the frame loop can always make progress.
    stderr_sink = tempfile.TemporaryFile()
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=stderr_sink,
    )
    if proc.stdout is None:  # pragma: no cover - defensive
        stderr_sink.close()
        raise RuntimeError("ffmpeg subprocess produced no stdout pipe")

    index = 0
    try:
        while True:
            chunk = proc.stdout.read(frame_size)
            if not chunk:
                break
            if len(chunk) < frame_size:
                # Truncated tail — most likely a corrupted final frame; skip.
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
            yield (index / fps, frame)
            index += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:  # pragma: no cover - best effort
            pass
        rc = proc.wait()
        try:
            if rc != 0:
                stderr = _read_stderr_tail(stderr_sink)
                raise RuntimeError(
                    f"ffmpeg failed (exit {rc}) streaming frames from {video_path}: {stderr}"
                )
        finally:
            stderr_sink.close()
