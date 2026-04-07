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
from collections.abc import Iterator
from pathlib import Path

import numpy as np


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


def iter_frames(
    video_path: str | Path,
    fps: int = 1,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_s, frame_bgr)`` for every frame at ``fps``."""
    width, height = _probe_dimensions(video_path)
    frame_size = width * height * 3  # bgr24

    cmd = [
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
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdout is None:  # pragma: no cover - defensive
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
        if rc != 0:
            stderr_bytes = b""
            if proc.stderr is not None:
                try:
                    stderr_bytes = proc.stderr.read() or b""
                except Exception:  # pragma: no cover - best effort
                    stderr_bytes = b""
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg failed (exit {rc}) streaming frames from {video_path}: {stderr}"
            )
