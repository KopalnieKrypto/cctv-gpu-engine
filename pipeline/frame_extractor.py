"""Single-frame MP4 extraction via ffmpeg subprocess.

Uses fast-seek (``-ss`` before ``-i``) to jump to the requested timestamp,
then decodes a single frame as PNG to stdout. The PNG bytes are decoded
in-process via OpenCV, so the only filesystem I/O is reading the input video.

For phase 2 (full video at 1 fps) we will write a separate ``extract_frames``
that streams JPEGs frame-by-frame; this module covers the issue #2 single-frame
proof-of-concept only.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def extract_frame_at(video_path: Path | str, timestamp_s: float) -> np.ndarray:
    """Extract a single decoded frame from ``video_path`` at ``timestamp_s``.

    Returns:
        A BGR ``numpy.ndarray`` of shape ``(H, W, 3)``, dtype ``uint8``.

    Raises:
        RuntimeError: when ffmpeg is missing, exits non-zero, returns no
            data, or produces output that cannot be decoded as an image.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(timestamp_s),  # fast seek BEFORE -i
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg not found on PATH — install it before running the pipeline"
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}) extracting frame at "
            f"t={timestamp_s}s from {video_path}: {stderr}"
        )

    if not result.stdout:
        raise RuntimeError(
            f"ffmpeg returned no frame data for {video_path} at t={timestamp_s}s "
            "(timestamp may be past the end of the video)"
        )

    buffer = np.frombuffer(result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(
            f"failed to decode ffmpeg output as image for {video_path} at t={timestamp_s}s"
        )
    return frame
