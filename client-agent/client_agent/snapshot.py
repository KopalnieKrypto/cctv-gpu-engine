"""Per-camera snapshot grabber for the Flask UI (issue #41).

Two backends, dispatched on URL scheme:

* HTTP / HTTPS — vendor's native ONVIF ``GetSnapshotUri`` (single GET,
  no RTSP handshake). Stdlib ``urllib.request`` — no third-party HTTP
  client needed for a one-shot JPEG fetch.
* RTSP — ffmpeg opens the stream, decodes past a keyframe (``-ss`` settle),
  captures one frame and encodes JPEG to stdout. Used when the camera has no
  HTTP snapshot endpoint, or in platform mode where the heartbeat config only
  carries ``rtsp_url``. ffmpeg (not cv2) because cv2's ``read()`` returns
  error-concealment garbage when a 4K H.265 stream is opened mid-GOP.

Both backends are injectable via :func:`build_snapshot_grabber` so unit
tests never touch the system boundary. The defaults (:func:`_http_fetch`
and :func:`_rtsp_frame_grab`) wrap urllib + ffmpeg with the timeout and
JPEG quality the issue specifies.
"""

from __future__ import annotations

import logging
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Scale to <= 640 px wide — issue #41 spec. Wider thumbs would just be
# wasted bytes on a "My cameras" preview tile.
_MAX_WIDTH_PX = 640
# ffmpeg ``-q:v`` for MJPEG output (2=best … 31=worst). 4 ≈ the old cv2 JPEG
# quality 70: visually fine for a thumbnail, small on the wire.
_FFMPEG_MJPEG_QSCALE = 4

# RTSP snapshot settle: seconds of the LIVE stream ffmpeg decodes-and-discards
# (``-ss`` after ``-i``) before capturing the frame. A 4K H.265 stream opened
# mid-GOP first emits error-concealment garbage — a flat grey/green undecoded
# bottom (the "most green" preview on 192.168.88.84/.85) — until a keyframe
# lands and its slices are all received. 1.0 s clears it on the operator's LAN
# batch (verified on the green-prone .85: bottom detail jumps from ~0 to ~29),
# while staying well inside the grab budget (~2.5-3.5 s wallclock end to end).
_RTSP_SNAPSHOT_SETTLE_S = 1.0
# Floor for the ffmpeg subprocess deadline. The grabber's ``timeout_s`` was
# tuned for the old single-cv2-read (~1-2 s); the settle-then-grab needs more,
# so never run ffmpeg on a deadline tighter than this regardless of the caller.
_RTSP_GRAB_MIN_TIMEOUT_S = 8.0

SnapshotGrabberFn = Callable[[str, float], bytes]


def build_snapshot_grabber(
    *,
    http_fetcher: SnapshotGrabberFn | None = None,
    rtsp_grabber: SnapshotGrabberFn | None = None,
) -> SnapshotGrabberFn:
    """Build a URL-scheme-dispatching snapshot grabber.

    Both backends default to the production implementations (urllib for
    HTTP, ffmpeg for RTSP). Tests inject in-memory fakes — neither default
    touches the network at import time, so a unit test that never calls
    the grabber pays no ffmpeg cost."""
    http_fetcher = http_fetcher or _http_fetch
    rtsp_grabber = rtsp_grabber or _rtsp_frame_grab

    def grab(url: str, timeout_s: float) -> bytes:
        if url.startswith(("http://", "https://")):
            return http_fetcher(url, timeout_s)
        return rtsp_grabber(url, timeout_s)

    return grab


def _http_fetch(url: str, timeout_s: float) -> bytes:
    """Single GET against the vendor's ONVIF snapshot endpoint.

    ``urllib.request.urlopen`` keeps the dependency surface minimal —
    httpx is already in the project but pulls async-io and a connection
    pool we don't need for a one-shot JPEG. The 5 s ``timeout`` covers
    both connect and read.

    No JPEG re-encoding: ONVIF snapshot endpoints return JPEG natively
    and the operator's thumbnail tile doesn't need a re-compression
    pass. Bytes flow through verbatim.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.URLError as exc:
        # Surface the URL so the operator can ping the camera directly.
        # Strip the embedded reason to keep the body single-line.
        raise RuntimeError(f"http snapshot fetch failed for {url!r}: {exc.reason}") from exc


def _build_ffmpeg_snapshot_cmd(url: str) -> list[str]:
    """ffmpeg argv for a settled, downscaled single-JPEG RTSP grab.

    * ``-ss _RTSP_SNAPSHOT_SETTLE_S`` *after* ``-i`` decodes and discards the
      first second of the live stream so the decoder is past a keyframe and
      emitting complete frames before ``-frames:v 1`` captures one — the cure
      for the flat grey/green undecoded-bottom preview cv2's ``read()`` produced
      on these 4K H.265 streams.
    * ``scale=640:-2`` caps the thumbnail width (height even for the encoder).
      Every managed stream here is ≥640 px wide, so this only ever downscales.
    * ``-f mjpeg … pipe:1`` writes the JPEG straight to stdout — no temp file to
      clean up. ``-an`` drops audio the thumbnail never needs.
    Kept pure so a unit test can assert the argv without spawning ffmpeg."""
    return [
        "ffmpeg",
        "-nostdin",
        "-rtsp_transport",
        "tcp",
        "-i",
        url,
        "-ss",
        str(_RTSP_SNAPSHOT_SETTLE_S),
        "-frames:v",
        "1",
        "-an",
        "-vf",
        f"scale={_MAX_WIDTH_PX}:-2",
        "-f",
        "mjpeg",
        "-q:v",
        str(_FFMPEG_MJPEG_QSCALE),
        "pipe:1",
    ]


def _rtsp_frame_grab(
    url: str, timeout_s: float, *, runner: Callable[..., object] = subprocess.run
) -> bytes:
    """Grab one settled JPEG frame from an RTSP stream via ffmpeg.

    ffmpeg — not ``cv2.VideoCapture`` — because a 4K H.265 stream opened mid-GOP
    makes cv2's ``read()`` hand back error-concealment garbage (a flat grey/green
    undecoded bottom) until a keyframe lands, and it does not recover within any
    bounded frame budget; ffmpeg's ``-ss`` settle decodes past the keyframe
    first, then captures a complete frame (see :func:`_build_ffmpeg_snapshot_cmd`).
    This is the same ffmpeg the recorder already relies on, so "the preview grab
    succeeds" tracks "the recording will succeed".

    ``runner`` is the ``subprocess.run``-shaped boundary (production default;
    tests inject a fake so no ffmpeg forks). The subprocess deadline is floored
    at ``_RTSP_GRAB_MIN_TIMEOUT_S`` because the caller's ``timeout_s`` was tuned
    for the old single cv2 read. Any failure — non-zero exit, empty output,
    timeout, ffmpeg missing — raises ``RuntimeError`` with the ffmpeg stderr tail
    so the poller/route reports a diagnostic message and re-enqueues."""
    cmd = _build_ffmpeg_snapshot_cmd(url)
    deadline = max(timeout_s, _RTSP_GRAB_MIN_TIMEOUT_S)
    try:
        result = runner(cmd, capture_output=True, timeout=deadline)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg snapshot grab timed out after {deadline}s for {url!r}") from exc
    except (OSError, ValueError) as exc:  # ffmpeg missing / bad argv
        raise RuntimeError(f"ffmpeg snapshot grab could not run for {url!r}: {exc}") from exc
    stdout = getattr(result, "stdout", b"") or b""
    if getattr(result, "returncode", 1) != 0 or not stdout:
        stderr = (getattr(result, "stderr", b"") or b"").decode("latin-1", "replace")
        tail = (
            stderr.strip().splitlines()[-1]
            if stderr.strip()
            else f"exit {getattr(result, 'returncode', '?')}"
        )
        raise RuntimeError(f"ffmpeg snapshot grab failed for {url!r}: {tail}")
    return bytes(stdout)
