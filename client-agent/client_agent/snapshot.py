"""Per-camera snapshot grabber for the Flask UI (issue #41).

Two backends, dispatched on URL scheme:

* HTTP / HTTPS — vendor's native ONVIF ``GetSnapshotUri`` (single GET,
  no RTSP handshake). Stdlib ``urllib.request`` — no third-party HTTP
  client needed for a one-shot JPEG fetch.
* RTSP — cv2.VideoCapture opens the stream, reads one frame, encodes
  JPEG q=70. Used when the camera has no HTTP snapshot endpoint, or in
  platform mode where the heartbeat config only carries ``rtsp_url``.

Both backends are injectable via :func:`build_snapshot_grabber` so unit
tests never touch the system boundary. The defaults (:func:`_http_fetch`
and :func:`_rtsp_frame_grab`) wrap urllib + cv2 with the timeout and
JPEG quality the issue specifies.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Scale to <= 640 px wide — issue #41 spec. Wider thumbs would just be
# wasted bytes on a "My cameras" preview tile.
_MAX_WIDTH_PX = 640
# JPEG quality 70 — the issue's spec. Visually fine for a thumbnail,
# significantly smaller than q=95.
_JPEG_QUALITY = 70

SnapshotGrabberFn = Callable[[str, float], bytes]


def build_snapshot_grabber(
    *,
    http_fetcher: SnapshotGrabberFn | None = None,
    rtsp_grabber: SnapshotGrabberFn | None = None,
) -> SnapshotGrabberFn:
    """Build a URL-scheme-dispatching snapshot grabber.

    Both backends default to the production implementations (urllib for
    HTTP, cv2 for RTSP). Tests inject in-memory fakes — neither default
    touches the network at import time, so a unit test that never calls
    the grabber pays no cv2 cost."""
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


def _rtsp_frame_grab(url: str, timeout_s: float) -> bytes:
    """Open an RTSP URL with cv2, read one frame, encode as JPEG q=70.

    Two cv2 capture properties cap the wallclock cost: OPEN_TIMEOUT_MSEC
    (TCP/RTSP handshake) and READ_TIMEOUT_MSEC (per-packet read). Both
    set to the same ceiling so a dead camera can't pin the request
    thread past the route-level 5 s budget. ``cv2.CAP_FFMPEG`` forces
    the FFmpeg backend — the GStreamer one isn't built into the
    headless wheel, and the default backend on Linux is platform-
    dependent.

    Released in a finally so a partial open doesn't leak the underlying
    AVFormatContext."""
    import cv2  # local import: defers the 50 MB shared-lib load to first call

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        timeout_ms = int(timeout_s * 1000)
        # These properties are no-ops on builds that don't recognize
        # them — harmless. The headless wheel built against FFmpeg 4.x
        # respects both.
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open {url!r}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"cv2.VideoCapture.read() returned no frame from {url!r}")
        height, width = frame.shape[:2]
        if width > _MAX_WIDTH_PX:
            scale = _MAX_WIDTH_PX / float(width)
            new_size = (_MAX_WIDTH_PX, int(height * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"cv2.imencode failed for {url!r}")
        return bytes(buf)
    finally:
        cap.release()
