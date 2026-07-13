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

# Decoder warm-up: the first frame(s) an FFmpeg RTSP session emits are
# frequently a uniform grey placeholder produced before the first keyframe
# (IDR) decodes — a 4K BCS/Hikvision H.265 main stream shows per-pixel stddev
# ~0.3 for frames 0-1 then ~58 from frame 2. Grabbing frame 0 (as the naive
# read did) uploads that grey placeholder as the preview. Read up to
# ``_SNAPSHOT_WARMUP_MAX_FRAMES`` until a frame clears ``_SNAPSHOT_MIN_STDDEV``.
# The cap keeps HEVC 4K decode cost bounded and a genuinely flat scene from
# spinning; 8.0 sits well above grey (~0.3) and below any real content (~58),
# so even a dim night scene clears it.
_SNAPSHOT_WARMUP_MAX_FRAMES = 15
_SNAPSHOT_MIN_STDDEV = 8.0

# Partial-decode guard: a 4K H.265 IDR sometimes surfaces *half-decoded* — the
# top slices carry real content but the bottom rows are still the decoder's
# uninitialised fill (renders as a flat green block, the "most green" preview
# seen on 192.168.88.84). Such a frame's GLOBAL stddev is high (the real top
# dominates) so the grey guard above waves it through. Catch it by measuring the
# bottom strip alone: an undecoded block is near-uniform (stddev ~0.5), whereas
# any real scene's ground/floor keeps sensor+texture noise well above this. The
# raster-scan decode order means undecoded rows are always at the BOTTOM.
_SNAPSHOT_BOTTOM_FRACTION = 0.25
_SNAPSHOT_MIN_BOTTOM_STDDEV = 3.0

SnapshotGrabberFn = Callable[[str, float], bytes]


def _bottom_region_stddev(
    frame: object, bottom_fraction: float = _SNAPSHOT_BOTTOM_FRACTION
) -> float:
    """Per-pixel stddev of the bottom ``bottom_fraction`` of ``frame``.

    Duck-typed on the numpy-array surface (``.shape`` + slicing + ``.std()``),
    so a real cv2 frame and the test fake both work. Isolating the bottom strip
    is what distinguishes a *partial-decode* frame (real top, flat undecoded
    bottom) from a fully-decoded one — the global stddev can't, because the real
    top swamps it."""
    height = frame.shape[0]  # type: ignore[attr-defined]
    start = int(height * (1.0 - bottom_fraction))
    strip = frame[start:, :, :]  # type: ignore[index]
    return float(strip.std())  # type: ignore[attr-defined]


def _select_snapshot_frame(
    read_fn: Callable[[], tuple[bool, object]],
    *,
    max_frames: int = _SNAPSHOT_WARMUP_MAX_FRAMES,
    min_stddev: float = _SNAPSHOT_MIN_STDDEV,
    min_bottom_stddev: float = _SNAPSHOT_MIN_BOTTOM_STDDEV,
) -> object | None:
    """Read frames until one is fully decoded, skipping warm-up + partial frames.

    ``read_fn`` is ``cv2.VideoCapture.read`` (or a fake): returns
    ``(ok, frame)`` where ``frame`` exposes ``.std()``, ``.shape`` and slicing
    (a numpy ndarray does). A frame is accepted when it clears BOTH gates:

    * global stddev ≥ ``min_stddev`` — skips the uniform grey pre-keyframe
      placeholder;
    * bottom-strip stddev ≥ ``min_bottom_stddev`` — skips a half-decoded IDR
      whose bottom rows are a flat (green) undecoded block.

    If no frame clears both within ``max_frames`` (a genuinely flat scene, a
    short stream, or a stream that never fully decodes), the best frame seen —
    the one with the most detail in its weakest region — is returned rather than
    just the last, so a trailing partial frame can't win the fallback. Returns
    ``None`` when ``read_fn`` never yields a frame."""
    best: object | None = None
    best_score = -1.0
    for _ in range(max_frames):
        ok, candidate = read_fn()
        if not ok or candidate is None:
            break
        global_std = float(candidate.std())  # type: ignore[attr-defined]
        bottom_std = _bottom_region_stddev(candidate)
        if global_std >= min_stddev and bottom_std >= min_bottom_stddev:
            return candidate
        # Rank fallbacks by their weakest region: a partial frame scores ~0 on
        # its flat bottom, a grey frame ~0 globally, so a more-complete frame
        # always outranks them.
        score = min(global_std, bottom_std)
        if score > best_score:
            best_score = score
            best = candidate
    return best


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
        # Skip the grey pre-keyframe warm-up frames (see _select_snapshot_frame)
        # rather than grabbing frame 0, which uploaded a blank grey preview.
        frame = _select_snapshot_frame(cap.read)
        if frame is None:
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
