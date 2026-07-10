"""Tests for the production snapshot grabber (issue #41).

The grabber dispatches on URL scheme: HTTP/HTTPS go through stdlib
``urllib.request`` (cheap, single GET), RTSP goes through OpenCV
(``cv2.VideoCapture`` opens a session, reads one frame, encodes JPEG).

Both backends are injectable so these tests never touch cv2 or open
sockets — the real cv2 / urllib calls are exercised by the manual
verification step in the issue's acceptance criteria.
"""

from __future__ import annotations

from client_agent.snapshot import build_snapshot_grabber


class _FakeFrame:
    """Minimal stand-in for a cv2/numpy frame — the warm-up selector only
    calls ``.std()``, so tests need neither numpy nor cv2."""

    def __init__(self, std_value: float) -> None:
        self._std = std_value

    def std(self) -> float:
        return self._std


def _reader(frames: list[tuple[bool, object]]):
    """Build a ``read_fn`` yielding successive ``(ok, frame)`` from ``frames``
    then ``(False, None)`` once exhausted — mirrors ``cv2.VideoCapture.read``."""
    it = iter(frames)

    def read() -> tuple[bool, object]:
        try:
            return next(it)
        except StopIteration:
            return (False, None)

    return read


def test_select_snapshot_frame_skips_grey_warmup_frames() -> None:
    """The first frame(s) off an H.264/H.265 RTSP stream are a uniform grey
    placeholder (std ~0.3) the decoder emits before the first keyframe; the
    selector must skip them and return the first frame with real detail."""
    from client_agent.snapshot import _select_snapshot_frame

    grey1, grey2 = _FakeFrame(0.3), _FakeFrame(0.5)
    real = _FakeFrame(58.0)
    got = _select_snapshot_frame(
        _reader([(True, grey1), (True, grey2), (True, real), (True, _FakeFrame(60.0))]),
        max_frames=15,
        min_stddev=8.0,
    )
    assert got is real


def test_select_snapshot_frame_falls_back_to_last_when_all_uniform() -> None:
    """A genuinely flat scene (blank wall, covered lens) never clears the
    threshold — the selector must still return the last real frame it read,
    not None, so the preview shows the actual (flat) view."""
    from client_agent.snapshot import _select_snapshot_frame

    last = _FakeFrame(1.0)
    got = _select_snapshot_frame(
        _reader([(True, _FakeFrame(0.3)), (True, _FakeFrame(0.4)), (True, last)]),
        max_frames=15,
        min_stddev=8.0,
    )
    assert got is last


def test_select_snapshot_frame_returns_none_when_no_frame_read() -> None:
    """A stream that yields no frame at all → None, so the caller raises the
    'no frame' RuntimeError instead of encoding a missing frame."""
    from client_agent.snapshot import _select_snapshot_frame

    assert _select_snapshot_frame(_reader([]), max_frames=15, min_stddev=8.0) is None


def test_select_snapshot_frame_is_bounded_by_max_frames() -> None:
    """An endlessly-grey stream must not spin — the read count is capped at
    ``max_frames`` (bounds HEVC 4K decode cost), returning the last frame."""
    from client_agent.snapshot import _select_snapshot_frame

    reads = 0

    def read() -> tuple[bool, object]:
        nonlocal reads
        reads += 1
        return (True, _FakeFrame(0.3))  # forever grey

    got = _select_snapshot_frame(read, max_frames=5, min_stddev=8.0)
    assert reads == 5
    assert got is not None


def test_http_url_dispatches_to_http_fetcher() -> None:
    """A vendor HTTP snapshot URL routes to the HTTP fetcher, not cv2.
    Saves an RTSP handshake for cameras whose ONVIF GetSnapshotUri
    surfaced a direct JPEG endpoint."""
    http_calls: list[tuple[str, float]] = []
    rtsp_calls: list[tuple[str, float]] = []

    def fake_http(url: str, timeout_s: float) -> bytes:
        http_calls.append((url, timeout_s))
        return b"http-jpeg"

    def fake_rtsp(url: str, timeout_s: float) -> bytes:
        rtsp_calls.append((url, timeout_s))
        return b"rtsp-jpeg"

    grabber = build_snapshot_grabber(http_fetcher=fake_http, rtsp_grabber=fake_rtsp)

    out = grabber("http://192.168.1.10/snapshot.jpg", 5.0)

    assert out == b"http-jpeg"
    assert http_calls == [("http://192.168.1.10/snapshot.jpg", 5.0)]
    assert rtsp_calls == []


def test_https_url_dispatches_to_http_fetcher() -> None:
    """``https://`` is also HTTP (with TLS) — must route through the
    HTTP fetcher, not cv2."""
    http_calls: list[str] = []

    def fake_http(url: str, _t: float) -> bytes:
        http_calls.append(url)
        return b"https-jpeg"

    grabber = build_snapshot_grabber(
        http_fetcher=fake_http,
        rtsp_grabber=lambda _u, _t: (_ for _ in ()).throw(AssertionError("cv2 hit")),
    )

    out = grabber("https://cam.example.com/snap", 5.0)

    assert out == b"https-jpeg"
    assert http_calls == ["https://cam.example.com/snap"]


def test_rtsp_url_dispatches_to_rtsp_grabber() -> None:
    """An ``rtsp://`` URL routes to the cv2-backed grabber. This is the
    fallback when the camera doesn't expose an HTTP snapshot endpoint."""
    rtsp_calls: list[tuple[str, float]] = []

    def fake_rtsp(url: str, timeout_s: float) -> bytes:
        rtsp_calls.append((url, timeout_s))
        return b"rtsp-jpeg"

    grabber = build_snapshot_grabber(
        http_fetcher=lambda _u, _t: (_ for _ in ()).throw(AssertionError("http hit")),
        rtsp_grabber=fake_rtsp,
    )

    out = grabber("rtsp://192.168.1.10:554/Streaming/Channels/101", 5.0)

    assert out == b"rtsp-jpeg"
    assert rtsp_calls == [("rtsp://192.168.1.10:554/Streaming/Channels/101", 5.0)]


def test_grabber_propagates_underlying_errors() -> None:
    """Whatever the underlying fetcher raises must surface unchanged —
    the route's try/except converts it into a 503 with the message in
    the body."""

    def bad_rtsp(_url: str, _t: float) -> bytes:
        raise RuntimeError("cv2.VideoCapture failed to open")

    grabber = build_snapshot_grabber(
        http_fetcher=lambda _u, _t: b"unused",
        rtsp_grabber=bad_rtsp,
    )

    try:
        grabber("rtsp://offline.local/stream", 5.0)
    except RuntimeError as exc:
        assert "cv2.VideoCapture" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
