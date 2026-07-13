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
    """Minimal stand-in for a cv2/numpy frame — the selector calls ``.std()``
    (global) plus ``.shape`` + slicing (to measure the bottom strip), so tests
    need neither numpy nor cv2. ``bottom_std`` defaults to the global value, so
    a frame that is uniform top-to-bottom needs only one number; a *partial-
    decode* frame passes a high global ``std_value`` with a low ``bottom_std``."""

    def __init__(
        self,
        std_value: float,
        *,
        bottom_std: float | None = None,
        shape: tuple[int, int, int] = (1080, 1920, 3),
    ) -> None:
        self._std = std_value
        self._bottom_std = std_value if bottom_std is None else bottom_std
        self.shape = shape

    def std(self) -> float:
        return self._std

    def __getitem__(self, _key: object) -> _FakeFrame:
        # The selector only ever slices the bottom strip; return a frame whose
        # global std IS this frame's bottom_std so the strip's ``.std()`` matches.
        return _FakeFrame(self._bottom_std, shape=self.shape)


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


def test_select_snapshot_frame_skips_partial_decode_green_bottom() -> None:
    """A half-decoded 4K H.265 IDR (real top, flat green undecoded bottom) has a
    HIGH global stddev — it clears the grey guard — but a near-zero bottom-strip
    stddev. The selector must skip it and return the next fully-decoded frame,
    not upload the "most green" preview seen on 192.168.88.84."""
    from client_agent.snapshot import _select_snapshot_frame

    partial = _FakeFrame(40.0, bottom_std=0.4)  # detailed top, flat green bottom
    full = _FakeFrame(52.0, bottom_std=47.0)  # decoded top-to-bottom
    got = _select_snapshot_frame(
        _reader([(True, partial), (True, full)]),
        max_frames=15,
        min_stddev=8.0,
        min_bottom_stddev=3.0,
    )
    assert got is full


def test_select_snapshot_frame_fallback_prefers_most_complete_frame() -> None:
    """When nothing fully decodes within the cap, the fallback must be the
    most-complete frame (highest detail in its weakest region), NOT merely the
    last read — so a trailing partial-decode frame can't win the preview."""
    from client_agent.snapshot import _select_snapshot_frame

    grey = _FakeFrame(0.3, bottom_std=0.3)
    partial = _FakeFrame(40.0, bottom_std=0.5)  # high global, flat bottom
    best_partial = _FakeFrame(45.0, bottom_std=2.5)  # closest to fully decoded
    trailing_partial = _FakeFrame(41.0, bottom_std=0.6)  # last, but still green
    got = _select_snapshot_frame(
        _reader([(True, grey), (True, partial), (True, best_partial), (True, trailing_partial)]),
        max_frames=15,
        min_stddev=8.0,
        min_bottom_stddev=3.0,
    )
    assert got is best_partial


def test_select_snapshot_frame_accepts_fully_decoded_first_frame() -> None:
    """No-regression: a frame that is detailed top-to-bottom (real content,
    fully decoded) is accepted immediately — the bottom-strip guard must not
    reject good frames."""
    from client_agent.snapshot import _select_snapshot_frame

    full = _FakeFrame(58.0, bottom_std=55.0)
    got = _select_snapshot_frame(
        _reader([(True, full), (True, _FakeFrame(59.0, bottom_std=56.0))]),
        max_frames=15,
        min_stddev=8.0,
        min_bottom_stddev=3.0,
    )
    assert got is full


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
