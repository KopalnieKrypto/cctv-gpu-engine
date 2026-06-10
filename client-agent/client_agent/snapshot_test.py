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
