"""Tests for the production snapshot grabber (issue #41).

The grabber dispatches on URL scheme: HTTP/HTTPS go through stdlib
``urllib.request`` (cheap, single GET), RTSP goes through ffmpeg (settle past
a keyframe, capture one frame, encode JPEG to stdout).

Both backends are injectable so these tests never fork ffmpeg or open
sockets — the real ffmpeg / urllib calls are exercised by the manual
verification step on a live camera.
"""

from __future__ import annotations

import subprocess

from client_agent.snapshot import build_snapshot_grabber


class _FakeCompleted:
    """Stand-in for ``subprocess.CompletedProcess`` — the grab reads only
    ``returncode``/``stdout``/``stderr``."""

    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _runner(completed: _FakeCompleted, *, calls: list | None = None):
    """A ``subprocess.run``-shaped fake returning ``completed`` and optionally
    recording each ``(cmd, kwargs)`` it saw."""

    def run(cmd, **kwargs):
        if calls is not None:
            calls.append((cmd, kwargs))
        return completed

    return run


def test_build_ffmpeg_snapshot_cmd_is_settled_single_jpeg() -> None:
    """The grab command must (a) settle past a keyframe via ``-ss`` AFTER ``-i``
    so it captures a fully-decoded frame, (b) take exactly one frame, (c) cap the
    width, and (d) stream one MJPEG to stdout — the shape proven to beat the
    mid-GOP grey/green preview on the operator's 4K H.265 cameras."""
    from client_agent.snapshot import _build_ffmpeg_snapshot_cmd

    cmd = _build_ffmpeg_snapshot_cmd("rtsp://192.168.88.85:554/unicast/c1/s0/live")
    assert cmd[0] == "ffmpeg"
    assert cmd[-1] == "pipe:1"
    i = cmd.index("-i")
    assert cmd[i + 1] == "rtsp://192.168.88.85:554/unicast/c1/s0/live"
    # -ss must come AFTER -i (decode-and-discard the live stream), not before.
    assert cmd.index("-ss") > i
    assert cmd[cmd.index("-ss") + 1] == "1.0"
    assert cmd[cmd.index("-frames:v") + 1] == "1"
    assert "-rtsp_transport" in cmd and "tcp" in cmd
    assert cmd[cmd.index("-f") + 1] == "mjpeg"
    assert cmd[cmd.index("-vf") + 1] == "scale=640:-2"


def test_rtsp_frame_grab_returns_stdout_jpeg_bytes() -> None:
    """A clean ffmpeg run (exit 0, JPEG on stdout) yields those bytes verbatim —
    they go straight to the presigned R2 PUT / HTTP response."""
    from client_agent.snapshot import _rtsp_frame_grab

    jpeg = b"\xff\xd8\xff\xe0" + b"fake-jpeg-body"
    out = _rtsp_frame_grab("rtsp://cam/stream", 5.0, runner=_runner(_FakeCompleted(0, jpeg)))
    assert out == jpeg


def test_rtsp_frame_grab_floors_the_subprocess_deadline() -> None:
    """The caller's ``timeout_s`` (tuned for the old single cv2 read) must not
    starve the settle-then-grab: the ffmpeg deadline is floored so a too-small
    caller value can't spuriously fail every grab."""
    from client_agent.snapshot import _RTSP_GRAB_MIN_TIMEOUT_S, _rtsp_frame_grab

    calls: list = []
    _rtsp_frame_grab(
        "rtsp://cam/stream", 1.0, runner=_runner(_FakeCompleted(0, b"jpeg"), calls=calls)
    )
    assert calls[0][1]["timeout"] == _RTSP_GRAB_MIN_TIMEOUT_S


def test_rtsp_frame_grab_raises_on_nonzero_exit_with_stderr_tail() -> None:
    """A failed ffmpeg (non-zero exit) surfaces its stderr tail so the poller's
    failure message is diagnostic (which camera, why), not a bare exit code."""
    from client_agent.snapshot import _rtsp_frame_grab

    completed = _FakeCompleted(1, b"", b"[tcp] connection refused\nConversion failed!")
    try:
        _rtsp_frame_grab("rtsp://cam/stream", 5.0, runner=_runner(completed))
    except RuntimeError as exc:
        assert "Conversion failed!" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on non-zero exit")


def test_rtsp_frame_grab_raises_on_empty_output() -> None:
    """Exit 0 but no bytes (ffmpeg produced no frame) must raise, not return an
    empty JPEG that the UI would render as a broken image."""
    from client_agent.snapshot import _rtsp_frame_grab

    try:
        _rtsp_frame_grab("rtsp://cam/stream", 5.0, runner=_runner(_FakeCompleted(0, b"")))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on empty output")


def test_rtsp_frame_grab_raises_on_timeout() -> None:
    """A wedged stream (ffmpeg past its deadline) becomes a clean RuntimeError,
    not a leaked ``TimeoutExpired`` that would crash the poll loop."""
    from client_agent.snapshot import _rtsp_frame_grab

    def timeout_runner(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 8.0))

    try:
        _rtsp_frame_grab("rtsp://cam/stream", 5.0, runner=timeout_runner)
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on timeout")


def test_rtsp_frame_grab_raises_when_ffmpeg_missing() -> None:
    """No ffmpeg on PATH (OSError) → RuntimeError, so a mis-provisioned host
    fails with a readable message instead of an uncaught FileNotFoundError."""
    from client_agent.snapshot import _rtsp_frame_grab

    def missing(cmd, **kwargs):
        raise FileNotFoundError("ffmpeg")

    try:
        _rtsp_frame_grab("rtsp://cam/stream", 5.0, runner=missing)
    except RuntimeError as exc:
        assert "could not run" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when ffmpeg missing")


def test_http_url_dispatches_to_http_fetcher() -> None:
    """A vendor HTTP snapshot URL routes to the HTTP fetcher, not cv2.
    Saves an RTSP handshake for cameras whose ONVIF GetSnapshotUri
    surfaced a direct JPEG endpoint."""
    http_calls: list[tuple[str, float]] = []
    rtsp_calls: list[tuple[str, float]] = []

    def fake_http(url: str, timeout_s: float, _v: str) -> bytes:
        http_calls.append((url, timeout_s))
        return b"http-jpeg"

    def fake_rtsp(url: str, timeout_s: float, _v: str) -> bytes:
        rtsp_calls.append((url, timeout_s))
        return b"rtsp-jpeg"

    grabber = build_snapshot_grabber(http_fetcher=fake_http, rtsp_grabber=fake_rtsp)

    out = grabber("http://192.168.1.10/snapshot.jpg", 5.0, "thumbnail")

    assert out == b"http-jpeg"
    assert http_calls == [("http://192.168.1.10/snapshot.jpg", 5.0)]
    assert rtsp_calls == []


def test_https_url_dispatches_to_http_fetcher() -> None:
    """``https://`` is also HTTP (with TLS) — must route through the
    HTTP fetcher, not cv2."""
    http_calls: list[str] = []

    def fake_http(url: str, _t: float, _v: str) -> bytes:
        http_calls.append(url)
        return b"https-jpeg"

    grabber = build_snapshot_grabber(
        http_fetcher=fake_http,
        rtsp_grabber=lambda _u, _t, _v: (_ for _ in ()).throw(AssertionError("cv2 hit")),
    )

    out = grabber("https://cam.example.com/snap", 5.0, "thumbnail")

    assert out == b"https-jpeg"
    assert http_calls == ["https://cam.example.com/snap"]


def test_rtsp_url_dispatches_to_rtsp_grabber() -> None:
    """An ``rtsp://`` URL routes to the cv2-backed grabber. This is the
    fallback when the camera doesn't expose an HTTP snapshot endpoint."""
    rtsp_calls: list[tuple[str, float]] = []

    def fake_rtsp(url: str, timeout_s: float, _v: str) -> bytes:
        rtsp_calls.append((url, timeout_s))
        return b"rtsp-jpeg"

    grabber = build_snapshot_grabber(
        http_fetcher=lambda _u, _t, _v: (_ for _ in ()).throw(AssertionError("http hit")),
        rtsp_grabber=fake_rtsp,
    )

    out = grabber("rtsp://192.168.1.10:554/Streaming/Channels/101", 5.0, "thumbnail")

    assert out == b"rtsp-jpeg"
    assert rtsp_calls == [("rtsp://192.168.1.10:554/Streaming/Channels/101", 5.0)]


def test_grabber_propagates_underlying_errors() -> None:
    """Whatever the underlying fetcher raises must surface unchanged —
    the route's try/except converts it into a 503 with the message in
    the body."""

    def bad_rtsp(_url: str, _t: float, _v: str) -> bytes:
        raise RuntimeError("cv2.VideoCapture failed to open")

    grabber = build_snapshot_grabber(
        http_fetcher=lambda _u, _t, _v: b"unused",
        rtsp_grabber=bad_rtsp,
    )

    try:
        grabber("rtsp://offline.local/stream", 5.0, "thumbnail")
    except RuntimeError as exc:
        assert "cv2.VideoCapture" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
